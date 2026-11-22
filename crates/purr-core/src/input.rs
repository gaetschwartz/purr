use ffmpeg_next as ffmpeg;
use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::Arc;
use tokio::sync::Notify;

pub const DEFAULT_BUFFER_CAPACITY: usize = 8 * 1024 * 1024;

struct BufferState {
    bytes: VecDeque<u8>,
    eof: bool,
    failure: Option<String>,
    closed: bool,
}

pub struct AsyncStreamBuffer {
    state: Mutex<BufferState>,
    readable: Condvar,
    space: Notify,
    capacity: usize,
}

impl AsyncStreamBuffer {
    pub fn new() -> Arc<Self> {
        Self::with_capacity(DEFAULT_BUFFER_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(BufferState {
                bytes: VecDeque::new(),
                eof: false,
                failure: None,
                closed: false,
            }),
            readable: Condvar::new(),
            space: Notify::new(),
            capacity: capacity.max(1),
        })
    }

    pub async fn push(&self, data: &[u8]) -> bool {
        let mut rest = data;
        loop {
            {
                let mut state = self.state.lock();
                if state.closed {
                    return false;
                }
                let free = self.capacity.saturating_sub(state.bytes.len());
                let (now, later) = rest.split_at(free.min(rest.len()));
                if !now.is_empty() {
                    state.bytes.extend(now);
                    self.readable.notify_all();
                }
                rest = later;
                if rest.is_empty() {
                    return true;
                }
            }
            self.space.notified().await;
        }
    }

    pub fn set_eof(&self) {
        self.state.lock().eof = true;
        self.readable.notify_all();
    }

    pub fn fail(&self, message: String) {
        self.state.lock().failure = Some(message);
        self.readable.notify_all();
    }

    pub fn failure(&self) -> Option<String> {
        self.state.lock().failure.clone()
    }

    pub fn close(&self) {
        {
            let mut state = self.state.lock();
            state.closed = true;
            state.bytes = VecDeque::new();
        }
        self.readable.notify_all();
        self.space.notify_one();
    }

    pub fn read(&self, out: &mut [u8]) -> i32 {
        if out.is_empty() {
            return 0;
        }
        let mut state = self.state.lock();
        loop {
            if !state.bytes.is_empty() {
                let count = out.len().min(state.bytes.len());
                let Ok(returned) = i32::try_from(count) else {
                    return ffmpeg_next::sys::AVERROR(ffmpeg_next::sys::EINVAL);
                };
                for (slot, byte) in out.iter_mut().zip(state.bytes.drain(..count)) {
                    *slot = byte;
                }
                drop(state);
                self.space.notify_one();
                return returned;
            }
            if state.eof || state.failure.is_some() || state.closed {
                return ffmpeg_next::sys::AVERROR_EOF;
            }
            self.readable.wait(&mut state);
        }
    }
}

unsafe extern "C" fn read_packet(opaque: *mut c_void, buf: *mut u8, buf_size: i32) -> i32 {
    let Ok(len) = usize::try_from(buf_size) else {
        return ffmpeg_next::sys::AVERROR_EOF;
    };
    let buffer = unsafe { &*opaque.cast::<AsyncStreamBuffer>() };
    let slice = unsafe { std::slice::from_raw_parts_mut(buf, len) };
    buffer.read(slice)
}

pub struct AsyncCustomInput {
    buffer: Arc<AsyncStreamBuffer>,
    context: ffmpeg::format::context::Input,
}

impl Drop for AsyncCustomInput {
    fn drop(&mut self) {
        self.buffer.close();
    }
}

impl AsyncCustomInput {
    /// Creates the input context - this is blocking and should be called in spawn_blocking
    pub fn create_blocking(buffer: Arc<AsyncStreamBuffer>) -> Result<Self, ffmpeg::Error> {
        let input = Self::open(&buffer);
        if input.is_err() {
            buffer.close();
        }
        input
    }

    fn open(buffer: &Arc<AsyncStreamBuffer>) -> Result<Self, ffmpeg::Error> {
        unsafe {
            let buffer_size = 32768;
            let avio_buffer = ffmpeg_next::sys::av_malloc(buffer_size).cast::<u8>();

            if avio_buffer.is_null() {
                return Err(ffmpeg::Error::Bug);
            }

            let opaque = Arc::into_raw(Arc::clone(buffer)) as *mut c_void;

            let avio_ctx = ffmpeg_next::sys::avio_alloc_context(
                avio_buffer,
                buffer_size as i32,
                0,
                opaque,
                Some(read_packet),
                None,
                None,
            );

            if avio_ctx.is_null() {
                ffmpeg_next::sys::av_free(avio_buffer.cast::<c_void>());
                Arc::from_raw(opaque as *const AsyncStreamBuffer);
                return Err(ffmpeg::Error::Bug);
            }

            let mut format_ctx = ffmpeg_next::sys::avformat_alloc_context();
            if format_ctx.is_null() {
                ffmpeg_next::sys::av_free(avio_ctx.cast::<c_void>());
                ffmpeg_next::sys::av_free(avio_buffer.cast::<c_void>());
                Arc::from_raw(opaque as *const AsyncStreamBuffer);
                return Err(ffmpeg::Error::Bug);
            }

            (*format_ctx).pb = avio_ctx;
            (*format_ctx).flags |= ffmpeg_next::sys::AVFMT_FLAG_CUSTOM_IO;

            let ret = ffmpeg_next::sys::avformat_open_input(
                &raw mut format_ctx,
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );

            if ret < 0 {
                ffmpeg_next::sys::avformat_free_context(format_ctx);
                Arc::from_raw(opaque as *const AsyncStreamBuffer);
                return Err(ffmpeg::Error::from(ret));
            }

            // This is the blocking call - it will wait for enough data
            let ret = ffmpeg_next::sys::avformat_find_stream_info(format_ctx, std::ptr::null_mut());

            if ret < 0 {
                ffmpeg_next::sys::avformat_close_input(&raw mut format_ctx);
                Arc::from_raw(opaque as *const AsyncStreamBuffer);
                return Err(ffmpeg::Error::from(ret));
            }

            Ok(Self {
                buffer: Arc::clone(buffer),
                context: ffmpeg::format::context::Input::wrap(format_ctx),
            })
        }
    }

    /// Async wrapper that runs the blocking creation in a blocking task
    pub async fn create(buffer: Arc<AsyncStreamBuffer>) -> Result<Self, ffmpeg::Error> {
        tokio::task::spawn_blocking(move || Self::create_blocking(buffer))
            .await
            .map_err(|_| ffmpeg::Error::External)?
    }

    pub fn input(&self) -> &ffmpeg::format::context::Input {
        &self.context
    }

    pub fn input_mut(&mut self) -> &mut ffmpeg::format::context::Input {
        &mut self.context
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use std::time::Duration;

    const EOF: i32 = ffmpeg_next::sys::AVERROR_EOF;

    #[tokio::test]
    async fn reader_blocks_until_data_is_pushed() {
        let buffer = AsyncStreamBuffer::with_capacity(16);
        let reader = {
            let buffer = Arc::clone(&buffer);
            std::thread::spawn(move || {
                let mut out = [0u8; 8];
                (buffer.read(&mut out), out)
            })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!reader.is_finished());

        assert!(buffer.push(b"ab").await);
        let (count, out) = reader.join().unwrap();
        assert_eq!(count, 2);
        assert_eq!(&out[..2], b"ab");
    }

    #[tokio::test]
    async fn eof_after_drain() {
        let buffer = AsyncStreamBuffer::with_capacity(16);
        assert!(buffer.push(b"xyz").await);
        buffer.set_eof();

        let mut out = [0u8; 8];
        assert_eq!(buffer.read(&mut out), 3);
        assert_eq!(buffer.read(&mut out), EOF);
    }

    #[tokio::test]
    async fn failure_is_reported_after_the_buffered_bytes() {
        let buffer = AsyncStreamBuffer::with_capacity(16);
        assert!(buffer.push(b"abc").await);
        buffer.fail("boom".to_owned());

        let mut out = [0u8; 8];
        assert_eq!(buffer.read(&mut out), 3);
        assert_eq!(buffer.read(&mut out), EOF);
        assert_eq!(buffer.failure(), Some("boom".to_owned()));
    }

    #[tokio::test]
    async fn push_waits_while_full() {
        let buffer = AsyncStreamBuffer::with_capacity(4);
        assert!(buffer.push(b"abcd").await);

        let pusher = {
            let buffer = Arc::clone(&buffer);
            tokio::spawn(async move { buffer.push(b"ef").await })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!pusher.is_finished());

        let mut out = [0u8; 2];
        assert_eq!(buffer.read(&mut out), 2);
        assert!(
            tokio::time::timeout(Duration::from_secs(5), pusher)
                .await
                .unwrap()
                .unwrap()
        );
    }

    #[tokio::test]
    async fn a_push_larger_than_the_capacity_goes_through_in_parts() {
        let buffer = AsyncStreamBuffer::with_capacity(4);
        let pusher = {
            let buffer = Arc::clone(&buffer);
            tokio::spawn(async move { buffer.push(b"abcdefghij").await })
        };
        let reader = {
            let buffer = Arc::clone(&buffer);
            tokio::task::spawn_blocking(move || {
                let mut collected = Vec::new();
                let mut out = [0u8; 3];
                while collected.len() < 10 {
                    let count = buffer.read(&mut out);
                    assert!(count > 0);
                    collected.extend_from_slice(&out[..count as usize]);
                }
                collected
            })
        };
        let collected = tokio::time::timeout(Duration::from_secs(5), reader)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(collected, b"abcdefghij");
        assert!(pusher.await.unwrap());
    }

    #[tokio::test]
    async fn push_returns_false_when_the_reader_is_gone() {
        let buffer = AsyncStreamBuffer::with_capacity(4);
        assert!(buffer.push(b"abcd").await);

        let pusher = {
            let buffer = Arc::clone(&buffer);
            tokio::spawn(async move { buffer.push(b"ef").await })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        buffer.close();

        assert!(
            !tokio::time::timeout(Duration::from_secs(5), pusher)
                .await
                .unwrap()
                .unwrap()
        );
        assert!(!buffer.push(b"g").await);
    }
}
