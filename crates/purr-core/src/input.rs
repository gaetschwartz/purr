use ffmpeg_next as ffmpeg;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

pub struct AsyncStreamBuffer {
    buffer: Mutex<VecDeque<u8>>,
    eof: Mutex<bool>,
    notify: Notify,
}

impl AsyncStreamBuffer {
    pub fn new() -> Arc<Self> {
        Arc::new(AsyncStreamBuffer {
            buffer: Mutex::new(VecDeque::new()),
            eof: Mutex::new(false),
            notify: Notify::new(),
        })
    }

    pub fn write(&self, data: &[u8]) {
        let mut buffer = self.buffer.lock().unwrap();
        buffer.extend(data);
        self.notify.notify_one();
    }

    pub fn set_eof(&self) {
        *self.eof.lock().unwrap() = true;
        self.notify.notify_one();
    }

    pub fn try_read(&self, out: &mut [u8]) -> i32 {
        let mut buffer = self.buffer.lock().unwrap();

        if buffer.is_empty() {
            if *self.eof.lock().unwrap() {
                return ffmpeg_next::sys::AVERROR_EOF;
            }
            // Return EAGAIN to indicate "try again later"
            return ffmpeg_next::sys::AVERROR(ffmpeg_next::sys::EAGAIN);
        }

        let to_read = out.len().min(buffer.len());
        for i in 0..to_read {
            out[i] = buffer.pop_front().unwrap();
        }

        to_read as i32
    }

    fn blocking_read(&self, out: &mut [u8]) -> i32 {
        loop {
            let result = self.try_read(out);
            if result != ffmpeg_next::sys::AVERROR(ffmpeg_next::sys::EAGAIN) {
                return result;
            }
            // Block until notified
            let _ = self.notify.notified();
        }
    }
}

unsafe extern "C" fn read_packet(opaque: *mut c_void, buf: *mut u8, buf_size: i32) -> i32 {
    let stream_buffer = Arc::from_raw(opaque as *const AsyncStreamBuffer);
    let result = {
        let slice = std::slice::from_raw_parts_mut(buf, buf_size as usize);
        stream_buffer.blocking_read(slice)
    };
    std::mem::forget(stream_buffer);
    result
}

pub struct AsyncCustomInput {
    _buffer: Arc<AsyncStreamBuffer>,
    context: ffmpeg::format::context::Input,
}

impl AsyncCustomInput {
    /// Creates the input context - this is blocking and should be called in spawn_blocking
    pub fn create_blocking(buffer: Arc<AsyncStreamBuffer>) -> Result<Self, ffmpeg::Error> {
        unsafe {
            let buffer_size = 32768;
            let avio_buffer = ffmpeg_next::sys::av_malloc(buffer_size) as *mut u8;

            if avio_buffer.is_null() {
                return Err(ffmpeg::Error::Bug);
            }

            let opaque = Arc::into_raw(buffer.clone()) as *mut c_void;

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
                ffmpeg_next::sys::av_free(avio_buffer as *mut c_void);
                Arc::from_raw(opaque as *const AsyncStreamBuffer);
                return Err(ffmpeg::Error::Bug);
            }

            let mut format_ctx = ffmpeg_next::sys::avformat_alloc_context();
            if format_ctx.is_null() {
                ffmpeg_next::sys::av_free(avio_ctx as *mut c_void);
                ffmpeg_next::sys::av_free(avio_buffer as *mut c_void);
                Arc::from_raw(opaque as *const AsyncStreamBuffer);
                return Err(ffmpeg::Error::Bug);
            }

            (*format_ctx).pb = avio_ctx;
            (*format_ctx).flags |= ffmpeg_next::sys::AVFMT_FLAG_CUSTOM_IO;

            let ret = ffmpeg_next::sys::avformat_open_input(
                &mut format_ctx,
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
                ffmpeg_next::sys::avformat_close_input(&mut format_ctx);
                Arc::from_raw(opaque as *const AsyncStreamBuffer);
                return Err(ffmpeg::Error::from(ret));
            }

            Ok(AsyncCustomInput {
                _buffer: buffer,
                context: ffmpeg::format::context::Input::wrap(format_ctx),
            })
        }
    }

    /// Async wrapper that runs the blocking creation in a blocking task
    pub async fn create(buffer: Arc<AsyncStreamBuffer>) -> Result<Self, ffmpeg::Error> {
        let buffer_clone = buffer.clone();
        tokio::task::spawn_blocking(move || Self::create_blocking(buffer_clone))
            .await
            .map_err(|_| ffmpeg::Error::External)?
    }

    pub fn input(&self) -> &ffmpeg::format::context::Input {
        &self.context
    }

    pub fn input_mut(&mut self) -> &mut ffmpeg::format::context::Input {
        &mut self.context
    }

    pub fn into_input(self) -> ffmpeg::format::context::Input {
        self.context
    }

    /// Create AsyncCustomInput from an existing FFmpeg input context
    /// This is useful for refactoring to avoid code duplication
    pub fn from_input_context(
        input_ctx: ffmpeg::format::context::Input,
    ) -> Result<Self, ffmpeg::Error> {
        // Create a dummy buffer since we're wrapping an existing context
        let buffer = AsyncStreamBuffer::new();

        Ok(AsyncCustomInput {
            _buffer: buffer,
            context: input_ctx,
        })
    }
}
