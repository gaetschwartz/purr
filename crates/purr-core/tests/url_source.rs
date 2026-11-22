use std::{
    convert::Infallible,
    future::IntoFuture,
    io,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use axum::{Router, body::Body, http::header, routing::get};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use pretty_assertions::assert_eq;
use purr_core::{
    AudioChunk, AudioProcessor, AudioStream, Error, SourceLength, config::InputConfig,
};
use tokio::{net::TcpListener, sync::oneshot};
use url::Url;

#[path = "common/flac.rs"]
mod flac;
#[path = "common/wav.rs"]
mod wav;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const SECOND: Duration = Duration::from_secs(1);

async fn serve(
    make_body: impl Fn() -> Body + Clone + Send + Sync + 'static,
) -> Result<Url, Box<dyn std::error::Error>> {
    let app = Router::new().route(
        "/audio",
        get(move || {
            let make_body = make_body.clone();
            async move { ([(header::CONTENT_TYPE, "audio/wav")], make_body()) }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    tokio::spawn(axum::serve(listener, app).into_future());
    Ok(format!("http://{address}/audio").parse()?)
}

fn sized(bytes: Vec<u8>) -> impl Fn() -> Body + Clone + Send + Sync + 'static {
    move || Body::from(bytes.clone())
}

fn streamed(pieces: Vec<Bytes>) -> impl Fn() -> Body + Clone + Send + Sync + 'static {
    move || {
        Body::from_stream(futures::stream::iter(
            pieces.clone().into_iter().map(Ok::<_, Infallible>),
        ))
    }
}

async fn drain(stream: &mut AudioStream) -> Result<Vec<AudioChunk>, Error> {
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item?);
    }
    Ok(chunks)
}

fn within(actual: Duration, expected: Duration, tolerance: Duration) -> bool {
    actual.abs_diff(expected) < tolerance
}

#[tokio::test]
async fn file_reports_exact_length_before_decoding() -> TestResult {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("three.wav");
    std::fs::write(&path, wav::wav(3))?;

    let stream = AudioProcessor::stream_file(&path).await?;

    assert_eq!(stream.length(), SourceLength::Exact(Duration::from_secs(3)));
    Ok(())
}

#[tokio::test]
async fn probe_file_length_matches() -> TestResult {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("three.wav");
    std::fs::write(&path, wav::wav(3))?;

    let length = AudioProcessor::probe_file_length(&path)?;

    assert_eq!(length, SourceLength::Exact(Duration::from_secs(3)));
    Ok(())
}

#[tokio::test]
async fn url_with_header_duration_reports_exact_length() -> TestResult {
    let url = serve(sized(flac::stream_info_only(2))).await?;

    let stream = AudioProcessor::stream_url_with_config(url, &InputConfig::default()).await?;

    assert_eq!(stream.length(), SourceLength::Exact(Duration::from_secs(2)));
    Ok(())
}

#[tokio::test]
async fn url_without_header_duration_but_with_content_length_reports_estimated() -> TestResult {
    let url = serve(sized(wav::open_ended_wav(2))).await?;

    let stream = AudioProcessor::stream_url_with_config(url, &InputConfig::default()).await?;

    assert!(
        matches!(
            stream.length(),
            SourceLength::Estimated(length)
                if within(length, Duration::from_secs(2), Duration::from_millis(50))
        ),
        "{:?}",
        stream.length()
    );
    Ok(())
}

#[tokio::test]
async fn chunked_body_of_unknown_size_reports_unknown() -> TestResult {
    let url = serve(streamed(vec![Bytes::from(wav::open_ended_wav(2))])).await?;

    let stream = AudioProcessor::stream_url_with_config(url, &InputConfig::default()).await?;

    assert_eq!(stream.length(), SourceLength::Unknown);
    Ok(())
}

#[tokio::test]
async fn body_that_fails_midway_surfaces_a_network_error() -> TestResult {
    let url = serve(|| {
        let audio = Bytes::from(wav::open_ended_wav(12));
        Body::from_stream(futures::stream::once(async move { Ok(audio) }).chain(
            futures::stream::once(async {
                tokio::time::sleep(Duration::from_millis(200)).await;
                Err(io::Error::other("the server broke the connection"))
            }),
        ))
    })
    .await?;

    let outcome = tokio::time::timeout(Duration::from_secs(10), async {
        let mut stream =
            AudioProcessor::stream_url_with_config(url, &InputConfig::default()).await?;
        let mut chunks = 0;
        loop {
            match stream.next().await {
                Some(Ok(_)) => chunks += 1,
                Some(Err(error)) => return Ok::<_, Error>((chunks, error)),
                None => panic!("the stream ended without reporting the failure"),
            }
        }
    })
    .await?;

    let (chunks, error) = outcome?;
    assert_eq!(chunks, 1);
    assert!(matches!(error, Error::NetworkError { .. }), "{error:?}");
    Ok(())
}

#[tokio::test]
async fn stalled_body_ends_with_a_read_timeout() -> TestResult {
    let url = serve(|| {
        let first = Bytes::from(wav::open_ended_wav(1));
        Body::from_stream(
            futures::stream::once(async move { Ok::<_, Infallible>(first) })
                .chain(futures::stream::pending()),
        )
    })
    .await?;
    let config = InputConfig::default().with_http_timeout(1);

    let outcome = tokio::time::timeout(Duration::from_secs(5), async {
        let mut stream = AudioProcessor::stream_url_with_config(url, &config).await?;
        drain(&mut stream).await
    })
    .await?;

    assert!(
        matches!(outcome, Err(Error::NetworkError { .. })),
        "{outcome:?}"
    );
    Ok(())
}

#[tokio::test]
async fn download_longer_than_the_http_timeout_completes() -> TestResult {
    let url = serve(|| {
        let header = Bytes::from(wav::sized_header(1));
        let audio = wav::silence_bytes(1);
        let pieces: Vec<Bytes> = audio
            .chunks(audio.len() / 15 + 1)
            .map(Bytes::copy_from_slice)
            .collect();
        let slow = futures::stream::iter(pieces).then(|piece| async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            piece
        });
        Body::from_stream(
            futures::stream::once(async move { header })
                .chain(slow)
                .map(Ok::<_, Infallible>),
        )
    })
    .await?;
    let config = InputConfig::default().with_http_timeout(1);

    let mut stream = AudioProcessor::stream_url_with_config(url, &config).await?;
    let chunks = tokio::time::timeout(Duration::from_secs(20), drain(&mut stream)).await??;

    let samples: usize = chunks.iter().map(|chunk| chunk.samples.len()).sum();
    assert_eq!(samples, wav::SAMPLE_RATE as usize);
    Ok(())
}

struct DropSignal(Option<oneshot::Sender<()>>);

impl Drop for DropSignal {
    fn drop(&mut self) {
        if let Some(signal) = self.0.take()
            && signal.send(()).is_err()
        {
            tracing::debug!("The test no longer waits for the body to drop");
        }
    }
}

fn endless(
    produced: Arc<AtomicUsize>,
    dropped: oneshot::Sender<()>,
) -> impl Stream<Item = Result<Bytes, Infallible>> {
    futures::stream::unfold(
        (
            Some(Bytes::from(wav::open_ended_header())),
            produced,
            DropSignal(Some(dropped)),
        ),
        |(mut header, produced, signal)| async move {
            let piece = header
                .take()
                .unwrap_or_else(|| Bytes::from(vec![0u8; 8192]));
            produced.fetch_add(piece.len(), Ordering::SeqCst);
            tokio::task::yield_now().await;
            Some((Ok(piece), (header, produced, signal)))
        },
    )
}

#[tokio::test]
async fn endless_body_is_bounded_and_dropping_stops_it() -> TestResult {
    let produced = Arc::new(AtomicUsize::new(0));
    let (dropped_tx, dropped_rx) = oneshot::channel();
    let body = Arc::new(parking_lot::Mutex::new(Some((
        Arc::clone(&produced),
        dropped_tx,
    ))));
    let url = serve({
        let body = Arc::clone(&body);
        move || {
            let taken = body.lock().take();
            match taken {
                Some((produced, dropped)) => Body::from_stream(endless(produced, dropped)),
                None => Body::empty(),
            }
        }
    })
    .await?;

    let mut stream = AudioProcessor::stream_url_with_config(url, &InputConfig::default()).await?;
    assert_eq!(stream.length(), SourceLength::Unknown);
    for _ in 0..3 {
        let item = tokio::time::timeout(Duration::from_secs(30), stream.next()).await?;
        assert!(matches!(item, Some(Ok(_))));
    }
    tokio::time::sleep(SECOND).await;
    drop(stream);

    tokio::time::timeout(Duration::from_secs(5), dropped_rx).await??;
    let produced = produced.load(Ordering::SeqCst);
    assert!(
        produced < 32 * 1024 * 1024,
        "{produced} bytes were produced"
    );
    Ok(())
}
