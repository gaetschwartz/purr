//! Benchmark utilities and performance testing helpers

#[cfg(feature = "benchmarks")]
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::{Duration, Instant};
use crate::fixtures::AudioFixtures;
use crate::generators::AudioData;

/// Performance benchmark suite for audio processing operations
pub struct AudioBenchmarks;

impl AudioBenchmarks {
    /// Benchmark audio processing throughput
    #[cfg(feature = "benchmarks")]
    pub fn bench_audio_processing(c: &mut Criterion) {
        let mut group = c.benchmark_group("audio_processing");

        let sample_rates = [16000, 44100, 48000];
        let durations = [1.0, 5.0, 10.0]; // seconds

        for &sample_rate in &sample_rates {
            for &duration in &durations {
                let fixtures = AudioFixtures::new();
                let audio_data = fixtures.generate_sine_wave(duration, 440.0, sample_rate);
                let data_size = audio_data.len() * std::mem::size_of::<f32>();

                group.throughput(Throughput::Bytes(data_size as u64));
                group.bench_with_input(
                    BenchmarkId::new("sine_wave_generation", format!("{}Hz_{}s", sample_rate, duration)),
                    &(duration, sample_rate),
                    |b, &(duration, sample_rate)| {
                        b.iter(|| {
                            let fixtures = AudioFixtures::new();
                            black_box(fixtures.generate_sine_wave(duration, 440.0, sample_rate))
                        })
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("audio_normalization", format!("{}Hz_{}s", sample_rate, duration)),
                    &audio_data,
                    |b, audio_data| {
                        b.iter(|| {
                            let max_amplitude = audio_data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
                            if max_amplitude > 0.0 {
                                black_box(audio_data.iter().map(|&x| x / max_amplitude).collect::<Vec<f32>>())
                            } else {
                                black_box(audio_data.clone())
                            }
                        })
                    },
                );
            }
        }

        group.finish();
    }

    /// Benchmark transcription operations
    #[cfg(feature = "benchmarks")]
    pub fn bench_transcription(c: &mut Criterion) {
        let mut group = c.benchmark_group("transcription");

        let audio_lengths = [1000, 10000, 100000, 1000000]; // samples

        for &length in &audio_lengths {
            let audio_data: Vec<f32> = (0..length).map(|i| (i as f32 / 16000.0).sin()).collect();
            let data_size = audio_data.len() * std::mem::size_of::<f32>();

            group.throughput(Throughput::Bytes(data_size as u64));
            group.bench_with_input(
                BenchmarkId::new("mock_transcription", length),
                &audio_data,
                |b, audio_data| {
                    b.iter(|| {
                        // Mock transcription processing
                        let chunk_size = 1024;
                        let chunks: Vec<_> = audio_data.chunks(chunk_size).collect();
                        black_box(chunks.len())
                    })
                },
            );
        }

        group.finish();
    }

    /// Benchmark WebGPU operations
    #[cfg(all(feature = "benchmarks", feature = "web"))]
    pub fn bench_webgpu_operations(c: &mut Criterion) {
        let mut group = c.benchmark_group("webgpu");

        let buffer_sizes = [1024, 4096, 16384, 65536]; // floats

        for &size in &buffer_sizes {
            let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let data_size = data.len() * std::mem::size_of::<f32>();

            group.throughput(Throughput::Bytes(data_size as u64));
            group.bench_with_input(
                BenchmarkId::new("mock_compute_operation", size),
                &data,
                |b, data| {
                    b.iter(|| {
                        // Mock GPU compute operation (CPU simulation)
                        black_box(data.iter().map(|&x| x * 2.0).collect::<Vec<f32>>())
                    })
                },
            );
        }

        group.finish();
    }
}

/// Memory usage benchmark utilities
pub struct MemoryBenchmarks;

impl MemoryBenchmarks {
    /// Benchmark memory allocation patterns
    #[cfg(feature = "benchmarks")]
    pub fn bench_memory_allocation(c: &mut Criterion) {
        let mut group = c.benchmark_group("memory");

        let allocation_sizes = [1024, 10240, 102400, 1024000]; // bytes

        for &size in &allocation_sizes {
            group.throughput(Throughput::Bytes(size as u64));
            group.bench_with_input(
                BenchmarkId::new("vec_allocation", size),
                &size,
                |b, &size| {
                    b.iter(|| {
                        let vec: Vec<u8> = black_box(vec![0; size]);
                        black_box(vec.len())
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("vec_with_capacity", size),
                &size,
                |b, &size| {
                    b.iter(|| {
                        let mut vec: Vec<u8> = Vec::with_capacity(size);
                        vec.resize(size, 0);
                        black_box(vec.len())
                    })
                },
            );
        }

        group.finish();
    }
}

/// Async operation benchmarks
pub struct AsyncBenchmarks;

impl AsyncBenchmarks {
    /// Benchmark async operation overhead
    #[cfg(feature = "benchmarks")]
    pub fn bench_async_operations(c: &mut Criterion) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let mut group = c.benchmark_group("async");

        group.bench_function("tokio_spawn_overhead", |b| {
            b.to_async(&runtime).iter(|| async {
                let handle = tokio::spawn(async {
                    black_box(42)
                });
                black_box(handle.await.unwrap())
            })
        });

        group.bench_function("async_channel_throughput", |b| {
            b.to_async(&runtime).iter(|| async {
                let (tx, mut rx) = tokio::sync::mpsc::channel(100);

                let sender = tokio::spawn(async move {
                    for i in 0..100 {
                        tx.send(i).await.unwrap();
                    }
                });

                let receiver = tokio::spawn(async move {
                    let mut sum = 0;
                    while let Some(value) = rx.recv().await {
                        sum += value;
                    }
                    sum
                });

                sender.await.unwrap();
                black_box(receiver.await.unwrap())
            })
        });

        group.finish();
    }
}

/// Simple performance profiler for non-criterion benchmarks
pub struct SimpleProfiler {
    measurements: std::collections::HashMap<String, Vec<Duration>>,
}

impl SimpleProfiler {
    pub fn new() -> Self {
        Self {
            measurements: std::collections::HashMap::new(),
        }
    }

    /// Time a synchronous operation
    pub fn time_sync<F, R>(&mut self, name: &str, operation: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed();

        self.measurements
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(duration);

        result
    }

    /// Time an asynchronous operation
    pub async fn time_async<F, Fut, R>(&mut self, name: &str, operation: F) -> R
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = R>,
    {
        let start = Instant::now();
        let result = operation().await;
        let duration = start.elapsed();

        self.measurements
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(duration);

        result
    }

    /// Get average time for an operation
    pub fn average_time(&self, name: &str) -> Option<Duration> {
        self.measurements.get(name).and_then(|measurements| {
            if measurements.is_empty() {
                None
            } else {
                let total: Duration = measurements.iter().sum();
                Some(total / measurements.len() as u32)
            }
        })
    }

    /// Get all measurements for an operation
    pub fn get_measurements(&self, name: &str) -> Option<&Vec<Duration>> {
        self.measurements.get(name)
    }

    /// Get summary statistics
    pub fn summary(&self) -> std::collections::HashMap<String, ProfileStats> {
        self.measurements
            .iter()
            .map(|(name, measurements)| {
                let stats = ProfileStats::from_measurements(measurements);
                (name.clone(), stats)
            })
            .collect()
    }

    /// Clear all measurements
    pub fn clear(&mut self) {
        self.measurements.clear();
    }
}

impl Default for SimpleProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for profiling results
#[derive(Debug, Clone)]
pub struct ProfileStats {
    pub mean: Duration,
    pub min: Duration,
    pub max: Duration,
    pub std_dev: Duration,
    pub count: usize,
}

impl ProfileStats {
    fn from_measurements(measurements: &[Duration]) -> Self {
        if measurements.is_empty() {
            return Self {
                mean: Duration::ZERO,
                min: Duration::ZERO,
                max: Duration::ZERO,
                std_dev: Duration::ZERO,
                count: 0,
            };
        }

        let count = measurements.len();
        let total: Duration = measurements.iter().sum();
        let mean = total / count as u32;

        let min = *measurements.iter().min().unwrap();
        let max = *measurements.iter().max().unwrap();

        // Calculate standard deviation
        let variance_sum: f64 = measurements
            .iter()
            .map(|&d| {
                let diff = d.as_secs_f64() - mean.as_secs_f64();
                diff * diff
            })
            .sum();

        let variance = variance_sum / count as f64;
        let std_dev = Duration::from_secs_f64(variance.sqrt());

        Self {
            mean,
            min,
            max,
            std_dev,
            count,
        }
    }
}

#[cfg(feature = "benchmarks")]
criterion_group!(
    benches,
    AudioBenchmarks::bench_audio_processing,
    AudioBenchmarks::bench_transcription,
    MemoryBenchmarks::bench_memory_allocation,
    AsyncBenchmarks::bench_async_operations
);

#[cfg(all(feature = "benchmarks", feature = "web"))]
criterion_group!(
    webgpu_benches,
    AudioBenchmarks::bench_webgpu_operations
);

#[cfg(feature = "benchmarks")]
criterion_main!(benches);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_profiler() {
        let mut profiler = SimpleProfiler::new();

        // Time a simple operation
        let result = profiler.time_sync("test_operation", || {
            std::thread::sleep(Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);

        let avg_time = profiler.average_time("test_operation");
        assert!(avg_time.is_some());
        assert!(avg_time.unwrap() >= Duration::from_millis(10));
    }

    #[tokio::test]
    async fn test_async_profiler() {
        let mut profiler = SimpleProfiler::new();

        let result = profiler.time_async("async_operation", || async {
            tokio::time::sleep(Duration::from_millis(5)).await;
            "done"
        }).await;

        assert_eq!(result, "done");

        let measurements = profiler.get_measurements("async_operation");
        assert!(measurements.is_some());
        assert_eq!(measurements.unwrap().len(), 1);
    }

    #[test]
    fn test_profile_stats() {
        let measurements = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
        ];

        let stats = ProfileStats::from_measurements(&measurements);
        assert_eq!(stats.count, 3);
        assert_eq!(stats.mean, Duration::from_millis(20));
        assert_eq!(stats.min, Duration::from_millis(10));
        assert_eq!(stats.max, Duration::from_millis(30));
    }
}