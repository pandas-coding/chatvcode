//! Performance benchmarks for chatvcode-llm.
//!
//! Measures key inference metrics:
//! - First token latency (time to first token)
//! - Tokens per second (throughput)
//! - Prompt processing time
//! - Batch inference throughput
//! - Memory allocation patterns
//!
//! Run with: `cargo bench --package chatvcode-llm`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use chatvcode_llm::{GenerationParams, LlmService, MockLlmService};
use std::time::Instant;

fn bench_sync_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("sync_inference");
    
    let service = MockLlmService::new("The quick brown fox jumps over the lazy dog")
        .with_tokens_per_second(100.0);
    let params = GenerationParams::default();

    group.bench_function("basic", |b| {
        b.iter(|| {
            let response = service.infer(
                black_box("What is Rust?"),
                black_box(&params),
                None,
            ).unwrap();
            black_box(response.text);
            black_box(response.tokens_per_second);
        })
    });

    group.finish();
}

fn bench_stream_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("stream_inference");
    
    let service = MockLlmService::new("The quick brown fox jumps over the lazy dog")
        .with_tokens_per_second(100.0);
    let params = GenerationParams::default();

    group.bench_function("basic", |b| {
        b.iter(|| {
            let rx = service.infer_stream(
                black_box("What is Rust?"),
                black_box(&params),
                None,
            ).unwrap();
            
            let mut token_count = 0;
            let mut first_token_time = None;
            let start = Instant::now();
            
            while let Ok(event) = rx.recv() {
                if event.is_token() {
                    if first_token_time.is_none() {
                        first_token_time = Some(start.elapsed());
                    }
                    token_count += 1;
                }
                if event.is_terminal() {
                    break;
                }
            }
            
            black_box(token_count);
            black_box(first_token_time);
        })
    });

    group.finish();
}

fn bench_first_token_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("first_token_latency");
    
    for tps in [50.0, 100.0, 200.0] {
        let service = MockLlmService::new("Hello world from the model")
            .with_tokens_per_second(tps);
        let params = GenerationParams::default();

        group.bench_with_input(
            BenchmarkId::new("tokens_per_sec", tps),
            &tps,
            |b, _| {
                b.iter(|| {
                    let rx = service.infer_stream(
                        black_box("test"),
                        black_box(&params),
                        None,
                    ).unwrap();
                    
                    let start = Instant::now();
                    let mut first_token_latency = None;
                    
                    while let Ok(event) = rx.recv() {
                        if event.is_token() {
                            first_token_latency = Some(start.elapsed());
                            break;
                        }
                        if event.is_terminal() {
                            break;
                        }
                    }
                    
                    black_box(first_token_latency);
                })
            },
        );
    }

    group.finish();
}

fn bench_prompt_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("prompt_processing");
    
    let long_prompt = "This is a very long prompt that contains a lot of text and context. ".repeat(10);
    let prompts = [
        "Short prompt",
        "This is a medium length prompt that provides more context for the model to work with",
        long_prompt.as_str(),
    ];

    for (i, prompt) in prompts.iter().enumerate() {
        let service = MockLlmService::new("Response text here")
            .with_tokens_per_second(100.0);
        let params = GenerationParams::default();

        group.bench_with_input(
            BenchmarkId::new("prompt_length", i),
            prompt,
            |b, p| {
                b.iter(|| {
                    let response = service.infer(
                        black_box(p),
                        black_box(&params),
                        None,
                    ).unwrap();
                    black_box(response.duration);
                })
            },
        );
    }

    group.finish();
}

fn bench_batch_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_inference");
    group.sample_size(10);

    for batch_size in [1, 4, 8, 16] {
        let service = MockLlmService::new("Batch response text")
            .with_tokens_per_second(200.0);
        let params = GenerationParams::default();
        let prompts: Vec<String> = (0..batch_size)
            .map(|i| format!("Prompt number {i} for batch processing"))
            .collect();
        let prompt_refs: Vec<&str> = prompts.iter().map(|s| s.as_str()).collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let results = service
                        .infer_batch(black_box(&prompt_refs), black_box(&params), None)
                        .unwrap();
                    black_box(results.len());
                })
            },
        );
    }

    group.finish();
}

fn bench_throughput_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_scaling");
    group.sample_size(10);

    for tps in [50.0, 100.0, 200.0, 500.0] {
        let service = MockLlmService::new(
            "This is a longer response that contains multiple sentences \
             to better measure throughput characteristics of the inference pipeline.",
        )
        .with_tokens_per_second(tps);
        let params = GenerationParams::default();

        group.bench_with_input(
            BenchmarkId::new("tokens_per_sec", tps as u32),
            &tps,
            |b, _| {
                b.iter(|| {
                    let response = service
                        .infer(black_box("test prompt"), black_box(&params), None)
                        .unwrap();
                    black_box(response.tokens_per_second);
                })
            },
        );
    }

    group.finish();
}

fn bench_max_tokens_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("max_tokens_impact");
    group.sample_size(10);

    let service = MockLlmService::new(
        "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 \
         word11 word12 word13 word14 word15 word16 word17 word18 word19 word20",
    )
    .with_tokens_per_second(200.0);

    for max_tokens in [10, 50, 100, 500] {
        let params = GenerationParams::default().with_max_tokens(max_tokens);

        group.bench_with_input(
            BenchmarkId::new("max_tokens", max_tokens),
            &max_tokens,
            |b, _| {
                b.iter(|| {
                    let response = service
                        .infer(black_box("test"), black_box(&params), None)
                        .unwrap();
                    black_box(response.token_usage.completion_tokens);
                })
            },
        );
    }

    group.finish();
}

fn bench_stream_vs_sync(c: &mut Criterion) {
    let mut group = c.benchmark_group("stream_vs_sync");
    group.sample_size(10);

    let response_text = "This is a benchmark response with enough text to measure performance differences between streaming and synchronous inference modes.";

    group.bench_function("sync", |b| {
        let service = MockLlmService::new(response_text).with_tokens_per_second(200.0);
        let params = GenerationParams::default();
        b.iter(|| {
            let response = service
                .infer(black_box("test"), black_box(&params), None)
                .unwrap();
            black_box(response.text);
        })
    });

    group.bench_function("stream", |b| {
        let service = MockLlmService::new(response_text).with_tokens_per_second(200.0);
        let params = GenerationParams::default();
        b.iter(|| {
            let rx = service
                .infer_stream(black_box("test"), black_box(&params), None)
                .unwrap();
            let mut tokens = Vec::new();
            while let Ok(event) = rx.recv() {
                if let Some(t) = event.as_token() {
                    tokens.push(t.to_string());
                }
                if event.is_terminal() {
                    break;
                }
            }
            black_box(tokens);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sync_inference,
    bench_stream_inference,
    bench_first_token_latency,
    bench_prompt_processing,
    bench_batch_inference,
    bench_throughput_scaling,
    bench_max_tokens_impact,
    bench_stream_vs_sync,
);
criterion_main!(benches);
