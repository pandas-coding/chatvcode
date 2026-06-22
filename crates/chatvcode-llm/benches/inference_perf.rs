//! Performance benchmarks for chatvcode-llm.
//!
//! Measures key inference metrics:
//! - First token latency (time to first token)
//! - Tokens per second (throughput)
//! - Prompt processing time
//!
//! Run with: `cargo bench --package chatvcode-llm`

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
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

criterion_group!(
    benches,
    bench_sync_inference,
    bench_stream_inference,
    bench_first_token_latency,
    bench_prompt_processing,
);
criterion_main!(benches);
