use atlas_vdb::{CompactVectorStore, EmbeddingVector, InMemoryVectorStore, VectorStore};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use tempfile::tempdir;

fn create_test_vectors(count: usize, dimension: usize) -> Vec<EmbeddingVector> {
    (0..count)
        .map(|i| {
            let vector: Vec<f32> = (0..dimension)
                .map(|j| (i * dimension + j) as f32 * 0.01)
                .collect();
            EmbeddingVector::new(format!("chunk_{i}"), vector)
        })
        .collect()
}

fn bench_save_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("save_load");

    for size in &[100, 1000, 5000] {
        let vectors = create_test_vectors(*size, 128);

        group.bench_with_input(BenchmarkId::new("InMemoryVectorStore", size), size, |b, &_size| {
            b.iter(|| {
                let dir = tempdir().unwrap();
                let path = dir.path().join("test.vdb");

                let mut store = InMemoryVectorStore::new();
                store.add(vectors.clone()).unwrap();
                store.save(&path).unwrap();
                let _loaded = InMemoryVectorStore::load(&path).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("CompactVectorStore", size), size, |b, &_size| {
            b.iter(|| {
                let dir = tempdir().unwrap();
                let path = dir.path().join("test.vdb");

                let mut store = CompactVectorStore::new();
                store.add(vectors.clone()).unwrap();
                store.save(&path).unwrap();
                let _loaded = CompactVectorStore::load(&path).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search");

    for size in &[1000, 5000, 10000] {
        let vectors = create_test_vectors(*size, 128);
        let query: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();

        group.bench_with_input(BenchmarkId::new("InMemoryVectorStore", size), size, |b, &_size| {
            let mut store = InMemoryVectorStore::new();
            store.add(vectors.clone()).unwrap();

            b.iter(|| {
                store.search(black_box(&query), 10, None).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("CompactVectorStore", size), size, |b, &_size| {
            let mut store = CompactVectorStore::new();
            store.add(vectors.clone()).unwrap();

            b.iter(|| {
                store.search(black_box(&query), 10, None).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    let vectors = create_test_vectors(10000, 128);

    group.bench_function("InMemoryVectorStore", |b| {
        b.iter(|| {
            let mut store = InMemoryVectorStore::new();
            store.add(vectors.clone()).unwrap();
            black_box(store);
        });
    });

    group.bench_function("CompactVectorStore", |b| {
        b.iter(|| {
            let mut store = CompactVectorStore::new();
            store.add(vectors.clone()).unwrap();
            black_box(store);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_save_load, bench_search, bench_memory_usage);
criterion_main!(benches);
