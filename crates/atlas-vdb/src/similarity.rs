/// Computes the cosine similarity between two vectors.
///
/// Returns a value in the range `[-1.0, 1.0]`. Returns `0.0` if either vector
/// has zero magnitude. Uses manual 4-way loop unrolling for performance.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
///
/// # Examples
///
/// ```
/// use atlas_vdb::cosine_similarity;
///
/// let a = vec![1.0, 0.0, 0.0];
/// let b = vec![1.0, 0.0, 0.0];
/// assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
///
/// let c = vec![0.0, 1.0, 0.0];
/// assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);
/// ```
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same dimension");
    let dot = dot_product(a, b);
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Computes the dot product (inner product) of two vectors.
///
/// Uses manual 4-way loop unrolling for performance.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
///
/// # Examples
///
/// ```
/// use atlas_vdb::dot_product;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// let result = dot_product(&a, &b);
/// assert!((result - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
/// ```
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same dimension");
    let mut sum = 0.0f32;
    let mut i = 0;
    let len = a.len();
    while i + 4 <= len {
        sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;
    let len = v.len();
    while i + 4 <= len {
        sum += v[i] * v[i] + v[i + 1] * v[i + 1] + v[i + 2] * v[i + 2] + v[i + 3] * v[i + 3];
        i += 4;
    }
    while i < len {
        sum += v[i] * v[i];
        i += 1;
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        let score = cosine_similarity(&v, &v);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_range() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let score = cosine_similarity(&a, &b);
        assert!((-1.0..=1.0).contains(&score));
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_both_zero() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_general() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dot = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0;
        let norm_a = (1.0 + 4.0 + 9.0_f32).sqrt();
        let norm_b = (16.0 + 25.0 + 36.0_f32).sqrt();
        let expected = dot / (norm_a * norm_b);
        let score = cosine_similarity(&a, &b);
        assert!((score - expected).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert!((result - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let result = dot_product(&a, &b);
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_zero() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = dot_product(&a, &b);
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_negative_components() {
        let a = vec![1.0, -1.0, 0.0];
        let b = vec![1.0, 1.0, 0.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_all_negative_range() {
        let vectors = vec![
            (vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]),
            (vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]),
            (vec![1.0, 0.0, 0.0], vec![-1.0, 0.0, 0.0]),
            (vec![1.0, 1.0, 0.0], vec![-1.0, -1.0, 0.0]),
            (vec![3.0, 4.0], vec![4.0, 3.0]),
        ];
        for (a, b) in vectors {
            let score = cosine_similarity(&a, &b);
            assert!(
                (-1.0 - 1e-6..=1.0 + 1e-6).contains(&score),
                "Score {score} out of range for {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn test_cosine_similarity_symmetry() {
        let pairs = vec![
            (vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]),
            (vec![1.0, 0.0], vec![0.0, 1.0]),
            (vec![3.0, 4.0], vec![1.0, 0.0]),
            (vec![1.0, -1.0, 0.0], vec![1.0, 1.0, 0.0]),
        ];
        for (a, b) in pairs {
            let ab = cosine_similarity(&a, &b);
            let ba = cosine_similarity(&b, &a);
            assert!((ab - ba).abs() < 1e-6, "Not symmetric: cos(a,b)={ab}, cos(b,a)={ba}");
        }
    }

    #[test]
    fn test_cosine_similarity_commutative_with_scaling() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let b_scaled: Vec<f32> = b.iter().map(|x| x * 10.0).collect();
        let s1 = cosine_similarity(&a, &b);
        let s2 = cosine_similarity(&a, &b_scaled);
        assert!((s1 - s2).abs() < 1e-6, "Scaling should not affect cosine similarity");
    }

    #[test]
    fn test_cosine_similarity_value_range_exhaustive() {
        let test_cases = vec![
            (vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0], 1.0),
            (vec![1.0, 0.0, 0.0], vec![-1.0, 0.0, 0.0], -1.0),
            (vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], 0.0),
        ];
        for (a, b, expected) in test_cases {
            let score = cosine_similarity(&a, &b);
            assert!(
                (score - expected).abs() < 1e-6,
                "Expected {expected}, got {score} for {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn test_dot_product_commutative() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let ab = dot_product(&a, &b);
        let ba = dot_product(&b, &a);
        assert!((ab - ba).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_distributive() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let c = vec![5.0, 6.0];
        let ab = dot_product(&a, &b);
        let ac = dot_product(&a, &c);
        let bc_sum: Vec<f32> = b.iter().zip(c.iter()).map(|(x, y)| x + y).collect();
        let a_bc = dot_product(&a, &bc_sum);
        assert!((ab + ac - a_bc).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_high_dimensional() {
        let dim = 256;
        let a: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();
        let score = cosine_similarity(&a, &b);
        assert!((-1.0..=1.0).contains(&score));
    }
}
