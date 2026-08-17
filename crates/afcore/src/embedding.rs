//! Embeddings and the model that produced them.

use std::fmt;

/// Identifies the similarity model that produced an [`Embedding`].
///
/// This is part of the stored schema, not a runtime detail. Changing models
/// invalidates every Embedding in the Corpus (ADR-0006), so every Embedding
/// carries the identity of its producer and comparisons across models are
/// rejected rather than silently producing nonsense.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelId(String);

impl ModelId {
    /// Creates a model identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Ways an [`Embedding`] can be invalid or incomparable.
///
/// Not `Eq`: [`EmbeddingError::NotUnitLength`] reports the magnitude it
/// measured, and that is a float.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum EmbeddingError {
    /// An embedding must have at least one dimension.
    #[error("embedding is empty")]
    Empty,

    /// Values must all be finite; a NaN or infinity means inference went wrong.
    #[error("embedding contains a non-finite value at index {index}")]
    NotFinite {
        /// Index of the first offending value.
        index: usize,
    },

    /// A zero-magnitude vector cannot be normalized and carries no direction.
    #[error("embedding has zero magnitude and cannot be normalized")]
    ZeroMagnitude,

    /// Values offered as already-normalized are not unit length.
    #[error("embedding magnitude is {magnitude}, expected unit length")]
    NotUnitLength {
        /// The magnitude measured.
        magnitude: f32,
    },

    /// Embeddings from different models describe different spaces.
    #[error("cannot compare embeddings from different models: {left} vs {right}")]
    ModelMismatch {
        /// Model of the left-hand embedding.
        left: ModelId,
        /// Model of the right-hand embedding.
        right: ModelId,
    },

    /// Same model, different widths — the Corpus has stale entries in it.
    #[error("cannot compare embeddings of different widths: {left} vs {right}")]
    DimensionMismatch {
        /// Width of the left-hand embedding.
        left: usize,
        /// Width of the right-hand embedding.
        right: usize,
    },
}

/// An L2-normalized vector describing one Face's appearance.
///
/// Width is **model-dependent** and nothing may assume a fixed size — DINOv2
/// ships in several ViT sizes with different widths (ADR-0007). Distance between
/// Embeddings is cosine distance.
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    model: ModelId,
    values: Vec<f32>,
}

impl Embedding {
    /// Creates an Embedding, normalizing `values` to unit length.
    ///
    /// # Errors
    ///
    /// Returns an error if `values` is empty, contains a non-finite value, or
    /// has zero magnitude.
    pub fn new(model: ModelId, values: Vec<f32>) -> Result<Self, EmbeddingError> {
        if values.is_empty() {
            return Err(EmbeddingError::Empty);
        }
        if let Some(index) = values.iter().position(|v| !v.is_finite()) {
            return Err(EmbeddingError::NotFinite { index });
        }

        let magnitude = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if magnitude == 0.0 || !magnitude.is_finite() {
            return Err(EmbeddingError::ZeroMagnitude);
        }

        let values = values.into_iter().map(|v| v / magnitude).collect();
        Ok(Self { model, values })
    }

    /// Rebuilds an Embedding from values that are already unit length.
    ///
    /// [`Embedding::new`] divides by the magnitude it measures, and dividing an
    /// already-normalized vector by a magnitude that is only *approximately*
    /// 1.0 in `f32` moves every value. That is fine for inference output and
    /// wrong for the Corpus: a Face read back must be the Face that was
    /// written, bit for bit, or the archive drifts a little every time it is
    /// re-read.
    ///
    /// # Errors
    ///
    /// Returns an error if `values` is empty, contains a non-finite value, or
    /// has a magnitude further than `tolerance` from 1.0 — which means the
    /// stored values are not an Embedding, not that they need rescaling.
    pub fn from_unit(
        model: ModelId,
        values: Vec<f32>,
        tolerance: f32,
    ) -> Result<Self, EmbeddingError> {
        if values.is_empty() {
            return Err(EmbeddingError::Empty);
        }
        if let Some(index) = values.iter().position(|v| !v.is_finite()) {
            return Err(EmbeddingError::NotFinite { index });
        }

        let magnitude = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if !magnitude.is_finite() || (magnitude - 1.0).abs() > tolerance {
            return Err(EmbeddingError::NotUnitLength { magnitude });
        }

        Ok(Self { model, values })
    }

    /// The model that produced this Embedding.
    pub fn model(&self) -> &ModelId {
        &self.model
    }

    /// Number of dimensions.
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// The normalized values.
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Cosine distance to `other`, in `0.0..=2.0`. Identical direction is `0.0`.
    ///
    /// # Errors
    ///
    /// Returns an error if the two Embeddings came from different models or
    /// have different widths. Comparing across models is always a bug, never a
    /// degraded result.
    pub fn cosine_distance(&self, other: &Self) -> Result<f32, EmbeddingError> {
        if self.model != other.model {
            return Err(EmbeddingError::ModelMismatch {
                left: self.model.clone(),
                right: other.model.clone(),
            });
        }
        if self.values.len() != other.values.len() {
            return Err(EmbeddingError::DimensionMismatch {
                left: self.values.len(),
                right: other.values.len(),
            });
        }

        let dot: f32 = self
            .values
            .iter()
            .zip(&other.values)
            .map(|(a, b)| a * b)
            .sum();

        Ok((1.0 - dot).clamp(0.0, 2.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> ModelId {
        ModelId::new("dinov2-vitb14")
    }

    fn embed(values: Vec<f32>) -> Embedding {
        Embedding::new(model(), values).expect("valid embedding")
    }

    #[test]
    fn normalizes_to_unit_length() {
        let e = embed(vec![3.0, 4.0]);
        let magnitude: f32 = e.as_slice().iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 1e-6, "magnitude was {magnitude}");
    }

    #[test]
    fn scale_does_not_change_direction() {
        let small = embed(vec![1.0, 2.0, 3.0]);
        let large = embed(vec![10.0, 20.0, 30.0]);
        let distance = small.cosine_distance(&large).expect("comparable");
        assert!(distance < 1e-6, "distance was {distance}");
    }

    #[test]
    fn distance_spans_identical_orthogonal_and_opposite() {
        let a = embed(vec![1.0, 0.0]);
        let b = embed(vec![0.0, 1.0]);
        let opposite = embed(vec![-1.0, 0.0]);

        assert!(a.cosine_distance(&a).expect("comparable") < 1e-6);
        assert!((a.cosine_distance(&b).expect("comparable") - 1.0).abs() < 1e-6);
        assert!((a.cosine_distance(&opposite).expect("comparable") - 2.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_empty() {
        assert_eq!(Embedding::new(model(), vec![]), Err(EmbeddingError::Empty));
    }

    #[test]
    fn rejects_non_finite() {
        assert_eq!(
            Embedding::new(model(), vec![1.0, f32::NAN]),
            Err(EmbeddingError::NotFinite { index: 1 })
        );
    }

    #[test]
    fn rejects_zero_magnitude() {
        assert_eq!(
            Embedding::new(model(), vec![0.0, 0.0]),
            Err(EmbeddingError::ZeroMagnitude)
        );
    }

    #[test]
    fn keeps_the_values_untouched_when_they_are_already_unit_length() {
        let normalized = embed(vec![0.3, -0.7, 0.2, 1.1]);

        let rebuilt = Embedding::from_unit(model(), normalized.as_slice().to_vec(), 1e-5)
            .expect("already unit length");

        assert_eq!(rebuilt, normalized);
    }

    #[test]
    fn refuses_values_offered_as_normalized_when_they_are_not() {
        assert!(matches!(
            Embedding::from_unit(model(), vec![3.0, 4.0], 1e-5),
            Err(EmbeddingError::NotUnitLength { .. })
        ));
    }

    #[test]
    fn refuses_to_compare_across_models() {
        let dinov2 = embed(vec![1.0, 0.0]);
        let other = Embedding::new(ModelId::new("some-other-model"), vec![1.0, 0.0])
            .expect("valid embedding");

        assert!(matches!(
            dinov2.cosine_distance(&other),
            Err(EmbeddingError::ModelMismatch { .. })
        ));
    }

    #[test]
    fn refuses_to_compare_across_widths() {
        let narrow = embed(vec![1.0, 0.0]);
        let wide = embed(vec![1.0, 0.0, 0.0]);

        assert_eq!(
            narrow.cosine_distance(&wide),
            Err(EmbeddingError::DimensionMismatch { left: 2, right: 3 })
        );
    }
}
