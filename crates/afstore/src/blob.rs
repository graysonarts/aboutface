//! How an Embedding's values sit in a database column.
//!
//! A blob of little-endian `f32`s, fixed-endian so a Corpus copied between
//! machines still reads. Embeddings are read whole — the layout stage loads
//! every one of them at once — so there is nothing to gain from a row per
//! dimension, and the width is stored beside the blob rather than inferred
//! from its length, so a truncated blob is detectable rather than silently a
//! narrower Embedding (ADR-0007: nothing may assume a width).

/// Bytes per stored value.
const VALUE_BYTES: usize = size_of::<f32>();

/// Packs `values` into a blob.
pub(crate) fn encode(values: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(values.len() * VALUE_BYTES);
    for value in values {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    blob
}

/// Unpacks a blob of exactly `dim` values.
///
/// Returns `None` when the blob's length disagrees with `dim` — that is a
/// corrupt row, not an Embedding.
pub(crate) fn decode(blob: &[u8], dim: usize) -> Option<Vec<f32>> {
    if blob.len() != dim * VALUE_BYTES {
        return None;
    }

    blob.chunks_exact(VALUE_BYTES)
        .map(|chunk| chunk.try_into().ok().map(f32::from_le_bytes))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_return_the_same_values_when_a_blob_round_trips() {
        let values = vec![0.5, -0.25, 1.0, 0.0];

        let decoded = decode(&encode(&values), values.len()).expect("the blob decodes");

        assert_eq!(decoded, values);
    }

    #[test]
    fn should_be_little_endian_whatever_the_machine_is() {
        assert_eq!(encode(&[1.0]), vec![0x00, 0x00, 0x80, 0x3f]);
    }

    #[test]
    fn should_refuse_when_the_blob_disagrees_with_the_recorded_width() {
        let blob = encode(&[1.0, 2.0]);

        assert_eq!(decode(&blob, 3), None);
        assert_eq!(decode(&blob[..7], 2), None);
    }
}
