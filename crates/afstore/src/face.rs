//! A Face on its way into the Corpus, and a Face read back out.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use afcore::{Embedding, FaceId};
use image::RgbImage;

use crate::paths::{ImageKind, in_dir};

/// One Capture's worth of material, ready to store.
///
/// Everything here is already derived: this crate detects nothing, aligns
/// nothing and runs no model. `afvision` produced the Embedding and the crops,
/// and the caller — the booth — decides when a Capture happened.
///
/// The original frame is retained at full quality because a model change
/// invalidates every Embedding and re-embedding reads the originals back
/// (ADR-0006).
#[derive(Debug, Clone)]
pub struct NewFace {
    embedding: Embedding,
    aligned: RgbImage,
    display: RgbImage,
    original: RgbImage,
    captured_at: SystemTime,
}

impl NewFace {
    /// Assembles a Face from its Embedding, its two crops, and its frame.
    pub fn new(
        embedding: Embedding,
        aligned: RgbImage,
        display: RgbImage,
        original: RgbImage,
        captured_at: SystemTime,
    ) -> Self {
        Self {
            embedding,
            aligned,
            display,
            original,
            captured_at,
        }
    }

    /// The Embedding this Face will be found by.
    pub fn embedding(&self) -> &Embedding {
        &self.embedding
    }

    /// The 112×112 crop the embedder saw.
    pub fn aligned(&self) -> &RgbImage {
        &self.aligned
    }

    /// The portrait the wall shows.
    pub fn display(&self) -> &RgbImage {
        &self.display
    }

    /// The full-quality frame the Capture produced.
    pub fn original(&self) -> &RgbImage {
        &self.original
    }

    /// When the Visitor pressed the Shutter.
    pub fn captured_at(&self) -> SystemTime {
        self.captured_at
    }
}

/// A Face read back out of the Corpus.
///
/// The images are paths rather than decoded pixels: the wall shows up to a
/// thousand Faces and decodes them on its own schedule, and the layout stage
/// wants the Embeddings without touching a single image file.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredFace {
    id: FaceId,
    embedding: Embedding,
    captured_at: SystemTime,
    dir: PathBuf,
}

impl StoredFace {
    pub(crate) fn new(
        id: FaceId,
        embedding: Embedding,
        captured_at: SystemTime,
        dir: PathBuf,
    ) -> Self {
        Self {
            id,
            embedding,
            captured_at,
            dir,
        }
    }

    /// Which Face this is.
    pub fn id(&self) -> FaceId {
        self.id
    }

    /// The Embedding stored for it, with the model that produced it.
    pub fn embedding(&self) -> &Embedding {
        &self.embedding
    }

    /// When the Visitor pressed the Shutter.
    pub fn captured_at(&self) -> SystemTime {
        self.captured_at
    }

    /// The full-quality frame the Capture produced.
    pub fn original_path(&self) -> PathBuf {
        in_dir(&self.dir, ImageKind::Original)
    }

    /// The 112×112 crop the embedder saw.
    pub fn aligned_path(&self) -> PathBuf {
        in_dir(&self.dir, ImageKind::Aligned)
    }

    /// The portrait the wall shows.
    pub fn display_path(&self) -> PathBuf {
        in_dir(&self.dir, ImageKind::Display)
    }

    /// The directory holding this Face's three images.
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

/// Milliseconds since the Unix epoch, negative before it.
///
/// The Corpus records when a Visitor pressed the Shutter, and an installation
/// machine whose clock is wrong is not a reason to refuse the Capture — so a
/// time before the epoch round-trips rather than erroring.
pub(crate) fn to_unix_millis(time: SystemTime) -> i64 {
    match time.duration_since(UNIX_EPOCH) {
        Ok(since) => since.as_millis() as i64,
        Err(before) => -(before.duration().as_millis() as i64),
    }
}

/// The inverse of [`to_unix_millis`].
pub(crate) fn from_unix_millis(millis: i64) -> SystemTime {
    if millis >= 0 {
        UNIX_EPOCH + Duration::from_millis(millis as u64)
    } else {
        UNIX_EPOCH - Duration::from_millis(millis.unsigned_abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_round_trip_a_capture_time_when_it_is_after_the_epoch() {
        let time = UNIX_EPOCH + Duration::from_millis(1_755_000_000_123);

        assert_eq!(from_unix_millis(to_unix_millis(time)), time);
    }

    #[test]
    fn should_round_trip_a_capture_time_when_the_machine_clock_is_before_the_epoch() {
        let time = UNIX_EPOCH - Duration::from_millis(86_400_000);

        assert_eq!(from_unix_millis(to_unix_millis(time)), time);
    }
}
