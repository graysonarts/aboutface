//! Where a Face's three image files live.
//!
//! One directory per Face, sharded a thousand Faces to a shard. The Corpus
//! grows without bound (`CONTEXT.md`), and a single directory holding three
//! files for every Visitor of every showing eventually becomes a directory no
//! tool enjoys listing. The shard is derived from the identifier, so a path is
//! a pure function of a `FaceId` and nothing needs to record it.

use std::path::{Path, PathBuf};

use afcore::FaceId;

/// The directory holding every Face's images.
const FACES_DIR: &str = "faces";

/// Faces per shard directory.
const SHARD_SIZE: u64 = 1_000;

/// Which image a path refers to.
///
/// The aligned crop and the display crop are not interchangeable and never
/// become so: one is what the model saw, the other is what the wall shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImageKind {
    /// The full-quality frame the Capture produced. Re-embedding reads this.
    Original,
    /// The 112×112 crop the embedder saw.
    Aligned,
    /// The portrait the wall shows.
    Display,
}

impl ImageKind {
    /// The file name this image is stored under.
    ///
    /// PNG throughout, including the original: a lossy re-encode would change
    /// what a re-embed after a model change sees (ADR-0006).
    pub(crate) const fn file_name(self) -> &'static str {
        match self {
            Self::Original => "original.png",
            Self::Aligned => "aligned.png",
            Self::Display => "display.png",
        }
    }
}

/// The directory holding `face`'s images, inside the Corpus at `root`.
pub(crate) fn face_dir(root: &Path, face: FaceId) -> PathBuf {
    root.join(FACES_DIR)
        .join(format!("{:04}", face.0 / SHARD_SIZE))
        .join(format!("{:06}", face.0))
}

/// The path of one image inside a Face's directory.
///
/// The one place a file name is joined to a directory: writing and reading
/// agree because they call the same function, not because two sites were kept
/// in step.
pub(crate) fn in_dir(dir: &Path, kind: ImageKind) -> PathBuf {
    dir.join(kind.file_name())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_shard_by_the_thousand_when_a_face_dir_is_named() {
        let root = Path::new("/corpus");

        assert_eq!(
            face_dir(root, FaceId(7)),
            Path::new("/corpus/faces/0000/000007")
        );
        assert_eq!(
            face_dir(root, FaceId(1_234)),
            Path::new("/corpus/faces/0001/001234")
        );
        assert_eq!(
            face_dir(root, FaceId(12_345_678)),
            Path::new("/corpus/faces/12345/12345678")
        );
    }

    #[test]
    fn should_give_each_image_its_own_name_when_a_face_is_stored() {
        let dir = face_dir(Path::new("/corpus"), FaceId(1));
        let paths = [ImageKind::Original, ImageKind::Aligned, ImageKind::Display]
            .map(|kind| in_dir(&dir, kind));

        assert_eq!(
            paths,
            [
                PathBuf::from("/corpus/faces/0000/000001/original.png"),
                PathBuf::from("/corpus/faces/0000/000001/aligned.png"),
                PathBuf::from("/corpus/faces/0000/000001/display.png"),
            ]
        );
    }
}
