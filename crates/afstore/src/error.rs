//! What can go wrong between a Capture and a row in the Corpus.

use std::path::PathBuf;

use afcore::FaceId;

/// Ways the Corpus can refuse a write or a read.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// SQLite refused the statement, the transaction, or the file.
    #[error("the corpus database failed: {0}")]
    Database(#[from] rusqlite::Error),

    /// A directory or image file could not be written or read.
    #[error("{path}: {source}")]
    File {
        /// The file or directory involved.
        path: PathBuf,
        /// What the filesystem said.
        source: std::io::Error,
    },

    /// An image could not be encoded to, or decoded from, PNG.
    #[error("{path}: {source}")]
    Image {
        /// The image file involved.
        path: PathBuf,
        /// What the codec said.
        source: image::ImageError,
    },

    /// The database holds a Face this build cannot read back.
    ///
    /// A stored Embedding whose blob length disagrees with its recorded width,
    /// or whose values no longer normalize, is corruption — not a Face to hand
    /// to the layout stage with a shrug.
    #[error("{face} is stored inconsistently: {detail}")]
    Corrupt {
        /// The Face the inconsistency was found under.
        face: FaceId,
        /// What disagreed.
        detail: String,
    },

    /// A Face's image directory was already occupied at ingest.
    ///
    /// Deleting it would be the convenient move and is refused: a Corpus whose
    /// database was restored from a backup while `faces/` survived would hand
    /// out an identifier that already has images under it, and those images are
    /// some Visitor's original frame — the one file a re-embed cannot do
    /// without (ADR-0006).
    #[error("{path} already holds images; the corpus and its database disagree")]
    FaceDirOccupied {
        /// The directory that was in the way.
        path: PathBuf,
    },

    /// The database was written by a newer build of the piece.
    ///
    /// Migrations only go forwards. An older binary pointed at a newer Corpus
    /// stops rather than writing rows the newer schema cannot read.
    #[error("corpus schema is version {found}, this build understands {understood}")]
    SchemaTooNew {
        /// The version stamped in the database.
        found: u32,
        /// The version this build migrates to.
        understood: u32,
    },
}

impl StoreError {
    /// Tags an I/O failure with the path that caused it.
    pub(crate) fn file(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::File {
            path: path.into(),
            source,
        }
    }

    /// Tags an image codec failure with the path that caused it.
    pub(crate) fn image(path: impl Into<PathBuf>, source: image::ImageError) -> Self {
        Self::Image {
            path: path.into(),
            source,
        }
    }
}
