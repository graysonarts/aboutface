//! Domain types for About:Face.
//!
//! Vocabulary here follows `CONTEXT.md` exactly. This crate performs no I/O, no
//! inference and no rendering — everything else in the workspace depends on it,
//! and it depends on nothing but `thiserror`.

mod embedding;
mod grid;

pub use embedding::{Embedding, EmbeddingError, ModelId};
pub use grid::{
    Assignment, AssignmentError, CellIndex, GridSpec, GridSpecError, MAX_CELLS, MIN_CELLS,
};

/// Identifies one Face — one captured portrait of one Visitor at one moment.
///
/// The Face aggregate itself (crops, Embedding, Consent Record, capture time)
/// lands alongside the store schema in Stage 1; only its identity is needed
/// here.
///
/// A Visitor who returns twice produces two Faces with two different `FaceId`s.
/// The system deliberately does not link them; it measures similarity and never
/// identifies anyone (see `CONTEXT.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FaceId(pub u64);

impl std::fmt::Display for FaceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "face:{}", self.0)
    }
}
