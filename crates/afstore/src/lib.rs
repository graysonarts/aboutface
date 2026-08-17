//! The Corpus.
//!
//! Every Face ever retained, across every showing. It persists between
//! installations and grows without bound (ADR-0005).
//!
//! Responsibilities:
//!
//! - Faces, their Embeddings, their crops, and the original frames.
//! - The [`afcore::ModelId`] that produced each Embedding, so that a model
//!   change is detectable and a re-embed migration is possible (ADR-0006).
//! - Consent Records and Receipt Codes.
//! - **Deletion.** A Receipt Code destroys a Face, its Embedding, its images,
//!   and its Consent Record. This is a real, tested operation, not a manual
//!   database edit under pressure at an opening.
//!
//! The Corpus never leaves the installation machine: no cloud storage, no
//! telemetry, no network egress.
//!
//! # Shape
//!
//! A Corpus is a directory: `corpus.db` beside a tree of PNG files.
//!
//! - [`Corpus`] — open it, ingest into it, read Faces back out.
//! - [`NewFace`] — one Capture's worth of derived material, ready to store.
//!   The crops and the Embedding arrive already made: this crate performs no
//!   detection, alignment or inference, and never loads a model.
//! - [`StoredFace`] — what comes back out, with the paths of its three images.
//!
//! Consent Records and Receipt Codes are Stage 4; the schema leaves room for
//! them rather than pretending they are here.

mod blob;
mod corpus;
mod error;
mod face;
mod paths;
mod schema;

pub use corpus::Corpus;
pub use error::StoreError;
pub use face::{NewFace, StoredFace};
pub use schema::SCHEMA_VERSION;
