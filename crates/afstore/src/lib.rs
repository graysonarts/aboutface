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

#![allow(dead_code, unused_imports)] // Stage 0 scaffold.
