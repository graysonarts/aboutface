//! The About:Face booth's wiring, as a library.
//!
//! The binary is a thin `main` over this: the tracer bullet — Shutter to
//! Corpus to wall — is exercised end to end by an integration test with the
//! fake `Camera` (ADR-0009), and a test cannot link a binary crate.

pub mod config;
pub mod pipeline;
pub mod selfcheck;
