//! Camera access and the Shutter.
//!
//! Capture is deliberate and opt-in: the Visitor triggers the Shutter
//! themselves, and that gesture is simultaneously the consent and the exposure
//! (ADR-0005). This crate never captures on its own initiative.
//!
//! Hardware is deferred (ADR-0006), so camera access sits behind a trait with
//! `nokhwa` as the initial cross-platform implementation. **No other crate in
//! the workspace imports a camera API** — this is the one place a platform
//! choice is allowed to leak, and containing it here is the whole point of the
//! crate.

#![allow(dead_code, unused_imports)] // Stage 0 scaffold.
