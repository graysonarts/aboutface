//! The wall.
//!
//! `wgpu` and `winit` — no openFrameworks (ADR-0002). The rendering demand is
//! modest: up to [`afcore::MAX_CELLS`] textured quads with animated per-quad
//! transforms.
//!
//! Two kinds of motion, and they are visually distinct:
//!
//! - **Re-solve** — Faces interpolate from their old Cell to their new one as
//!   the crowd reorganizes around an arrival.
//! - **Drift** — Faces enter and leave the Window entirely as it wanders across
//!   the Corpus, which needs its own transition treatment (ADR-0004).
//!
//! Only Window-resident Faces need decoded textures, so the texture budget is
//! bounded by Grid size rather than by Corpus size.

#![allow(dead_code, unused_imports)] // Stage 0 scaffold.
