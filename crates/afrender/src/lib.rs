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
//!
//! # Shape
//!
//! Stage 1 draws a still wall: a [`Window`] of Faces, one per Cell, in whatever
//! order they arrive. Ordering, Assignment and motion are Stage 2.
//!
//! - [`show`] — opens a window and draws until it closes.
//! - [`WallSpec`] — the Grid, the [`Framing`] of its Cells, the background.
//! - [`Portrait`] — one Face's display crop, named by path rather than decoded:
//!   this crate never opens the Corpus, and takes what `afbooth` reads out of
//!   it.
//!
//! Three things the wall does are arithmetic rather than graphics, and are
//! tested on machines with no GPU: [`Layout`] places Cells on a surface,
//! [`Window`] decides which Faces are on screen, and slot residency decides
//! which portraits are worth a texture.

mod app;
mod error;
mod geometry;
mod gpu;
mod portrait;
mod residency;
mod window;

pub use app::{WallSpec, show, show_live};
pub use error::RenderError;
pub use geometry::{CellRect, Framing, Layout};
pub use gpu::adapter_report;
pub use portrait::{Portrait, SLOT_HEIGHT, SLOT_WIDTH};
pub use residency::{Residency, Upload};
pub use window::Window;
