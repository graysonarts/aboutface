//! Detect, align, and embed.
//!
//! One Visitor-initiated Capture goes in; one [`afcore::Embedding`] and a pair
//! of crops come out.
//!
//! - **Detect and align** with YuNet (MIT), which emits a bounding box and the
//!   five landmarks needed to warp the frame to a consistently aligned crop.
//! - **Embed** with DINOv2 (Apache 2.0). The piece measures *apparent*
//!   resemblance — colouring, hair, glasses, expression — not identity
//!   resemblance, and face-recognition weights are deliberately not used
//!   (ADR-0007).
//!
//! Two crops per Face, and they are not interchangeable: the aligned crop that
//! feeds the model, and a looser portrait crop for display. The original
//! full-quality frame is retained so that re-embedding stays possible
//! (ADR-0006).
//!
//! This crate owns the [`afcore::ModelId`] and selects the ONNX Runtime
//! execution provider at startup. The CPU path must remain viable; inference
//! runs once per Visitor, not per frame, so its latency budget is generous.
//!
//! The ONNX files are fetched, not committed — see `docs/models.md`. Where they
//! live and which identifier each carries come from `afbooth`'s configuration,
//! because which DINOv2 ViT size the piece runs is still open (ADR-0006).

mod model;
mod provider;

pub use model::{ModelError, ModelRole, ModelSpec};
pub use provider::{ExecutionProviderKind, runtime_info, select_execution_provider};
