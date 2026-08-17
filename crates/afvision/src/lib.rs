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
//!
//! # Shape
//!
//! - [`FaceDetector`] — YuNet over `ort`; hands back [`Faces`], which
//!   distinguishes none, one and several. There is no way to ask for "the
//!   face".
//! - [`align`] — the 112×112 crop the embedder sees, warped so the landmarks
//!   land on [`aligned_template`].
//! - [`display_crop`] — the portrait the wall shows, framed by
//!   [`DisplayCropSpec`] because the framing is an open question.
//! - [`FaceEmbedder`] — DINOv2 over `ort`; hands back an [`afcore::Embedding`]
//!   whose width came from the loaded graph and whose [`afcore::ModelId`] is
//!   the loaded file's, so comparisons across models are refused rather than
//!   degraded.
//!
//! The whole path runs off files on disk, with no camera and no booth:
//!
//! ```text
//! cargo run -p afvision --example detect_face -- samples/1.jpg
//! cargo run -p afvision --example embed_faces -- samples/1.jpg samples/3.jpg
//! ```

mod align;
mod detect;
mod embed;
mod geometry;
mod model;
mod provider;

pub use align::{
    ALIGNED_SIZE, DisplayCropError, DisplayCropSpec, align, aligned_template, display_crop,
};
pub use detect::{DetectError, Detection, FaceCountError, FaceDetector, Faces};
pub use embed::{EmbedError, FaceEmbedder};
pub use geometry::{BoundingBox, GeometryError, Landmarks, Point, SimilarityTransform};
pub use model::{ModelError, ModelRole, ModelSpec};
pub use provider::{ExecutionProviderKind, runtime_info, select_execution_provider};
