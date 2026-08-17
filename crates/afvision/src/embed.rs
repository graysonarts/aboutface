//! Turning an aligned crop into an [`afcore::Embedding`].
//!
//! DINOv2 (Apache 2.0) measures *apparent* resemblance — colouring, hair,
//! glasses, expression — rather than identity (ADR-0007). The graph is a plain
//! ViT: `pixel_values` in, `last_hidden_state` out, one token per patch plus a
//! leading CLS token. The CLS token is the image-level summary and is what this
//! module keeps (ADR-0010).
//!
//! **Nothing here knows how wide an Embedding is.** The ViT size follows the
//! hardware decision (ADR-0006) and the widths differ between sizes, so the
//! width is discovered from the loaded graph at open time and the [`ModelId`]
//! recorded on every Embedding is the one the loaded file resolved to — never a
//! default. That is what lets the Corpus refuse to compare across models
//! instead of returning a degraded number.

use std::path::PathBuf;

use afcore::{Embedding, EmbeddingError, ModelId};
use image::RgbImage;
use image::imageops::FilterType;
use ort::session::Session;
use ort::value::Tensor;

use crate::model::ModelSpec;
use crate::provider::ExecutionProviderKind;

/// DINOv2's patch side. The input must be a whole number of patches.
const PATCH_SIZE: u32 = 14;

/// Square input side used when the graph leaves its spatial dimensions dynamic.
///
/// The onnx-community exports keep height and width symbolic, so a size has to
/// be chosen: 224 is what the model was trained and evaluated at, and it is
/// 16×16 patches exactly.
const DEFAULT_INPUT_SIZE: u32 = 224;

/// The ImageNet channel means DINOv2's preprocessing subtracts.
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// The ImageNet channel standard deviations DINOv2's preprocessing divides by.
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Ways embedding can fail.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    /// The ONNX file would not load into a Session.
    #[error("cannot open the embedder at {path}: {source}")]
    Open {
        /// The model file that failed to load.
        path: PathBuf,
        /// The underlying ONNX Runtime failure.
        source: ort::Error,
    },

    /// Inference itself failed.
    #[error("the embedder failed to run: {0}")]
    Inference(#[source] ort::Error),

    /// The graph is not the ViT shape this module was written against.
    #[error("the embedder's graph is not the expected DINOv2 shape: {detail}")]
    UnexpectedGraph {
        /// What was expected, and what was found.
        detail: String,
    },

    /// Inference produced something that is not a usable Embedding.
    #[error("the embedder produced an unusable vector: {0}")]
    Unusable(#[from] EmbeddingError),
}

/// DINOv2, loaded and ready to embed aligned crops.
pub struct FaceEmbedder {
    session: Session,
    input_name: String,
    output_name: String,
    input_size: u32,
    model: ModelId,
    dim: usize,
}

impl FaceEmbedder {
    /// Loads the embedder, running on the given execution provider.
    ///
    /// Opening includes one inference over a blank crop. It settles the
    /// Embedding width by observation rather than by assumption, and it pays
    /// the graph's first-run cost here instead of on the first Visitor.
    ///
    /// # Errors
    ///
    /// Returns an error if the file will not load, if its graph is not a ViT
    /// taking a square `pixel_values` tensor, or if the probe inference fails.
    pub fn open(spec: &ModelSpec, provider: ExecutionProviderKind) -> Result<Self, EmbedError> {
        let open = || -> ort::Result<Session> {
            let mut builder =
                Session::builder()?.with_execution_providers([provider.dispatch()])?;
            builder.commit_from_file(spec.path())
        };
        let session = open().map_err(|source| EmbedError::Open {
            path: spec.path().to_path_buf(),
            source,
        })?;

        let input = session
            .inputs()
            .first()
            .ok_or_else(|| EmbedError::UnexpectedGraph {
                detail: "the model takes no inputs".to_owned(),
            })?;
        let input_name = input.name().to_owned();
        let input_size = square_input_size(input.dtype().tensor_shape().map(|shape| &shape[..]))?;

        let output_name = session
            .outputs()
            .first()
            .ok_or_else(|| EmbedError::UnexpectedGraph {
                detail: "the model produces no outputs".to_owned(),
            })?
            .name()
            .to_owned();

        let mut session = session;
        let blank = RgbImage::new(input_size, input_size);
        let dim = summary(&mut session, &input_name, &output_name, input_size, &blank)?.len();

        Ok(Self {
            session,
            input_name,
            output_name,
            input_size,
            model: spec.id().clone(),
            dim,
        })
    }

    /// The square input side this embedder feeds the graph.
    pub fn input_size(&self) -> u32 {
        self.input_size
    }

    /// The identifier recorded on every Embedding this embedder produces.
    pub fn model(&self) -> &ModelId {
        &self.model
    }

    /// The Embedding width this graph produces, discovered when it was opened.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Embeds an aligned crop.
    ///
    /// The crop is resized to the graph's input; it is expected to be the
    /// 112×112 aligned crop, not the display crop, and not a raw frame.
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails, if the output is not the shape the
    /// probe established, or if the vector cannot be normalized.
    pub fn embed(&mut self, aligned: &RgbImage) -> Result<Embedding, EmbedError> {
        let values = summary(
            &mut self.session,
            &self.input_name,
            &self.output_name,
            self.input_size,
            aligned,
        )?;
        if values.len() != self.dim {
            return Err(EmbedError::UnexpectedGraph {
                detail: format!(
                    "the graph produced {} dimensions, having produced {} when it was opened",
                    values.len(),
                    self.dim
                ),
            });
        }

        Ok(Embedding::new(self.model.clone(), values)?)
    }
}

/// Runs the graph and returns the CLS token: one vector per image.
///
/// `last_hidden_state` is `[1, patches + 1, width]` and the leading token is the
/// CLS summary (ADR-0010). The width is whatever the loaded graph produced — it
/// is read off the output, never assumed.
///
/// Free rather than a method because [`FaceEmbedder::open`] calls it to settle
/// `dim`, and a half-built embedder with a placeholder width is exactly the
/// invariant this crate exists to not have.
fn summary(
    session: &mut Session,
    input_name: &str,
    output_name: &str,
    input_size: u32,
    image: &RgbImage,
) -> Result<Vec<f32>, EmbedError> {
    let side = i64::from(input_size);
    let blob = preprocess(image, input_size);
    let tensor =
        Tensor::from_array((vec![1, 3, side, side], blob)).map_err(EmbedError::Inference)?;

    let outputs = session
        .run(ort::inputs![input_name => tensor])
        .map_err(EmbedError::Inference)?;
    let (shape, data) = outputs
        .get(output_name)
        .ok_or_else(|| EmbedError::UnexpectedGraph {
            detail: format!("no output named {output_name}"),
        })?
        .try_extract_tensor::<f32>()
        .map_err(|source| EmbedError::UnexpectedGraph {
            detail: format!("output {output_name} is not a float tensor: {source}"),
        })?;

    let width = match &shape[..] {
        [1, tokens, width] if *tokens > 0 && *width > 0 => *width as usize,
        other => {
            return Err(EmbedError::UnexpectedGraph {
                detail: format!("output shape is {other:?}, expected [1, tokens, width]"),
            });
        }
    };

    data.get(..width)
        .map(<[f32]>::to_vec)
        .ok_or_else(|| EmbedError::UnexpectedGraph {
            detail: format!(
                "output holds {} values, too few for a {width}-wide token",
                data.len()
            ),
        })
}

/// Turns a crop into the NCHW RGB float tensor DINOv2 expects.
///
/// Resize to the graph's square input, scale to `0.0..=1.0`, then subtract the
/// ImageNet channel means and divide by their standard deviations — the
/// preprocessing DINOv2 was trained with. Channel-planar, red plane first.
///
/// Bicubic, as `facebook/dinov2-*`'s image processor specifies. The reference
/// also resizes the shortest edge to 256 and centre-crops to 224; that step is
/// deliberately skipped, because the input here is already the aligned crop and
/// cropping it again would cut the face (ADR-0010).
fn preprocess(image: &RgbImage, size: u32) -> Vec<f32> {
    let resized = image::imageops::resize(image, size, size, FilterType::CatmullRom);
    let plane = (size * size) as usize;
    let mut blob = vec![0.0; plane * 3];

    for (index, pixel) in resized.pixels().enumerate() {
        for channel in 0..3 {
            blob[channel * plane + index] = (f32::from(pixel.0[channel]) / 255.0
                - IMAGENET_MEAN[channel])
                / IMAGENET_STD[channel];
        }
    }

    blob
}

/// The square input side the graph declares, or the default when it is dynamic.
fn square_input_size(shape: Option<&[i64]>) -> Result<u32, EmbedError> {
    let Some(shape) = shape else {
        return Ok(DEFAULT_INPUT_SIZE);
    };

    match shape {
        // A symbolic dimension is reported as -1, and the onnx-community
        // exports leave every dimension of `pixel_values` symbolic — channels
        // included. A graph that fixes its channel count to something other
        // than three is not an RGB ViT and is rejected.
        [_, channels, height, width] if *channels == 3 || *channels <= 0 => {
            // The ViT accepts any whole number of patches, so a dynamic
            // spatial dimension is the ordinary case, not a defect.
            if *height <= 0 || *width <= 0 {
                Ok(DEFAULT_INPUT_SIZE)
            } else if height == width {
                let side = *height as u32;
                if !side.is_multiple_of(PATCH_SIZE) {
                    return Err(EmbedError::UnexpectedGraph {
                        detail: format!(
                            "input side {side} is not a whole number of {PATCH_SIZE}px patches"
                        ),
                    });
                }
                Ok(side)
            } else {
                Err(EmbedError::UnexpectedGraph {
                    detail: format!("input is {width}x{height}; a square input is expected"),
                })
            }
        }
        other => Err(EmbedError::UnexpectedGraph {
            detail: format!("input shape is {other:?}, expected [1, 3, size, size]"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(color: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(4, 4, image::Rgb(color))
    }

    fn normalized(value: u8, channel: usize) -> f32 {
        (f32::from(value) / 255.0 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
    }

    #[test]
    fn should_resize_to_the_graph_input_when_preprocessing() {
        let blob = preprocess(&solid([10, 20, 30]), 28);

        assert_eq!(blob.len(), 3 * 28 * 28);
    }

    #[test]
    fn should_lay_channels_out_in_planes_with_red_first_when_preprocessing() {
        let blob = preprocess(&solid([255, 0, 0]), 2);

        let plane = 2 * 2;
        assert!((blob[0] - normalized(255, 0)).abs() < 1e-5, "{}", blob[0]);
        assert!(
            (blob[plane] - normalized(0, 1)).abs() < 1e-5,
            "{}",
            blob[plane]
        );
        assert!(
            (blob[2 * plane] - normalized(0, 2)).abs() < 1e-5,
            "{}",
            blob[2 * plane]
        );
    }

    #[test]
    fn should_scale_to_imagenet_statistics_when_preprocessing() {
        let blob = preprocess(&solid([128, 128, 128]), 2);

        for (index, channel) in [(0usize, 0usize), (4, 1), (8, 2)] {
            assert!(
                (blob[index] - normalized(128, channel)).abs() < 1e-5,
                "channel {channel} was {}",
                blob[index]
            );
        }
    }

    #[test]
    fn should_take_the_declared_side_when_the_graph_fixes_a_square_input() {
        assert_eq!(square_input_size(Some(&[1, 3, 518, 518])).unwrap(), 518);
    }

    #[test]
    fn should_fall_back_to_the_default_side_when_the_graph_is_dynamic() {
        assert_eq!(
            square_input_size(Some(&[-1, 3, -1, -1])).unwrap(),
            DEFAULT_INPUT_SIZE
        );
        assert_eq!(square_input_size(None).unwrap(), DEFAULT_INPUT_SIZE);
    }

    #[test]
    fn should_accept_the_graph_when_even_its_channel_count_is_symbolic() {
        // What the onnx-community DINOv2 exports actually declare.
        assert_eq!(
            square_input_size(Some(&[-1, -1, -1, -1])).unwrap(),
            DEFAULT_INPUT_SIZE
        );
    }

    #[test]
    fn should_reject_the_graph_when_it_fixes_a_channel_count_that_is_not_rgb() {
        let error = square_input_size(Some(&[1, 1, 224, 224])).expect_err("greyscale input");

        assert!(
            matches!(error, EmbedError::UnexpectedGraph { .. }),
            "{error}"
        );
    }

    #[test]
    fn should_reject_the_graph_when_its_input_is_not_square() {
        let error = square_input_size(Some(&[1, 3, 224, 168])).expect_err("rectangular input");

        assert!(
            matches!(error, EmbedError::UnexpectedGraph { .. }),
            "{error}"
        );
    }

    #[test]
    fn should_reject_the_graph_when_its_input_is_not_an_image_tensor() {
        let error = square_input_size(Some(&[1, 224, 224])).expect_err("wrong rank");

        assert!(
            matches!(error, EmbedError::UnexpectedGraph { .. }),
            "{error}"
        );
    }

    #[test]
    fn should_reject_the_graph_when_its_input_is_not_a_whole_number_of_patches() {
        let error = square_input_size(Some(&[1, 3, 225, 225])).expect_err("ragged input");

        assert!(
            matches!(error, EmbedError::UnexpectedGraph { .. }),
            "{error}"
        );
    }

    #[test]
    fn should_keep_the_default_input_a_whole_number_of_patches() {
        assert!(DEFAULT_INPUT_SIZE.is_multiple_of(PATCH_SIZE));
    }
}
