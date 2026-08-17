//! Finding the face, and refusing to guess which one.
//!
//! YuNet (MIT) detects and hands back the five landmarks alignment needs in one
//! step, which is why it was chosen (ADR-0001). It is an anchor-free detector
//! with three heads — strides 8, 16 and 32 — and each head predicts, per grid
//! cell, a class score, an objectness, a box offset and the five landmarks.
//! Decoding is therefore ours to do: the ONNX graph stops at the raw heads.
//!
//! **The frame's face count is a result, not a detail.** [`Faces`] distinguishes
//! none, one and several, and there is no way to ask this module for "the
//! face": the old C++ `face_isolator`'s silent `faces[0]` is the cautionary tale
//! the implementation plan names, and the multi-face policy at the Shutter is
//! still an open question. Deciding it here would settle it by accident.

use std::path::PathBuf;

use image::RgbImage;
use image::imageops::FilterType;
use ort::session::Session;
use ort::value::Tensor;

use crate::geometry::{BoundingBox, Landmarks, Point};
use crate::model::ModelSpec;
use crate::provider::ExecutionProviderKind;

/// The strides YuNet's three heads predict at.
const STRIDES: [u32; 3] = [8, 16, 32];

/// Square input side assumed when the graph does not fix one.
const DEFAULT_INPUT_SIZE: u32 = 640;

/// Detections below this score are dropped before suppression.
const DEFAULT_SCORE_THRESHOLD: f32 = 0.6;

/// Detections overlapping a stronger one by more than this are dropped.
const DEFAULT_NMS_THRESHOLD: f32 = 0.3;

/// One face YuNet found: where it is, and the five points to align it by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    score: f32,
    bbox: BoundingBox,
    landmarks: Landmarks,
}

impl Detection {
    /// Confidence, in `0.0..=1.0`.
    pub fn score(&self) -> f32 {
        self.score
    }

    /// The detection box, in source-image pixels.
    pub fn bbox(&self) -> &BoundingBox {
        &self.bbox
    }

    /// The five landmarks, in source-image pixels.
    pub fn landmarks(&self) -> &Landmarks {
        &self.landmarks
    }
}

/// What was in the frame.
///
/// Three outcomes, all of them nameable. A caller that wants exactly one face
/// says so with [`Faces::into_sole`]; a caller with a policy for crowds matches
/// on [`Faces::Many`] and applies it.
#[derive(Debug, Clone, PartialEq)]
pub enum Faces {
    /// Nobody was found. An empty booth, a Visitor out of frame, a dark room.
    None,
    /// Exactly one face.
    One(Detection),
    /// Several faces, strongest first. Which one — if any — the Capture belongs
    /// to is the caller's decision.
    Many(Vec<Detection>),
}

impl Faces {
    /// Sorts detections by score and classifies the frame.
    pub fn from_detections(mut detections: Vec<Detection>) -> Self {
        detections.sort_by(|a, b| b.score.total_cmp(&a.score));

        match detections.len() {
            0 => Self::None,
            1 => Self::One(detections.remove(0)),
            _ => Self::Many(detections),
        }
    }

    /// How many faces were found.
    pub fn count(&self) -> usize {
        match self {
            Self::None => 0,
            Self::One(_) => 1,
            Self::Many(detections) => detections.len(),
        }
    }

    /// Every detection, strongest first.
    pub fn detections(&self) -> &[Detection] {
        match self {
            Self::None => &[],
            Self::One(detection) => std::slice::from_ref(detection),
            Self::Many(detections) => detections,
        }
    }

    /// The single face in the frame.
    ///
    /// # Errors
    ///
    /// Returns [`FaceCountError`] when the frame held none or several. Both are
    /// ordinary outcomes at a booth, and both are the caller's to handle.
    pub fn into_sole(self) -> Result<Detection, FaceCountError> {
        match self {
            Self::One(detection) => Ok(detection),
            Self::None => Err(FaceCountError::NoFace),
            Self::Many(detections) => Err(FaceCountError::SeveralFaces {
                count: detections.len(),
            }),
        }
    }
}

/// Why a frame did not yield exactly one face.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FaceCountError {
    /// Nothing that looked like a face was in the frame.
    #[error("no face in the frame")]
    NoFace,

    /// More than one face. The policy for this at the Shutter is an open
    /// question in the implementation plan; nothing here decides it.
    #[error("{count} faces in the frame; which one the Capture belongs to is not decided here")]
    SeveralFaces {
        /// How many faces were found.
        count: usize,
    },
}

/// Ways detection can fail.
#[derive(Debug, thiserror::Error)]
pub enum DetectError {
    /// The ONNX file would not load into a Session.
    #[error("cannot open the detector at {path}: {source}")]
    Open {
        /// The model file that failed to load.
        path: PathBuf,
        /// The underlying ONNX Runtime failure.
        source: ort::Error,
    },

    /// Inference itself failed.
    #[error("the detector failed to run: {0}")]
    Inference(#[source] ort::Error),

    /// The graph is not the YuNet shape this decoder was written against.
    #[error("the detector's graph is not the expected YuNet shape: {detail}")]
    UnexpectedGraph {
        /// What was expected, and what was found.
        detail: String,
    },
}

/// YuNet, loaded and ready to be asked about a frame.
pub struct FaceDetector {
    session: Session,
    input_name: String,
    input_size: u32,
    score_threshold: f32,
    nms_threshold: f32,
}

impl FaceDetector {
    /// Loads the detector, running on the given execution provider.
    ///
    /// # Errors
    ///
    /// Returns an error if the file will not load, or if its graph is not the
    /// three-head YuNet shape this module decodes.
    pub fn open(spec: &ModelSpec, provider: ExecutionProviderKind) -> Result<Self, DetectError> {
        let open = || -> ort::Result<Session> {
            let mut builder =
                Session::builder()?.with_execution_providers([provider.dispatch()])?;
            builder.commit_from_file(spec.path())
        };
        let session = open().map_err(|source| DetectError::Open {
            path: spec.path().to_path_buf(),
            source,
        })?;

        let input = session
            .inputs()
            .first()
            .ok_or_else(|| DetectError::UnexpectedGraph {
                detail: "the model takes no inputs".to_owned(),
            })?;
        let input_name = input.name().to_owned();
        let input_size = square_input_size(input.dtype().tensor_shape().map(|shape| &shape[..]))?;

        for stride in STRIDES {
            for prefix in ["cls", "obj", "bbox", "kps"] {
                let name = format!("{prefix}_{stride}");
                if !session.outputs().iter().any(|out| out.name() == name) {
                    return Err(DetectError::UnexpectedGraph {
                        detail: format!("no output named {name}"),
                    });
                }
            }
        }

        Ok(Self {
            session,
            input_name,
            input_size,
            score_threshold: DEFAULT_SCORE_THRESHOLD,
            nms_threshold: DEFAULT_NMS_THRESHOLD,
        })
    }

    /// The square input side the graph declares.
    pub fn input_size(&self) -> u32 {
        self.input_size
    }

    /// Finds every face in `image`, in source-image coordinates.
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails or an output is not the expected
    /// shape.
    pub fn detect(&mut self, image: &RgbImage) -> Result<Faces, DetectError> {
        let (width, height) = image.dimensions();
        let letterbox = Letterbox::new(width, height, self.input_size);
        let blob = to_blob(&letterbox.place(image));
        let side = i64::from(self.input_size);

        let tensor =
            Tensor::from_array((vec![1, 3, side, side], blob)).map_err(DetectError::Inference)?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(DetectError::Inference)?;

        let mut detections = Vec::new();
        for stride in STRIDES {
            let cols = self.input_size / stride;
            let head = Head {
                cls: extract(&outputs, &format!("cls_{stride}"))?,
                obj: extract(&outputs, &format!("obj_{stride}"))?,
                bbox: extract(&outputs, &format!("bbox_{stride}"))?,
                kps: extract(&outputs, &format!("kps_{stride}"))?,
            };
            head.check(stride, cols)?;
            detections.extend(decode_head(
                &head,
                stride,
                cols,
                self.score_threshold,
                &letterbox,
            ));
        }

        Ok(Faces::from_detections(suppress(
            detections,
            self.nms_threshold,
        )))
    }
}

/// One head's four output tensors, borrowed.
struct Head<'a> {
    cls: &'a [f32],
    obj: &'a [f32],
    bbox: &'a [f32],
    kps: &'a [f32],
}

impl Head<'_> {
    /// Checks the head's four tensors against the grid they claim to cover.
    ///
    /// A short tensor would otherwise lose cells silently, which is the
    /// failure mode this module exists to avoid: a graph that is not the one
    /// this decoder was written against should say so, not quietly find fewer
    /// faces.
    fn check(&self, stride: u32, cols: u32) -> Result<(), DetectError> {
        let cells = (cols * cols) as usize;
        for (name, actual, expected) in [
            ("cls", self.cls.len(), cells),
            ("obj", self.obj.len(), cells),
            ("bbox", self.bbox.len(), cells * 4),
            ("kps", self.kps.len(), cells * 10),
        ] {
            if actual != expected {
                return Err(DetectError::UnexpectedGraph {
                    detail: format!(
                        "output {name}_{stride} holds {actual} values, expected {expected} for a \
                         {cols}x{cols} grid"
                    ),
                });
            }
        }

        Ok(())
    }
}

fn extract<'a>(
    outputs: &'a ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<&'a [f32], DetectError> {
    outputs
        .get(name)
        .ok_or_else(|| DetectError::UnexpectedGraph {
            detail: format!("no output named {name}"),
        })?
        .try_extract_tensor::<f32>()
        .map(|(_, data)| data)
        .map_err(|source| DetectError::UnexpectedGraph {
            detail: format!("output {name} is not a float tensor: {source}"),
        })
}

/// The graph's square input side.
fn square_input_size(shape: Option<&[i64]>) -> Result<u32, DetectError> {
    let Some(shape) = shape else {
        return Ok(DEFAULT_INPUT_SIZE);
    };

    match shape {
        [_, 3, height, width] => {
            // A dynamic dimension is reported as -1; the 2023mar export fixes
            // both at 640, and a rectangular input would invalidate the
            // square-grid decode below.
            if *height <= 0 || *width <= 0 {
                Ok(DEFAULT_INPUT_SIZE)
            } else if height == width {
                Ok(*height as u32)
            } else {
                Err(DetectError::UnexpectedGraph {
                    detail: format!("input is {width}x{height}; a square input is expected"),
                })
            }
        }
        other => Err(DetectError::UnexpectedGraph {
            detail: format!("input shape is {other:?}, expected [1, 3, size, size]"),
        }),
    }
}

/// Fits an image into a square input without distorting it.
///
/// YuNet's input side is fixed, and stretching a wide frame to a square would
/// move the landmarks relative to the face. Scaling by the tighter axis and
/// padding the remainder keeps the geometry, and undoing it is one division.
struct Letterbox {
    scale: f32,
    target: u32,
}

impl Letterbox {
    /// The mapping that fits a `width`x`height` image into a `target` square.
    fn new(width: u32, height: u32, target: u32) -> Self {
        let scale = (target as f32 / width as f32).min(target as f32 / height as f32);

        Self { scale, target }
    }

    /// Maps a point in the padded square back to source-image pixels.
    fn to_source(&self, point: Point) -> Point {
        Point::new(point.x / self.scale, point.y / self.scale)
    }

    /// Renders the source into the padded square, anchored at the top left.
    fn place(&self, image: &RgbImage) -> RgbImage {
        let (width, height) = image.dimensions();
        let scaled_width = ((width as f32 * self.scale).round() as u32).clamp(1, self.target);
        let scaled_height = ((height as f32 * self.scale).round() as u32).clamp(1, self.target);
        let scaled =
            image::imageops::resize(image, scaled_width, scaled_height, FilterType::Triangle);

        let mut canvas = RgbImage::new(self.target, self.target);
        image::imageops::replace(&mut canvas, &scaled, 0, 0);

        canvas
    }
}

/// Turns the padded square into the NCHW BGR float tensor YuNet expects.
///
/// No normalisation: the reference implementation feeds raw 0–255 values, and
/// the network's first convolution absorbs the scale.
fn to_blob(image: &RgbImage) -> Vec<f32> {
    let (width, height) = image.dimensions();
    let pixels = (width * height) as usize;
    let mut blob = vec![0.0; pixels * 3];

    for (index, pixel) in image.pixels().enumerate() {
        blob[index] = f32::from(pixel.0[2]);
        blob[pixels + index] = f32::from(pixel.0[1]);
        blob[2 * pixels + index] = f32::from(pixel.0[0]);
    }

    blob
}

/// Turns one head's raw cells into detections in source-image pixels.
///
/// Anchor-free: a cell's own position is the anchor. The box centre is the cell
/// origin plus a predicted offset, both in cell units; the extent is predicted
/// in log space, which is why it is exponentiated. The score is the geometric
/// mean of the class score and the objectness, as the reference decoder does it.
fn decode_head(
    head: &Head<'_>,
    stride: u32,
    cols: u32,
    score_threshold: f32,
    letterbox: &Letterbox,
) -> Vec<Detection> {
    let stride = stride as f32;
    let mut detections = Vec::new();

    for (index, (class, objectness)) in head.cls.iter().zip(head.obj).enumerate() {
        let score = (class.clamp(0.0, 1.0) * objectness.clamp(0.0, 1.0)).sqrt();
        if score < score_threshold {
            continue;
        }

        let (Some(box_offsets), Some(landmark_offsets)) = (
            head.bbox.get(index * 4..index * 4 + 4),
            head.kps.get(index * 10..index * 10 + 10),
        ) else {
            continue;
        };

        let column = (index as u32 % cols) as f32;
        let row = (index as u32 / cols) as f32;

        let center_x = (column + box_offsets[0]) * stride;
        let center_y = (row + box_offsets[1]) * stride;
        let width = box_offsets[2].exp() * stride;
        let height = box_offsets[3].exp() * stride;

        let top_left =
            letterbox.to_source(Point::new(center_x - width / 2.0, center_y - height / 2.0));
        let bottom_right =
            letterbox.to_source(Point::new(center_x + width / 2.0, center_y + height / 2.0));

        let Ok(bbox) = BoundingBox::new(
            top_left.x,
            top_left.y,
            bottom_right.x - top_left.x,
            bottom_right.y - top_left.y,
        ) else {
            continue;
        };

        let mut points = [Point::new(0.0, 0.0); 5];
        for (point, offsets) in points.iter_mut().zip(landmark_offsets.chunks_exact(2)) {
            *point = letterbox.to_source(Point::new(
                (column + offsets[0]) * stride,
                (row + offsets[1]) * stride,
            ));
        }

        detections.push(Detection {
            score,
            bbox,
            landmarks: Landmarks::from_array(points),
        });
    }

    detections
}

/// Non-maximum suppression: one detection per face, not one per grid cell.
///
/// The three heads fire on overlapping cells around the same face, so a single
/// Visitor arrives as a cluster. The strongest detection in a cluster wins and
/// swallows everything overlapping it.
fn suppress(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    detections.sort_by(|a, b| b.score.total_cmp(&a.score));

    let mut kept: Vec<Detection> = Vec::new();
    for detection in detections {
        let overlaps = kept
            .iter()
            .any(|other| other.bbox.intersection_over_union(&detection.bbox) > iou_threshold);
        if !overlaps {
            kept.push(detection);
        }
    }

    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detection(score: f32, x: f32, y: f32, size: f32) -> Detection {
        Detection {
            score,
            bbox: BoundingBox::new(x, y, size, size).expect("positive extent"),
            landmarks: Landmarks::from_array([Point::new(x, y); 5]),
        }
    }

    /// Owns one stride's head outputs so a [`Head`] can borrow them.
    struct HeadBuffers {
        cls: Vec<f32>,
        obj: Vec<f32>,
        bbox: Vec<f32>,
        kps: Vec<f32>,
    }

    impl HeadBuffers {
        fn head(&self) -> Head<'_> {
            Head {
                cls: &self.cls,
                obj: &self.obj,
                bbox: &self.bbox,
                kps: &self.kps,
            }
        }
    }

    /// One stride's worth of head outputs, with a single cell lit up.
    fn head(cols: usize, cell: usize, score: f32, bbox: [f32; 4], kps: [f32; 10]) -> HeadBuffers {
        let cells = cols * cols;
        let mut head = HeadBuffers {
            cls: vec![0.0; cells],
            obj: vec![0.0; cells],
            bbox: vec![0.0; cells * 4],
            kps: vec![0.0; cells * 10],
        };
        // score = sqrt(cls * obj), so a unit objectness makes cls the square.
        head.cls[cell] = score * score;
        head.obj[cell] = 1.0;
        head.bbox[cell * 4..cell * 4 + 4].copy_from_slice(&bbox);
        head.kps[cell * 10..cell * 10 + 10].copy_from_slice(&kps);
        head
    }

    #[test]
    fn should_scale_by_the_tighter_axis_when_letterboxing_a_wide_image() {
        let letterbox = Letterbox::new(1000, 500, 640);

        assert!((letterbox.scale - 0.64).abs() < 1e-6);
    }

    #[test]
    fn should_map_a_letterboxed_point_back_to_the_source_image() {
        let letterbox = Letterbox::new(1000, 500, 640);

        let source = letterbox.to_source(Point::new(64.0, 32.0));

        assert!(
            (source.x - 100.0).abs() < 1e-3 && (source.y - 50.0).abs() < 1e-3,
            "{source:?}"
        );
    }

    #[test]
    fn should_pad_rather_than_stretch_when_letterboxing() {
        // A 200x100 image scaled to fit 100x100 occupies the top 100x50; the
        // rest is padding, and the aspect ratio is untouched.
        let letterbox = Letterbox::new(200, 100, 100);
        let source = RgbImage::from_pixel(200, 100, image::Rgb([200, 100, 50]));

        let placed = letterbox.place(&source);

        assert_eq!(placed.dimensions(), (100, 100));
        assert_eq!(placed.get_pixel(50, 20), &image::Rgb([200, 100, 50]));
        assert_eq!(placed.get_pixel(50, 80), &image::Rgb([0, 0, 0]));
    }

    #[test]
    fn should_decode_a_cell_into_pixels_when_it_scores_above_the_threshold() {
        // Cell (col 3, row 2) at stride 32: centre (3.5, 2.5) cells, extent
        // e^ln2 and e^ln3 cells, so 64x96 pixels centred on (112, 80).
        let head = head(
            20,
            2 * 20 + 3,
            0.9,
            [0.5, 0.5, 2.0f32.ln(), 3.0f32.ln()],
            [0.25, 0.25, 0.75, 0.25, 0.5, 0.5, 0.25, 0.75, 0.75, 0.75],
        );

        let decoded = decode_head(&head.head(), 32, 20, 0.5, &Letterbox::new(640, 640, 640));

        assert_eq!(decoded.len(), 1);
        let face = &decoded[0];
        assert!((face.score() - 0.9).abs() < 1e-4);
        assert_eq!(
            (
                face.bbox().x(),
                face.bbox().y(),
                face.bbox().width(),
                face.bbox().height()
            ),
            (80.0, 32.0, 64.0, 96.0)
        );
        assert_eq!(face.landmarks().left_eye, Point::new(104.0, 72.0));
        assert_eq!(face.landmarks().mouth_right, Point::new(120.0, 88.0));
    }

    #[test]
    fn should_rescale_a_decoded_cell_when_the_image_was_letterboxed() {
        let head = head(20, 0, 1.0, [0.5, 0.5, 0.0, 0.0], [0.0; 10]);

        // A 1280x1280 image is halved to fit 640, so decoded pixels double.
        let decoded = decode_head(&head.head(), 32, 20, 0.5, &Letterbox::new(1280, 1280, 640));

        assert_eq!(decoded[0].bbox().center(), Point::new(32.0, 32.0));
    }

    #[test]
    fn should_ignore_a_cell_when_it_scores_below_the_threshold() {
        let head = head(20, 5, 0.2, [0.0, 0.0, 0.0, 0.0], [0.0; 10]);

        let decoded = decode_head(&head.head(), 32, 20, 0.5, &Letterbox::new(640, 640, 640));

        assert!(decoded.is_empty());
    }

    #[test]
    fn should_reject_a_head_when_a_tensor_is_shorter_than_its_grid() {
        let mut head = head(20, 0, 1.0, [0.0; 4], [0.0; 10]);
        head.kps.truncate(10);

        let error = head.head().check(32, 20).expect_err("short tensor");

        assert!(
            matches!(error, DetectError::UnexpectedGraph { .. }),
            "{error}"
        );
    }

    #[test]
    fn should_accept_a_head_when_every_tensor_covers_its_grid() {
        let head = head(20, 0, 1.0, [0.0; 4], [0.0; 10]);

        assert!(head.head().check(32, 20).is_ok());
    }

    #[test]
    fn should_keep_the_strongest_when_overlapping_detections_are_suppressed() {
        let detections = vec![
            detection(0.7, 0.0, 0.0, 10.0),
            detection(0.9, 1.0, 1.0, 10.0),
            detection(0.8, 2.0, 2.0, 10.0),
        ];

        let kept = suppress(detections, 0.3);

        assert_eq!(kept.len(), 1);
        assert!((kept[0].score() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn should_keep_both_when_two_faces_do_not_overlap() {
        let detections = vec![
            detection(0.9, 0.0, 0.0, 10.0),
            detection(0.8, 50.0, 0.0, 10.0),
        ];

        let kept = suppress(detections, 0.3);

        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn should_report_no_face_when_nothing_was_detected() {
        assert_eq!(Faces::from_detections(Vec::new()), Faces::None);
    }

    #[test]
    fn should_report_one_face_when_exactly_one_was_detected() {
        let faces = Faces::from_detections(vec![detection(0.9, 0.0, 0.0, 10.0)]);

        assert!(matches!(faces, Faces::One(_)));
        assert_eq!(faces.count(), 1);
    }

    #[test]
    fn should_report_every_face_in_score_order_when_several_were_detected() {
        let faces = Faces::from_detections(vec![
            detection(0.5, 0.0, 0.0, 10.0),
            detection(0.9, 50.0, 0.0, 10.0),
            detection(0.7, 100.0, 0.0, 10.0),
        ]);

        match &faces {
            Faces::Many(detections) => {
                let scores: Vec<f32> = detections.iter().map(Detection::score).collect();
                assert_eq!(scores, vec![0.9, 0.7, 0.5]);
            }
            other => panic!("expected several faces, got {other:?}"),
        }
    }

    #[test]
    fn should_refuse_to_pick_a_face_when_the_frame_holds_more_than_one() {
        let faces = Faces::from_detections(vec![
            detection(0.9, 0.0, 0.0, 10.0),
            detection(0.8, 50.0, 0.0, 10.0),
        ]);

        assert_eq!(
            faces.into_sole(),
            Err(FaceCountError::SeveralFaces { count: 2 })
        );
    }

    #[test]
    fn should_refuse_to_pick_a_face_when_the_frame_holds_none() {
        assert_eq!(Faces::None.into_sole(), Err(FaceCountError::NoFace));
    }

    #[test]
    fn should_hand_back_the_face_when_there_is_exactly_one() {
        let faces = Faces::from_detections(vec![detection(0.9, 3.0, 4.0, 10.0)]);

        let face = faces.into_sole().expect("one face");

        assert_eq!(face.bbox().x(), 3.0);
    }
}
