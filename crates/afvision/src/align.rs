//! The two crops, and why they are two.
//!
//! Every Face carries an *aligned* crop and a *display* crop, and they are not
//! interchangeable and never become so (implementation plan, "Key technical
//! choices"). The aligned crop is 112×112, warped so the five landmarks land on
//! a fixed template: the embedder sees faces in a consistent pose, and nothing
//! about it is meant to be looked at. The display crop is the photograph a
//! Visitor recognises — looser, framed by configuration rather than by the
//! model's needs.
//!
//! Whether display crops are square or portrait, and how tightly framed, is
//! still an open question in the implementation plan. Hence
//! [`DisplayCropSpec`]: the framing is a value the operator sets, not a
//! constant this module decides.

use image::RgbImage;
use image::imageops::FilterType;

use crate::geometry::{BoundingBox, GeometryError, Landmarks, Point, SimilarityTransform};

/// Side of the aligned crop, in pixels.
///
/// 112 is the size the alignment template is defined at; the embedder resizes
/// from here to whatever its own input wants.
pub const ALIGNED_SIZE: u32 = 112;

/// The canonical landmark positions inside a 112×112 aligned crop.
///
/// These are the five-point template that ArcFace-style alignment established
/// and that the ecosystem's aligned datasets are built on. The piece does not
/// use face-recognition weights (ADR-0007), but the template is still the
/// sensible target: eyes level, face centred, a consistent amount of head in
/// frame.
pub fn aligned_template() -> [Point; 5] {
    [
        Point::new(38.2946, 51.6963),
        Point::new(73.5318, 51.5014),
        Point::new(56.0252, 71.7366),
        Point::new(41.5493, 92.3655),
        Point::new(70.7299, 92.2041),
    ]
}

/// Warps `image` so the landmarks land on the canonical template.
///
/// A similarity transform only — rotation, uniform scale, translation. The
/// crop normalises pose without reshaping the person.
///
/// # Errors
///
/// Returns [`GeometryError::CoincidentPoints`] when the landmarks carry no
/// spread to fit a transform to.
pub fn align(image: &RgbImage, landmarks: &Landmarks) -> Result<RgbImage, GeometryError> {
    let to_template = SimilarityTransform::estimate(&landmarks.as_array(), &aligned_template())?;
    let to_source = to_template
        .inverse()
        .ok_or(GeometryError::CoincidentPoints)?;

    Ok(RgbImage::from_fn(ALIGNED_SIZE, ALIGNED_SIZE, |x, y| {
        // Sample at pixel centres: the destination pixel is an area, not a
        // point, and its centre is half a pixel in from its corner.
        let source = to_source.apply(Point::new(x as f32 + 0.5, y as f32 + 0.5));
        sample_bilinear(image, source.x - 0.5, source.y - 0.5)
    }))
}

/// Ways a display-crop framing can be unusable.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum DisplayCropError {
    /// A negative margin would frame inside the detection box.
    #[error("the display crop margin must not be negative, got {margin}")]
    NegativeMargin {
        /// The rejected margin.
        margin: f32,
    },

    /// An aspect ratio of zero or less describes no rectangle.
    #[error("the display crop aspect ratio must be positive, got {aspect_ratio}")]
    NonPositiveAspect {
        /// The rejected ratio.
        aspect_ratio: f32,
    },

    /// A crop no pixels wide is not an image.
    #[error("the display crop width must be at least one pixel")]
    ZeroWidth,
}

/// How the display crop is framed around a detection.
///
/// Configuration, not constants: how tightly a Visitor's portrait is framed is
/// an unresolved question in the implementation plan, and it is meant to be
/// settled by eye on the wall rather than in code.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisplayCropSpec {
    margin: f32,
    aspect_ratio: f32,
    width: u32,
    vertical_bias: f32,
}

impl Default for DisplayCropSpec {
    /// The house framing: a 4:5 portrait, loose enough to include hair and
    /// shoulders, lifted a little because a box drawn around a face sits low
    /// once the hair is in it.
    ///
    /// One source of truth. `booth.toml`'s `[display]` table and the
    /// `detect_face` example both start here and override what they need, so
    /// the numbers cannot drift apart between them.
    fn default() -> Self {
        Self {
            margin: 0.35,
            aspect_ratio: 0.8,
            width: 512,
            vertical_bias: 0.06,
        }
    }
}

impl DisplayCropSpec {
    /// A framing.
    ///
    /// `margin` expands the detection box by that fraction on every side;
    /// `aspect_ratio` is width over height, so 1.0 is square and 0.8 is a 4:5
    /// portrait; `width` is the rendered crop's width in pixels.
    ///
    /// # Errors
    ///
    /// Returns an error if the margin is negative, the aspect ratio is not
    /// positive, or the width is zero.
    pub fn new(margin: f32, aspect_ratio: f32, width: u32) -> Result<Self, DisplayCropError> {
        if margin < 0.0 {
            return Err(DisplayCropError::NegativeMargin { margin });
        }
        if aspect_ratio <= 0.0 {
            return Err(DisplayCropError::NonPositiveAspect { aspect_ratio });
        }
        if width == 0 {
            return Err(DisplayCropError::ZeroWidth);
        }

        Ok(Self {
            margin,
            aspect_ratio,
            width,
            vertical_bias: 0.0,
        })
    }

    /// Lifts the frame by that fraction of its own height.
    ///
    /// Faces sit low in a box drawn around them once hair and forehead are
    /// included, so a small upward bias is usually what looks right. Clamped
    /// to `0.0..=1.0`; out-of-range values are a framing preference, not an
    /// error worth refusing a Capture over.
    pub fn with_vertical_bias(mut self, bias: f32) -> Self {
        self.vertical_bias = bias.clamp(0.0, 1.0);
        self
    }

    /// The expansion applied to the detection box, as a fraction per side.
    pub fn margin(&self) -> f32 {
        self.margin
    }

    /// Width over height of the rendered crop.
    pub fn aspect_ratio(&self) -> f32 {
        self.aspect_ratio
    }

    /// Rendered width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Rendered height in pixels, from the width and the aspect ratio.
    pub fn height(&self) -> u32 {
        ((self.width as f32 / self.aspect_ratio).round() as u32).max(1)
    }

    /// How far the frame is lifted, as a fraction of its own height.
    pub fn vertical_bias(&self) -> f32 {
        self.vertical_bias
    }

    /// The region of the source image this framing selects.
    ///
    /// The frame always stays inside the image: it slides in from an edge when
    /// it can, and shrinks — keeping its aspect ratio — when the image is too
    /// small to hold it. A Visitor standing at the edge of frame gets a tighter
    /// portrait, never a band of invented pixels.
    pub fn frame(&self, face: &BoundingBox, image_width: u32, image_height: u32) -> BoundingBox {
        let (image_width, image_height) = (image_width as f32, image_height as f32);

        let mut width = face.width() * (1.0 + 2.0 * self.margin);
        let mut height = face.height() * (1.0 + 2.0 * self.margin);
        if width / height < self.aspect_ratio {
            width = height * self.aspect_ratio;
        } else {
            height = width / self.aspect_ratio;
        }

        // A zero-sized image would scale the frame to nothing, so the extents
        // are floored at a pixel: the crop stays a rectangle whatever arrives.
        let fit = (image_width / width).min(image_height / height).min(1.0);
        width = (width * fit).max(1.0);
        height = (height * fit).max(1.0);

        let center = face.center();
        let x = (center.x - width / 2.0).clamp(0.0, (image_width - width).max(0.0));
        let y = (center.y - height / 2.0 - self.vertical_bias * height)
            .clamp(0.0, (image_height - height).max(0.0));

        // INVARIANT: both extents were floored at one pixel above, so the box
        // is never empty and the constructor never rejects it.
        BoundingBox::new(x, y, width, height).unwrap_or(*face)
    }
}

/// Renders the display crop: the framed region, resampled to the spec's size.
pub fn display_crop(image: &RgbImage, face: &BoundingBox, spec: &DisplayCropSpec) -> RgbImage {
    let (image_width, image_height) = image.dimensions();
    let frame = spec.frame(face, image_width, image_height);

    let region = RgbImage::from_fn(
        (frame.width().round() as u32).max(1),
        (frame.height().round() as u32).max(1),
        |x, y| sample_bilinear(image, frame.x() + x as f32, frame.y() + y as f32),
    );

    image::imageops::resize(&region, spec.width(), spec.height(), FilterType::CatmullRom)
}

/// Samples the image at a continuous position, blending the four neighbours.
///
/// Positions outside the image clamp to the nearest edge pixel rather than
/// failing: the warp's corners can reach past the frame on a face near an edge,
/// and a Capture is not worth refusing over a few boundary pixels.
fn sample_bilinear(image: &RgbImage, x: f32, y: f32) -> image::Rgb<u8> {
    let (width, height) = image.dimensions();
    let max_x = (width - 1) as f32;
    let max_y = (height - 1) as f32;

    let x = x.clamp(0.0, max_x);
    let y = y.clamp(0.0, max_y);

    let x0 = x.floor();
    let y0 = y.floor();
    let (fx, fy) = (x - x0, y - y0);
    let x0 = x0 as u32;
    let y0 = y0 as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let mut channels = [0u8; 3];
    for (channel, value) in channels.iter_mut().enumerate() {
        let at = |px: u32, py: u32| image.get_pixel(px, py).0[channel] as f32;
        let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
        let bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
        *value = (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8;
    }

    image::Rgb(channels)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A source image whose pixels encode their own coordinates, so a sampled
    /// pixel says where it was sampled from.
    fn ramp(width: u32, height: u32) -> RgbImage {
        RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 0])
        })
    }

    fn bbox(x: f32, y: f32, width: f32, height: f32) -> BoundingBox {
        BoundingBox::new(x, y, width, height).expect("positive extent")
    }

    #[test]
    fn should_keep_the_eyes_level_and_inside_the_frame_in_the_canonical_template() {
        let template = aligned_template();

        assert!(
            (template[0].y - template[1].y).abs() < 1.0,
            "eyes not level"
        );
        for point in template {
            assert!(
                point.x > 0.0
                    && point.x < ALIGNED_SIZE as f32
                    && point.y > 0.0
                    && point.y < ALIGNED_SIZE as f32,
                "{point:?} outside the aligned crop"
            );
        }
    }

    #[test]
    fn should_produce_a_112_square_when_a_face_is_aligned() {
        let aligned = align(&ramp(300, 300), &Landmarks::from_array(aligned_template()))
            .expect("alignable landmarks");

        assert_eq!(aligned.dimensions(), (ALIGNED_SIZE, ALIGNED_SIZE));
    }

    #[test]
    fn should_copy_the_source_pixels_when_the_landmarks_already_sit_on_the_template() {
        let source = ramp(300, 300);

        let aligned = align(&source, &Landmarks::from_array(aligned_template()))
            .expect("alignable landmarks");

        for (x, y) in [(0, 0), (37, 51), (56, 71), (111, 111)] {
            let expected = source.get_pixel(x, y);
            let actual = aligned.get_pixel(x, y);
            assert!(
                actual.0[0].abs_diff(expected.0[0]) <= 2
                    && actual.0[1].abs_diff(expected.0[1]) <= 2,
                "aligned ({x}, {y}) = {actual:?}, source = {expected:?}"
            );
        }
    }

    #[test]
    fn should_halve_the_face_when_the_landmarks_are_twice_the_template() {
        let source = ramp(300, 300);
        let doubled = aligned_template().map(|point| Point::new(point.x * 2.0, point.y * 2.0));

        let aligned = align(&source, &Landmarks::from_array(doubled)).expect("alignable landmarks");

        for (x, y) in [(10, 10), (56, 56), (100, 100)] {
            let expected = source.get_pixel(x * 2, y * 2);
            let actual = aligned.get_pixel(x, y);
            assert!(
                actual.0[0].abs_diff(expected.0[0]) <= 3
                    && actual.0[1].abs_diff(expected.0[1]) <= 3,
                "aligned ({x}, {y}) = {actual:?}, source ({}, {}) = {expected:?}",
                x * 2,
                y * 2
            );
        }
    }

    #[test]
    fn should_refuse_to_align_when_the_landmarks_are_all_the_same_point() {
        let landmarks = Landmarks::from_array([Point::new(5.0, 5.0); 5]);

        assert_eq!(
            align(&ramp(64, 64), &landmarks),
            Err(GeometryError::CoincidentPoints)
        );
    }

    #[test]
    fn should_square_up_the_detection_box_when_the_spec_asks_for_a_square() {
        let spec = DisplayCropSpec::new(0.0, 1.0, 256).expect("valid spec");

        let frame = spec.frame(&bbox(100.0, 100.0, 50.0, 60.0), 500, 500);

        assert_eq!(
            (frame.x(), frame.y(), frame.width(), frame.height()),
            (95.0, 100.0, 60.0, 60.0)
        );
    }

    #[test]
    fn should_widen_the_frame_when_the_spec_asks_for_a_margin() {
        let spec = DisplayCropSpec::new(0.5, 1.0, 256).expect("valid spec");

        let frame = spec.frame(&bbox(100.0, 100.0, 50.0, 60.0), 500, 500);

        assert_eq!(
            (frame.x(), frame.y(), frame.width(), frame.height()),
            (65.0, 70.0, 120.0, 120.0)
        );
    }

    #[test]
    fn should_keep_the_portrait_aspect_when_the_spec_asks_for_one() {
        // 4:5 portrait around a square detection: the height grows, not the
        // width, so the framing never crops into the face.
        let spec = DisplayCropSpec::new(0.0, 0.8, 200).expect("valid spec");

        let frame = spec.frame(&bbox(100.0, 100.0, 80.0, 80.0), 1000, 1000);

        assert_eq!((frame.width(), frame.height()), (80.0, 100.0));
        assert_eq!(spec.height(), 250);
    }

    #[test]
    fn should_lift_the_frame_when_the_spec_biases_it_upwards() {
        let spec = DisplayCropSpec::new(0.0, 1.0, 256)
            .expect("valid spec")
            .with_vertical_bias(0.1);

        let frame = spec.frame(&bbox(100.0, 100.0, 50.0, 60.0), 500, 500);

        assert_eq!((frame.x(), frame.y()), (95.0, 94.0));
    }

    #[test]
    fn should_slide_the_frame_inside_the_image_when_the_face_is_against_an_edge() {
        let spec = DisplayCropSpec::new(0.5, 1.0, 256).expect("valid spec");

        let frame = spec.frame(&bbox(0.0, 0.0, 50.0, 50.0), 500, 500);

        assert_eq!((frame.x(), frame.y(), frame.width()), (0.0, 0.0, 100.0));
    }

    #[test]
    fn should_shrink_the_frame_when_the_margin_exceeds_the_image() {
        let spec = DisplayCropSpec::new(0.5, 1.0, 256).expect("valid spec");

        let frame = spec.frame(&bbox(0.0, 0.0, 50.0, 60.0), 100, 100);

        assert_eq!(
            (frame.x(), frame.y(), frame.width(), frame.height()),
            (0.0, 0.0, 100.0, 100.0)
        );
    }

    #[test]
    fn should_reject_a_spec_when_its_framing_numbers_are_impossible() {
        assert!(matches!(
            DisplayCropSpec::new(-0.1, 1.0, 256),
            Err(DisplayCropError::NegativeMargin { .. })
        ));
        assert!(matches!(
            DisplayCropSpec::new(0.0, 0.0, 256),
            Err(DisplayCropError::NonPositiveAspect { .. })
        ));
        assert!(matches!(
            DisplayCropSpec::new(0.0, 1.0, 0),
            Err(DisplayCropError::ZeroWidth)
        ));
        assert!(matches!(
            DisplayCropSpec::new(0.0, 1.0, 256)
                .expect("valid spec")
                .with_vertical_bias(2.0)
                .vertical_bias(),
            bias if (bias - 1.0).abs() < f32::EPSILON
        ));
    }

    #[test]
    fn should_render_the_display_crop_at_the_configured_size() {
        let spec = DisplayCropSpec::new(0.25, 0.8, 200).expect("valid spec");

        let crop = display_crop(&ramp(600, 600), &bbox(200.0, 200.0, 100.0, 100.0), &spec);

        assert_eq!(crop.dimensions(), (200, 250));
    }

    #[test]
    fn should_sample_the_framed_region_when_the_display_crop_is_rendered() {
        // A 1:1 crop rendered at its own pixel size is a plain copy, so the
        // top-left pixel must be the framed region's top-left pixel.
        let source = ramp(600, 600);
        let spec = DisplayCropSpec::new(0.0, 1.0, 100).expect("valid spec");

        let crop = display_crop(&source, &bbox(200.0, 300.0, 100.0, 100.0), &spec);

        let expected = source.get_pixel(200, 300);
        let actual = crop.get_pixel(0, 0);
        assert!(
            actual.0[0].abs_diff(expected.0[0]) <= 2 && actual.0[1].abs_diff(expected.0[1]) <= 2,
            "crop (0, 0) = {actual:?}, source (200, 300) = {expected:?}"
        );
    }
}
