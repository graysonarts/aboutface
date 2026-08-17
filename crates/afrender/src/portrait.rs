//! Turning a Face's display crop into texels.
//!
//! One Face, one portrait, one texture slot. Every slot in the array is the
//! same size, so a display crop is fitted to it here rather than by stretching
//! the quad it lands on: a Visitor's face is not restretched to whatever shape
//! the Grid happens to be.

use std::path::PathBuf;

use afcore::FaceId;
use image::imageops::FilterType;
use image::{RgbImage, RgbaImage};

use crate::error::RenderError;

/// The size of one texture slot, in texels.
///
/// Large enough that a ten-Cell Grid on a 4K wall is not visibly soft, small
/// enough that a thousand of them is a few hundred megabytes rather than a few
/// gigabytes. It follows the 4:5 framing `booth.toml` crops portraits at.
pub const SLOT_WIDTH: u32 = 256;

/// The height of one texture slot, in texels. See [`SLOT_WIDTH`].
pub const SLOT_HEIGHT: u32 = 320;

/// A Face's display crop, on disk.
///
/// The path rather than the pixels: the wall decodes on its own schedule, and
/// only for Faces the Window actually reaches (ADR-0004).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Portrait {
    face: FaceId,
    display_crop: PathBuf,
}

impl Portrait {
    /// Names the display crop belonging to `face`.
    pub fn new(face: FaceId, display_crop: impl Into<PathBuf>) -> Self {
        Self {
            face,
            display_crop: display_crop.into(),
        }
    }

    /// Which Face this portrait shows.
    pub fn face(&self) -> FaceId {
        self.face
    }

    /// Decodes it into one texture slot's worth of texels.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or is not an image.
    pub(crate) fn decode(&self) -> Result<RgbaImage, RenderError> {
        let image = image::open(&self.display_crop)
            .map_err(|source| RenderError::image(&self.display_crop, source))?;

        Ok(fit_to_slot(&image.to_rgb8()))
    }
}

/// Fits a display crop to a texture slot by covering it and trimming the
/// overhang, centred.
///
/// Letterboxing would leave bars inside a Cell, and stretching would change a
/// face's proportions. Trimming loses the edges of a crop that disagrees with
/// the house framing, which is the least damaging of the three.
fn fit_to_slot(crop: &RgbImage) -> RgbaImage {
    let (width, height) = crop.dimensions();
    if width == 0 || height == 0 {
        return RgbaImage::from_pixel(SLOT_WIDTH, SLOT_HEIGHT, image::Rgba([0, 0, 0, 0]));
    }

    let scale = (SLOT_WIDTH as f32 / width as f32).max(SLOT_HEIGHT as f32 / height as f32);
    let covered = image::imageops::resize(
        crop,
        (width as f32 * scale).ceil().max(SLOT_WIDTH as f32) as u32,
        (height as f32 * scale).ceil().max(SLOT_HEIGHT as f32) as u32,
        FilterType::CatmullRom,
    );

    let (covered_width, covered_height) = covered.dimensions();
    let cropped = image::imageops::crop_imm(
        &covered,
        (covered_width - SLOT_WIDTH) / 2,
        (covered_height - SLOT_HEIGHT) / 2,
        SLOT_WIDTH,
        SLOT_HEIGHT,
    )
    .to_image();

    RgbaImage::from_fn(SLOT_WIDTH, SLOT_HEIGHT, |x, y| {
        let pixel = cropped.get_pixel(x, y).0;
        image::Rgba([pixel[0], pixel[1], pixel[2], 255])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: u32, height: u32, colour: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(width, height, image::Rgb(colour))
    }

    #[test]
    fn should_produce_exactly_one_slot_when_a_crop_is_fitted() {
        let fitted = fit_to_slot(&solid(512, 640, [10, 20, 30]));

        assert_eq!(fitted.dimensions(), (SLOT_WIDTH, SLOT_HEIGHT));
        assert_eq!(fitted.get_pixel(0, 0).0, [10, 20, 30, 255]);
    }

    #[test]
    fn should_trim_the_overhang_when_a_crop_is_wider_than_a_slot() {
        // A wide crop with a distinct left edge: covering it to slot height
        // pushes that edge outside the slot, and it is trimmed rather than
        // squeezed into view.
        let mut crop = solid(1000, 500, [255, 255, 255]);
        for y in 0..500 {
            for x in 0..100 {
                crop.put_pixel(x, y, image::Rgb([0, 0, 0]));
            }
        }

        let fitted = fit_to_slot(&crop);

        assert_eq!(fitted.dimensions(), (SLOT_WIDTH, SLOT_HEIGHT));
        assert_eq!(fitted.get_pixel(0, SLOT_HEIGHT / 2).0, [255, 255, 255, 255]);
    }

    #[test]
    fn should_enlarge_the_crop_when_it_is_smaller_than_a_slot() {
        let fitted = fit_to_slot(&solid(64, 80, [7, 7, 7]));

        assert_eq!(fitted.dimensions(), (SLOT_WIDTH, SLOT_HEIGHT));
        assert_eq!(
            fitted.get_pixel(SLOT_WIDTH / 2, SLOT_HEIGHT / 2).0,
            [7, 7, 7, 255]
        );
    }

    #[test]
    fn should_produce_an_empty_slot_when_the_crop_has_no_pixels() {
        let fitted = fit_to_slot(&RgbImage::new(0, 0));

        assert_eq!(fitted.dimensions(), (SLOT_WIDTH, SLOT_HEIGHT));
        assert_eq!(fitted.get_pixel(0, 0).0[3], 0);
    }
}
