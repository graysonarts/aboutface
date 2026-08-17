//! The one thing a [`Camera`](crate::Camera) hands back.

/// Ways a [`Frame`] can fail to describe an image.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FrameError {
    /// A frame with a zero dimension has no pixels to capture.
    #[error("frame has a zero dimension: {width}x{height}")]
    ZeroDimension {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
    },

    /// The buffer does not hold exactly `width * height * 3` bytes.
    #[error("frame buffer is {actual} bytes, expected {expected} for {width}x{height} RGB8")]
    SizeMismatch {
        /// Frame width.
        width: u32,
        /// Frame height.
        height: u32,
        /// Bytes the dimensions require.
        expected: usize,
        /// Bytes actually supplied.
        actual: usize,
    },
}

/// One frame off a camera, as 8-bit RGB.
///
/// This is deliberately a plain buffer rather than a backend type: nothing
/// outside this crate may see a camera API (ADR-0006), and the frame is what
/// crosses that boundary.
///
/// The frame is the *original*, kept at full quality — re-embedding the Corpus
/// after a model change depends on it (ADR-0006), so nothing here crops,
/// resizes or re-encodes lossily.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl Frame {
    /// Bytes per pixel in the RGB8 layout a [`Frame`] holds.
    pub const CHANNELS: usize = 3;

    /// Creates a frame from a tightly packed RGB8 buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if either dimension is zero, or if `pixels` is not
    /// exactly `width * height * 3` bytes.
    pub fn from_rgb8(width: u32, height: u32, pixels: Vec<u8>) -> Result<Self, FrameError> {
        if width == 0 || height == 0 {
            return Err(FrameError::ZeroDimension { width, height });
        }

        let expected = width as usize * height as usize * Self::CHANNELS;
        if pixels.len() != expected {
            return Err(FrameError::SizeMismatch {
                width,
                height,
                expected,
                actual: pixels.len(),
            });
        }

        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Frame width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The packed RGB8 pixels.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Consumes the frame, yielding its packed RGB8 pixels.
    pub fn into_pixels(self) -> Vec<u8> {
        self.pixels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_expose_dimensions_when_buffer_matches() {
        let frame = Frame::from_rgb8(2, 3, vec![7; 2 * 3 * 3]).expect("valid frame");

        assert_eq!((frame.width(), frame.height()), (2, 3));
    }

    #[test]
    fn should_reject_frame_when_a_dimension_is_zero() {
        assert_eq!(
            Frame::from_rgb8(0, 3, vec![]),
            Err(FrameError::ZeroDimension {
                width: 0,
                height: 3
            })
        );
    }

    #[test]
    fn should_reject_frame_when_buffer_length_disagrees_with_dimensions() {
        assert_eq!(
            Frame::from_rgb8(2, 2, vec![0; 10]),
            Err(FrameError::SizeMismatch {
                width: 2,
                height: 2,
                expected: 12,
                actual: 10,
            })
        );
    }
}
