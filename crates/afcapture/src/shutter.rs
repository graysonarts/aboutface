//! The Shutter: one press, one Capture.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::camera::{Camera, CameraError};

/// Ways a Shutter press fails.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ShutterError {
    /// The camera could not be opened, or the frame never arrived.
    #[error("camera failed")]
    Camera(#[from] CameraError),

    /// The directory captures are written to could not be created.
    #[error("could not create capture directory {path}")]
    Directory {
        /// The directory in question.
        path: PathBuf,
        /// The underlying filesystem error.
        #[source]
        source: std::io::Error,
    },

    /// The frame arrived but could not be written.
    #[error("could not write capture to {path}")]
    Write {
        /// Where the capture was going.
        path: PathBuf,
        /// The underlying encoding or filesystem error.
        #[source]
        source: image::ImageError,
    },
}

/// One Capture that landed on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capture {
    /// Where the original frame was written.
    pub path: PathBuf,

    /// Frame width in pixels.
    pub width: u32,

    /// Frame height in pixels.
    pub height: u32,

    /// When the Shutter was pressed.
    pub captured_at: SystemTime,
}

/// The control a Visitor presses to consent to and trigger a Capture.
///
/// The Shutter *is* the consent gesture — there is no separate consent step
/// (`CONTEXT.md`, ADR-0005) — so this type never grabs a frame on its own
/// initiative. Every frame that reaches disk is the direct result of a
/// [`Shutter::press`] call, and one press yields exactly one frame.
///
/// The Consent Record and Receipt Code that must accompany a Capture are Stage
/// 4 work; this is the code path they will hang off.
#[derive(Debug)]
pub struct Shutter<C: Camera> {
    camera: C,
    directory: PathBuf,
    sequence: u64,
}

impl<C: Camera> Shutter<C> {
    /// Builds a Shutter over `camera`, writing Captures into `directory`.
    ///
    /// The directory is created on the first press, not here.
    pub fn new(camera: C, directory: impl Into<PathBuf>) -> Self {
        Self {
            camera,
            directory: directory.into(),
            sequence: 0,
        }
    }

    /// Opens the camera, so the first press is not also the first exposure.
    ///
    /// # Errors
    ///
    /// Propagates whatever [`Camera::open`] reported — absent, busy, or
    /// permission-denied are distinct there.
    pub fn open(&mut self) -> Result<(), ShutterError> {
        self.camera.open()?;
        Ok(())
    }

    /// Grabs exactly one frame and writes it to disk.
    ///
    /// # Errors
    ///
    /// Returns [`ShutterError::Camera`] if the frame never arrived — including
    /// [`CameraError::Disconnected`] for a camera unplugged mid-run — and
    /// [`ShutterError::Directory`] or [`ShutterError::Write`] if it arrived but
    /// could not be stored.
    pub fn press(&mut self) -> Result<Capture, ShutterError> {
        let frame = self.camera.grab()?;
        let captured_at = SystemTime::now();

        std::fs::create_dir_all(&self.directory).map_err(|source| ShutterError::Directory {
            path: self.directory.clone(),
            source,
        })?;

        let path = self
            .directory
            .join(capture_file_name(captured_at, self.sequence));
        self.sequence += 1;

        let width = frame.width();
        let height = frame.height();
        image::save_buffer(
            &path,
            frame.pixels(),
            width,
            height,
            image::ExtendedColorType::Rgb8,
        )
        .map_err(|source| ShutterError::Write {
            path: path.clone(),
            source,
        })?;

        Ok(Capture {
            path,
            width,
            height,
            captured_at,
        })
    }

    /// Releases the camera.
    ///
    /// # Errors
    ///
    /// Propagates whatever [`Camera::close`] reported.
    pub fn close(&mut self) -> Result<(), ShutterError> {
        self.camera.close()?;
        Ok(())
    }

    /// The camera behind this Shutter, for the startup self-check.
    pub fn camera(&self) -> &C {
        &self.camera
    }

    /// Where Captures are written.
    pub fn directory(&self) -> &Path {
        &self.directory
    }
}

/// Names a Capture file.
///
/// Wall-clock milliseconds order the archive readably; the sequence number
/// keeps two presses inside the same millisecond from colliding. PNG, because
/// the original frame is retained at full quality for re-embedding (ADR-0006).
fn capture_file_name(captured_at: SystemTime, sequence: u64) -> String {
    let millis = captured_at
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_millis());

    format!("capture-{millis:013}-{sequence:04}.png")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::CameraSelector;
    use crate::testing::{FakeCamera, sample_path};

    fn shutter_over(camera: FakeCamera) -> (Shutter<FakeCamera>, tempfile::TempDir) {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let shutter = Shutter::new(camera, directory.path().join("captures"));
        (shutter, directory)
    }

    #[test]
    fn should_grab_exactly_one_frame_when_pressed_once() {
        let (mut shutter, _directory) = shutter_over(FakeCamera::still(4, 4));
        shutter.open().expect("camera opens");

        shutter.press().expect("a capture");

        assert_eq!(shutter.camera().grabs(), 1);
    }

    #[test]
    fn should_write_the_frame_to_disk_when_pressed() {
        let (mut shutter, _directory) = shutter_over(FakeCamera::replaying([sample_path("1.jpg")]));
        shutter.open().expect("camera opens");

        let capture = shutter.press().expect("a capture");

        let written = image::open(&capture.path)
            .expect("a readable image")
            .to_rgb8();
        assert_eq!(
            (written.width(), written.height()),
            (capture.width, capture.height)
        );
    }

    #[test]
    fn should_write_one_file_per_press_when_pressed_repeatedly() {
        let (mut shutter, _directory) = shutter_over(FakeCamera::still(4, 4));
        shutter.open().expect("camera opens");

        let first = shutter.press().expect("first capture");
        let second = shutter.press().expect("second capture");

        assert_ne!(first.path, second.path);
        assert_eq!(
            std::fs::read_dir(shutter.directory())
                .expect("capture directory exists")
                .count(),
            2
        );
    }

    #[test]
    fn should_create_the_capture_directory_when_it_does_not_exist() {
        let (mut shutter, _directory) = shutter_over(FakeCamera::still(4, 4));
        shutter.open().expect("camera opens");
        assert!(!shutter.directory().exists());

        shutter.press().expect("a capture");

        assert!(shutter.directory().is_dir());
    }

    #[test]
    fn should_report_camera_absence_when_opening_without_a_camera() {
        let (mut shutter, _directory) =
            shutter_over(FakeCamera::failing_to_open(CameraError::NotFound {
                requested: CameraSelector::Default,
            }));

        assert!(matches!(
            shutter.open(),
            Err(ShutterError::Camera(CameraError::NotFound { .. }))
        ));
    }

    #[test]
    fn should_report_a_busy_camera_distinctly_from_an_absent_one() {
        let (mut shutter, _directory) =
            shutter_over(FakeCamera::failing_to_open(CameraError::Busy {
                device: "FaceTime HD".to_owned(),
                message: "Device or resource busy".to_owned(),
            }));

        assert!(matches!(
            shutter.open(),
            Err(ShutterError::Camera(CameraError::Busy { .. }))
        ));
    }

    #[test]
    fn should_report_disconnection_when_the_camera_vanishes_mid_run() {
        let (mut shutter, _directory) =
            shutter_over(FakeCamera::still(4, 4).disconnecting_after(1));
        shutter.open().expect("camera opens");
        shutter.press().expect("the one capture that works");

        assert!(matches!(
            shutter.press(),
            Err(ShutterError::Camera(CameraError::Disconnected { .. }))
        ));
    }

    #[test]
    fn should_not_write_anything_when_the_frame_never_arrives() {
        let (mut shutter, _directory) = shutter_over(FakeCamera::still(4, 4));

        let _ = shutter.press().expect_err("camera was never opened");

        assert!(!shutter.directory().exists());
    }

    #[test]
    fn should_order_capture_names_by_press_when_the_clock_does_not_tick() {
        let now = UNIX_EPOCH;

        assert!(capture_file_name(now, 0) < capture_file_name(now, 1));
    }
}
