//! The Shutter: one press, one Capture.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::camera::{Camera, CameraError};
use crate::frame::Frame;

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

    /// A Shutter built to expose only was asked to write.
    #[error("this shutter has no capture directory — it exposes, it does not press")]
    NoDirectory,

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
    /// Where [`Shutter::press`] writes. `None` for a Shutter built with
    /// [`Shutter::over`], which only exposes.
    directory: Option<PathBuf>,
    sequence: u64,
}

impl<C: Camera> Shutter<C> {
    /// Builds a Shutter over `camera`, writing Captures into `directory`.
    ///
    /// The directory is created on the first press, not here.
    pub fn new(camera: C, directory: impl Into<PathBuf>) -> Self {
        Self {
            camera,
            directory: Some(directory.into()),
            sequence: 0,
        }
    }

    /// Builds a Shutter that only ever [`Shutter::expose`]s.
    ///
    /// The booth stores the original frame in the Corpus itself (ADR-0006), so
    /// it has nowhere to write loose Captures and should not be made to name a
    /// directory it will never use. Pressing one of these is
    /// [`ShutterError::NoDirectory`].
    pub fn over(camera: C) -> Self {
        Self {
            camera,
            directory: None,
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

    /// Grabs exactly one frame and hands it back, writing nothing.
    ///
    /// This is the press the booth makes: the Corpus retains the original
    /// frame itself (ADR-0006), so a second copy in a loose directory would be
    /// an archive nobody deletes on a Receipt Code. [`Shutter::press`] is the
    /// same gesture for a caller that has no Corpus to put it in.
    ///
    /// # Errors
    ///
    /// Propagates whatever [`Camera::grab`] reported, including
    /// [`CameraError::Disconnected`] for a camera unplugged mid-run.
    pub fn expose(&mut self) -> Result<Frame, ShutterError> {
        Ok(self.camera.grab()?)
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
        let directory = self.directory.clone().ok_or(ShutterError::NoDirectory)?;
        let frame = self.expose()?;
        let captured_at = SystemTime::now();

        std::fs::create_dir_all(&directory).map_err(|source| ShutterError::Directory {
            path: directory.clone(),
            source,
        })?;

        let path = directory.join(capture_file_name(captured_at, self.sequence));
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

    /// Where Captures are written, if this Shutter writes them at all.
    pub fn directory(&self) -> Option<&Path> {
        self.directory.as_deref()
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

    /// Where a writing Shutter puts its Captures.
    fn writes_to(shutter: &Shutter<FakeCamera>) -> &Path {
        shutter
            .directory()
            .expect("a shutter built with a directory")
    }

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
    fn should_refuse_to_press_when_the_shutter_only_exposes() {
        let mut shutter = Shutter::over(FakeCamera::still(4, 4));
        shutter.open().expect("camera opens");

        assert!(matches!(shutter.press(), Err(ShutterError::NoDirectory)));
    }

    #[test]
    fn should_hand_back_one_frame_without_writing_it_when_exposed() {
        let (mut shutter, _directory) = shutter_over(FakeCamera::replaying([sample_path("1.jpg")]));
        shutter.open().expect("camera opens");

        let frame = shutter.expose().expect("a frame");

        assert_eq!(shutter.camera().grabs(), 1);
        assert!(frame.width() > 0);
        assert!(
            !writes_to(&shutter).exists(),
            "an exposure the booth stores itself must leave no second copy"
        );
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
            std::fs::read_dir(writes_to(&shutter))
                .expect("capture directory exists")
                .count(),
            2
        );
    }

    #[test]
    fn should_create_the_capture_directory_when_it_does_not_exist() {
        let (mut shutter, _directory) = shutter_over(FakeCamera::still(4, 4));
        shutter.open().expect("camera opens");
        assert!(!writes_to(&shutter).exists());

        shutter.press().expect("a capture");

        assert!(writes_to(&shutter).is_dir());
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

        assert!(!writes_to(&shutter).exists());
    }

    #[test]
    fn should_order_capture_names_by_press_when_the_clock_does_not_tick() {
        let now = UNIX_EPOCH;

        assert!(capture_file_name(now, 0) < capture_file_name(now, 1));
    }
}
