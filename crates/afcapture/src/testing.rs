//! A [`Camera`] that needs no camera.
//!
//! Every later stage has to be exercisable on a machine with no webcam — CI, a
//! laptop with the lid shut, a developer who is not in front of the booth — so
//! the fake is part of the deliverable rather than a test fixture hidden inside
//! this crate's own test module.
//!
//! It is compiled unconditionally instead of behind a `testing` feature.
//! Cargo unifies features across a workspace, so a feature enabled by one
//! crate's dev-dependencies is enabled for every build of `afcapture` in that
//! graph anyway; the gate would buy no isolation and would only add a way for
//! the workspace to fail to compile. The cost is a few hundred lines of image
//! decoding in the booth binary.

use std::path::{Path, PathBuf};

use crate::camera::{Camera, CameraDescription, CameraError};
use crate::frame::Frame;

/// Where a [`FakeCamera`]'s frames come from.
#[derive(Debug, Clone)]
enum Replay {
    /// Image files on disk, decoded on demand and cycled through.
    Files(Vec<PathBuf>),

    /// A flat mid-grey frame, for tests that only care about the plumbing.
    Synthetic { width: u32, height: u32 },
}

/// A [`Camera`] that replays frames from disk.
///
/// Construct it with [`FakeCamera::replaying`] over real photographs (the
/// repository's `samples/` directory holds four), or with
/// [`FakeCamera::still`] when the test only cares that *a* frame arrived.
///
/// Failure modes are scripted rather than simulated by luck:
/// [`FakeCamera::failing_to_open`] covers the absent and busy camera, and
/// [`FakeCamera::disconnecting_after`] covers the camera yanked mid-run.
#[derive(Debug)]
pub struct FakeCamera {
    replay: Replay,
    open_result: Option<CameraError>,
    disconnect_after: Option<usize>,
    open: bool,
    grabs: usize,
}

impl FakeCamera {
    /// Replays the given image files in order, cycling when they run out.
    ///
    /// A real camera never runs out of frames, so neither does this one.
    pub fn replaying<P: Into<PathBuf>>(paths: impl IntoIterator<Item = P>) -> Self {
        Self::with_replay(Replay::Files(paths.into_iter().map(Into::into).collect()))
    }

    /// Returns the same flat synthetic frame every time.
    pub fn still(width: u32, height: u32) -> Self {
        Self::with_replay(Replay::Synthetic { width, height })
    }

    /// A camera that refuses to open, with the error of your choosing —
    /// [`CameraError::NotFound`] for absent, [`CameraError::Busy`] for held by
    /// something else.
    pub fn failing_to_open(error: CameraError) -> Self {
        Self {
            open_result: Some(error),
            ..Self::still(1, 1)
        }
    }

    /// Reports [`CameraError::Disconnected`] once `grabs` frames have been
    /// handed out.
    #[must_use]
    pub fn disconnecting_after(mut self, grabs: usize) -> Self {
        self.disconnect_after = Some(grabs);
        self
    }

    /// How many frames this camera has handed out.
    pub fn grabs(&self) -> usize {
        self.grabs
    }

    /// Whether [`Camera::open`] has succeeded and [`Camera::close`] has not yet
    /// run.
    pub fn is_open(&self) -> bool {
        self.open
    }

    fn with_replay(replay: Replay) -> Self {
        Self {
            replay,
            open_result: None,
            disconnect_after: None,
            open: false,
            grabs: 0,
        }
    }

    fn next_frame(&self) -> Result<Frame, CameraError> {
        match &self.replay {
            Replay::Synthetic { width, height } => Frame::from_rgb8(
                *width,
                *height,
                vec![128; *width as usize * *height as usize * Frame::CHANNELS],
            )
            .map_err(CameraError::from),
            Replay::Files(paths) => {
                if paths.is_empty() {
                    return Err(CameraError::Backend {
                        operation: "grab",
                        message: "fake camera has no frames to replay".to_owned(),
                    });
                }
                decode_rgb8(&paths[self.grabs % paths.len()])
            }
        }
    }
}

fn decode_rgb8(path: &Path) -> Result<Frame, CameraError> {
    let image = image::open(path)
        .map_err(|source| CameraError::Backend {
            operation: "grab",
            message: format!("fake camera could not read {}: {source}", path.display()),
        })?
        .to_rgb8();

    Frame::from_rgb8(image.width(), image.height(), image.into_raw()).map_err(CameraError::from)
}

impl Camera for FakeCamera {
    fn open(&mut self) -> Result<(), CameraError> {
        if let Some(error) = self.open_result.take() {
            return Err(error);
        }
        self.open = true;
        Ok(())
    }

    fn grab(&mut self) -> Result<Frame, CameraError> {
        if !self.open {
            return Err(CameraError::NotOpen);
        }
        if let Some(limit) = self.disconnect_after
            && self.grabs >= limit
        {
            self.open = false;
            return Err(CameraError::Disconnected {
                device: "fake".to_owned(),
                message: format!("scripted disconnect after {limit} frames"),
            });
        }

        let frame = self.next_frame()?;
        self.grabs += 1;
        Ok(frame)
    }

    fn close(&mut self) -> Result<(), CameraError> {
        self.open = false;
        Ok(())
    }

    fn describe(&self) -> CameraDescription {
        CameraDescription {
            name: match &self.replay {
                Replay::Files(paths) => format!("replay of {} file(s)", paths.len()),
                Replay::Synthetic { width, height } => format!("synthetic {width}x{height}"),
            },
            backend: "fake".to_owned(),
        }
    }
}

/// A path to one of the repository's sample photographs, for tests that want a
/// real face rather than a flat frame.
///
/// Resolved from `CARGO_MANIFEST_DIR` so it works from any working directory.
///
/// # Panics
///
/// Panics if the crate is not two directories below the repository root, which
/// would mean the workspace layout changed.
pub fn sample_path(file_name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("afcapture sits two directories below the repository root")
        .join("samples")
        .join(file_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::CameraSelector;

    #[test]
    fn should_refuse_to_grab_when_not_opened() {
        let mut camera = FakeCamera::still(2, 2);

        assert!(matches!(camera.grab(), Err(CameraError::NotOpen)));
    }

    #[test]
    fn should_replay_a_photograph_from_disk_when_opened() {
        let mut camera = FakeCamera::replaying([sample_path("1.jpg")]);
        camera.open().expect("fake camera opens");

        let frame = camera.grab().expect("a decoded frame");

        assert_eq!(
            frame.pixels().len(),
            frame.width() as usize * frame.height() as usize * Frame::CHANNELS
        );
    }

    #[test]
    fn should_cycle_through_the_replay_list_when_it_runs_out() {
        let mut camera = FakeCamera::replaying([sample_path("1.jpg"), sample_path("2.jpg")]);
        camera.open().expect("fake camera opens");

        let first = camera.grab().expect("frame one");
        let second = camera.grab().expect("frame two");
        let third = camera.grab().expect("frame three");

        assert_ne!(first, second, "the samples are different photographs");
        assert_eq!(first, third, "replay wraps back to the first file");
    }

    #[test]
    fn should_report_the_scripted_error_when_opening_fails() {
        let mut camera = FakeCamera::failing_to_open(CameraError::NotFound {
            requested: CameraSelector::Index(9),
        });

        assert!(matches!(camera.open(), Err(CameraError::NotFound { .. })));
    }

    #[test]
    fn should_report_disconnection_when_the_scripted_frame_budget_is_spent() {
        let mut camera = FakeCamera::still(2, 2).disconnecting_after(1);
        camera.open().expect("fake camera opens");
        camera.grab().expect("the one budgeted frame");

        assert!(matches!(
            camera.grab(),
            Err(CameraError::Disconnected { .. })
        ));
    }

    #[test]
    fn should_report_closed_when_closed() {
        let mut camera = FakeCamera::still(2, 2);
        camera.open().expect("fake camera opens");
        camera.close().expect("close never fails");

        assert!(!camera.is_open());
    }
}
