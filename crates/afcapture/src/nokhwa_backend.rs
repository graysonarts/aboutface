//! The real camera. The only file in the workspace that imports a camera API.

use std::time::Duration;

use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::{Camera as NokhwaDevice, NokhwaError};

use crate::camera::{
    Camera, CameraDescription, CameraError, CameraSelector, WARMUP_BUDGET, settle,
};
use crate::frame::Frame;

/// How long to wait for the operating system's camera-permission prompt.
///
/// Long enough for a Visitor-free first run where an operator clicks the macOS
/// dialog; short enough that an unattended booth reports a problem instead of
/// hanging forever.
const PERMISSION_TIMEOUT: Duration = Duration::from_secs(30);

/// How many frames to pull and throw away before the first Capture.
///
/// AVFoundation returns an open stream before the sensor has anything on it:
/// measured on a FaceTime HD camera, the first two frames were byte-identical
/// pure black and the scene only appeared on the third. A Visitor presses the
/// Shutter once, so that first frame is the one that matters, and a black
/// photograph of someone who agreed to be photographed is worse than a failure.
///
/// Measured on that camera in a dark room, which is the slow case because
/// auto-exposure converges on a clock rather than a frame counter: 15 frames
/// still under-exposed, 45 reached the same result as 90, and 90 cost an extra
/// 1.3s for nothing. 45 frames is roughly 1.3s at 33fps, paid once when the
/// booth opens the camera, not per Capture.
///
/// A brighter room settles sooner, so this is an upper bound rather than a
/// requirement. Override it with [`NokhwaCamera::with_warmup_frames`].
const WARMUP_FRAMES: u32 = 45;

/// A camera driven by [`nokhwa`].
///
/// `nokhwa` is the initial cross-platform implementation and is deliberately
/// replaceable: a platform-specific backend can be swapped in behind
/// [`Camera`] if it proves inadequate (ADR-0006). Nothing about it escapes this
/// module — the trait, [`Frame`] and [`CameraError`] are what the rest of the
/// workspace sees.
pub struct NokhwaCamera {
    selector: CameraSelector,
    device: Option<NokhwaDevice>,
    name: String,
    warmup_frames: u32,
}

// `nokhwa::Camera` is not `Debug`, and it is not this crate's job to make the
// backend printable — only to keep it invisible.
impl std::fmt::Debug for NokhwaCamera {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NokhwaCamera")
            .field("selector", &self.selector)
            .field("name", &self.name)
            .field("open", &self.device.is_some())
            .finish()
    }
}

impl NokhwaCamera {
    /// Prepares a camera. Nothing touches the hardware until
    /// [`Camera::open`].
    pub fn new(selector: CameraSelector) -> Self {
        // Until the device list is consulted the only name we have is the
        // request itself.
        let name = selector.to_string();
        Self {
            selector,
            device: None,
            name,
            warmup_frames: WARMUP_FRAMES,
        }
    }

    /// Overrides how many frames [`Camera::open`] discards before returning.
    ///
    /// Worth raising on a sensor that settles slowly, and worth setting to zero
    /// only when something else guarantees the stream is already live.
    pub fn with_warmup_frames(mut self, frames: u32) -> Self {
        self.warmup_frames = frames;
        self
    }

    /// Asks the operating system for camera access, blocking until it answers.
    ///
    /// A no-op everywhere except macOS, where AVFoundation requires explicit
    /// consent before any device will enumerate.
    fn ensure_permission(&self) -> Result<(), CameraError> {
        if nokhwa::nokhwa_check() {
            return Ok(());
        }

        let (sender, receiver) = std::sync::mpsc::channel();
        nokhwa::nokhwa_initialize(move |granted| {
            let _ = sender.send(granted);
        });

        match receiver.recv_timeout(PERMISSION_TIMEOUT) {
            Ok(true) => Ok(()),
            Ok(false) => Err(CameraError::PermissionDenied {
                device: self.name.clone(),
                message: "the operating system refused camera access".to_owned(),
            }),
            Err(_) => Err(CameraError::PermissionDenied {
                device: self.name.clone(),
                message: format!(
                    "no answer to the camera permission prompt within {}s",
                    PERMISSION_TIMEOUT.as_secs()
                ),
            }),
        }
    }

    /// Resolves the selector against the devices the platform can see.
    ///
    /// Enumerating first means an absent camera is reported as
    /// [`CameraError::NotFound`] on the evidence of the device list, rather
    /// than inferred from whatever the backend says when the open fails.
    fn resolve(&self) -> Result<(CameraIndex, String), CameraError> {
        let backend = nokhwa::native_api_backend().ok_or_else(|| CameraError::Backend {
            operation: "open",
            message: format!("no native camera backend for {}", std::env::consts::OS),
        })?;

        let devices = nokhwa::query(backend)
            .map_err(|error| classify(&error, "open", &self.selector, &self.name))?;

        let device = match &self.selector {
            CameraSelector::Default => devices.first(),
            CameraSelector::Index(wanted) => devices
                .iter()
                .find(|info| info.index().as_index().is_ok_and(|index| index == *wanted)),
        }
        .ok_or_else(|| CameraError::NotFound {
            requested: self.selector.clone(),
        })?;

        Ok((device.index().clone(), device.human_name()))
    }
}

impl Camera for NokhwaCamera {
    fn open(&mut self) -> Result<(), CameraError> {
        self.ensure_permission()?;

        let (index, name) = self.resolve()?;
        self.name = name;

        // Highest resolution the device offers: the original frame is retained
        // at full quality so re-embedding stays possible (ADR-0006), and
        // Capture is once-per-Visitor, so frame rate does not matter.
        let format =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestResolution);

        let mut device = NokhwaDevice::new(index, format)
            .map_err(|error| classify(&error, "open", &self.selector, &self.name))?;
        device
            .open_stream()
            .map_err(|error| classify(&error, "open", &self.selector, &self.name))?;

        // The stream is open but the sensor is not ready; see `WARMUP_FRAMES`.
        let discarded = settle(self.warmup_frames, WARMUP_BUDGET, || {
            device.frame().map(|_| ())
        });

        // A camera asked for frames that produced none is blind, not merely
        // slow. Saying so here stops it passing the startup self-check and
        // then failing — mislabelled as a disconnection — under a Visitor.
        if self.warmup_frames > 0 && discarded == 0 {
            return Err(CameraError::NoFrames {
                device: self.name.clone(),
                message: format!("no frame in {} warm-up attempts", self.warmup_frames),
            });
        }

        self.device = Some(device);
        Ok(())
    }

    fn grab(&mut self) -> Result<Frame, CameraError> {
        let name = self.name.clone();
        let device = self.device.as_mut().ok_or(CameraError::NotOpen)?;

        // A read failure on an already-open stream means the device went away
        // mid-run — the case an operator most needs named.
        let buffer = device.frame().map_err(|error| match error {
            NokhwaError::ReadFrameError(message) => CameraError::Disconnected {
                device: name.clone(),
                message,
            },
            other => CameraError::Backend {
                operation: "grab",
                message: other.to_string(),
            },
        })?;

        let image = buffer
            .decode_image::<RgbFormat>()
            .map_err(|error| CameraError::Backend {
                operation: "grab",
                message: error.to_string(),
            })?;

        Ok(Frame::from_rgb8(
            image.width(),
            image.height(),
            image.into_raw(),
        )?)
    }

    fn close(&mut self) -> Result<(), CameraError> {
        let Some(mut device) = self.device.take() else {
            return Ok(());
        };

        device.stop_stream().map_err(|error| CameraError::Backend {
            operation: "close",
            message: error.to_string(),
        })
    }

    fn describe(&self) -> CameraDescription {
        CameraDescription {
            name: self.name.clone(),
            backend: match nokhwa::native_api_backend() {
                Some(backend) => format!("nokhwa/{backend}"),
                None => "nokhwa/unsupported".to_owned(),
            },
        }
    }
}

/// Sorts a [`NokhwaError`] into the failures an operator can act on.
///
/// `nokhwa` flattens every platform's failure into a message string — there is
/// no error code to switch on (see `NokhwaError` in `nokhwa-core`) — so busy
/// and permission-denied are recognised by what the platform wrote. The
/// classification is a best effort and falls back to
/// [`CameraError::Backend`], which still reports the backend's own words
/// rather than panicking.
fn classify(
    error: &NokhwaError,
    operation: &'static str,
    selector: &CameraSelector,
    device: &str,
) -> CameraError {
    let message = error.to_string();
    let haystack = message.to_lowercase();

    let contains_any = |needles: &[&str]| needles.iter().any(|needle| haystack.contains(needle));

    if contains_any(&["busy", "in use", "already open", "already in use"]) {
        return CameraError::Busy {
            device: device.to_owned(),
            message,
        };
    }

    if contains_any(&[
        "permission",
        "denied",
        "not authorized",
        "unauthorized",
        "authorization",
    ]) {
        return CameraError::PermissionDenied {
            device: device.to_owned(),
            message,
        };
    }

    if contains_any(&[
        "no such",
        "not found",
        "does not exist",
        "nonexistent",
        "no device",
        "no camera",
    ]) {
        return CameraError::NotFound {
            requested: selector.clone(),
        };
    }

    CameraError::Backend { operation, message }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classified(error: NokhwaError) -> CameraError {
        classify(&error, "open", &CameraSelector::Default, "FaceTime HD")
    }

    #[test]
    fn should_report_busy_when_the_platform_says_the_device_is_in_use() {
        let error = classified(NokhwaError::OpenDeviceError(
            "/dev/video0".to_owned(),
            "Device or resource busy".to_owned(),
        ));

        assert!(matches!(error, CameraError::Busy { .. }), "{error}");
    }

    #[test]
    fn should_report_permission_denied_when_the_platform_says_access_was_refused() {
        let error = classified(NokhwaError::OpenDeviceError(
            "/dev/video0".to_owned(),
            "Permission denied".to_owned(),
        ));

        assert!(
            matches!(error, CameraError::PermissionDenied { .. }),
            "{error}"
        );
    }

    #[test]
    fn should_report_not_found_when_the_platform_says_there_is_no_such_device() {
        let error = classified(NokhwaError::OpenDeviceError(
            "/dev/video9".to_owned(),
            "No such file or directory".to_owned(),
        ));

        assert!(matches!(error, CameraError::NotFound { .. }), "{error}");
    }

    #[test]
    fn should_fall_back_to_the_backend_error_when_the_message_is_unrecognised() {
        let error = classified(NokhwaError::GeneralError("something new".to_owned()));

        assert!(matches!(error, CameraError::Backend { .. }), "{error}");
    }

    #[test]
    fn should_warm_up_by_default_so_the_first_capture_is_not_black() {
        let camera = NokhwaCamera::new(CameraSelector::Default);

        assert!(
            camera.warmup_frames > 0,
            "a default with no warm-up reintroduces the black first frame"
        );
    }

    #[test]
    fn should_refuse_to_grab_when_never_opened() {
        let mut camera = NokhwaCamera::new(CameraSelector::Index(0));

        assert!(matches!(camera.grab(), Err(CameraError::NotOpen)));
    }

    #[test]
    fn should_close_without_error_when_never_opened() {
        let mut camera = NokhwaCamera::new(CameraSelector::Default);

        assert!(camera.close().is_ok());
    }
}
