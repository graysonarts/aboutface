//! The seam that keeps the hardware decision open.

use std::fmt;

use crate::frame::{Frame, FrameError};

/// Which camera to open.
///
/// Deliberately narrow: the booth has one camera pointed at one Visitor.
/// Selecting by human-readable name would need a device-enumeration story that
/// does not exist yet, so it is left out rather than half-built.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum CameraSelector {
    /// Whatever the platform considers the first camera.
    #[default]
    Default,

    /// A specific device index as the platform enumerates them.
    Index(u32),
}

impl fmt::Display for CameraSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => f.write_str("default camera"),
            Self::Index(index) => write!(f, "camera index {index}"),
        }
    }
}

/// What resolved when a camera was opened.
///
/// A misconfigured install should be obvious rather than mysteriously wrong, so
/// the booth's startup self-check reports this (ADR-0006).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CameraDescription {
    /// Human-readable device name, or a stand-in when the backend has none.
    pub name: String,

    /// Which backend produced this camera — `"nokhwa/avfoundation"`, `"fake"`, …
    pub backend: String,
}

impl fmt::Display for CameraDescription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name, self.backend)
    }
}

/// Ways camera access fails.
///
/// The three field failures the booth actually suffers — no camera plugged in,
/// a camera held by something else, and a camera yanked mid-run — are separate
/// variants on purpose. An operator standing in a gallery needs to be told
/// which one happened, and none of them is a panic.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CameraError {
    /// No such device. Nothing is plugged in, or the index does not exist.
    #[error("no camera matching {requested}")]
    NotFound {
        /// What was asked for.
        requested: CameraSelector,
    },

    /// The device exists but is held by another process.
    #[error("camera {device} is busy: {message}")]
    Busy {
        /// Device the backend named.
        device: String,
        /// The backend's own words.
        message: String,
    },

    /// The device went away after it had been opened.
    #[error("camera {device} disconnected: {message}")]
    Disconnected {
        /// Device the backend named.
        device: String,
        /// The backend's own words.
        message: String,
    },

    /// The operating system refused access — on macOS, camera TCC consent.
    #[error("camera access denied for {device}: {message}")]
    PermissionDenied {
        /// Device the backend named.
        device: String,
        /// The backend's own words.
        message: String,
    },

    /// A frame was requested from a camera that is not open.
    #[error("camera is not open")]
    NotOpen,

    /// The frame arrived but could not be turned into a [`Frame`].
    #[error("camera frame was malformed")]
    MalformedFrame(#[from] FrameError),

    /// Anything the backend reported that does not map to the cases above.
    #[error("camera backend failed during {operation}: {message}")]
    Backend {
        /// The operation under way — `"open"`, `"grab"`, `"close"`.
        operation: &'static str,
        /// The backend's own words.
        message: String,
    },
}

/// A camera, with the platform API contained behind it.
///
/// `afcapture` is the only crate in the workspace allowed to import a camera
/// API; this trait is how the rest of the system asks for a frame (ADR-0006).
///
/// Implementations must be usable as `dyn Camera`, so the booth can choose a
/// backend at runtime.
pub trait Camera {
    /// Opens the device and starts its stream.
    ///
    /// # Errors
    ///
    /// Returns [`CameraError::NotFound`], [`CameraError::Busy`] or
    /// [`CameraError::PermissionDenied`] where the backend allows them to be
    /// told apart, and [`CameraError::Backend`] otherwise.
    fn open(&mut self) -> Result<(), CameraError>;

    /// Grabs exactly one frame.
    ///
    /// # Errors
    ///
    /// Returns [`CameraError::NotOpen`] if [`Camera::open`] has not succeeded,
    /// and [`CameraError::Disconnected`] if the device vanished mid-run.
    fn grab(&mut self) -> Result<Frame, CameraError>;

    /// Stops the stream and releases the device. Closing a closed camera is not
    /// an error.
    ///
    /// # Errors
    ///
    /// Returns [`CameraError::Backend`] if the backend refuses to shut down.
    fn close(&mut self) -> Result<(), CameraError>;

    /// What this camera is, for the startup self-check (ADR-0006).
    fn describe(&self) -> CameraDescription;
}

/// So a boxed camera is still a camera, and the booth can pick its backend at
/// runtime without every consumer becoming generic over the choice.
impl<C: Camera + ?Sized> Camera for Box<C> {
    fn open(&mut self) -> Result<(), CameraError> {
        (**self).open()
    }

    fn grab(&mut self) -> Result<Frame, CameraError> {
        (**self).grab()
    }

    fn close(&mut self) -> Result<(), CameraError> {
        (**self).close()
    }

    fn describe(&self) -> CameraDescription {
        (**self).describe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::FakeCamera;

    #[test]
    fn should_be_object_safe_so_the_backend_can_be_chosen_at_runtime() {
        let mut camera: Box<dyn Camera> = Box::new(FakeCamera::still(2, 2));
        camera.open().expect("fake camera opens");

        assert_eq!(camera.grab().expect("a frame").width(), 2);
    }

    #[test]
    fn should_render_selector_readably_when_reported_as_not_found() {
        let error = CameraError::NotFound {
            requested: CameraSelector::Index(3),
        };

        assert_eq!(error.to_string(), "no camera matching camera index 3");
    }
}
