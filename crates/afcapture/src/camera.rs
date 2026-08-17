//! The seam that keeps the hardware decision open.

use std::fmt;
use std::time::{Duration, Instant};

use crate::frame::{Frame, FrameError};

/// The longest a warm-up may take, whatever frame rate the sensor turns out to
/// run at.
///
/// Nothing pins frame rate — a backend asks for a resolution, not a cadence —
/// so a frame count means one duration on the camera it was measured on and
/// another on the hardware ADR-0006 has not chosen yet. This bounds the wait so
/// an unknown sensor delays the booth's startup rather than hanging it.
pub(crate) const WARMUP_BUDGET: Duration = Duration::from_secs(3);

/// Pulls and discards frames so a sensor can settle, per [`Camera::open`]'s
/// contract. Returns how many were discarded.
///
/// Stops at whichever comes first: `rounds` frames, `budget` elapsed, or a
/// failing grab.
///
/// Both limits are needed. Auto-exposure converges on a clock, but the only
/// thing a backend can pull is frames, and nothing pins how fast they arrive —
/// `budget` keeps a slow sensor from stalling the booth, `rounds` keeps a fast
/// one from spinning through hundreds of frames to fill the time.
///
/// A failing grab ends the warm-up rather than retrying: a discarded frame is
/// not the place to diagnose a device. The caller decides what zero frames
/// means — for [`Camera::open`] it is [`CameraError::NoFrames`].
pub(crate) fn settle<E>(
    rounds: u32,
    budget: Duration,
    mut grab: impl FnMut() -> Result<(), E>,
) -> u32 {
    let deadline = Instant::now() + budget;
    let mut discarded = 0;

    while discarded < rounds {
        if grab().is_err() {
            break;
        }
        discarded += 1;

        if Instant::now() >= deadline {
            break;
        }
    }

    discarded
}

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

    /// The stream started but never yielded a frame.
    ///
    /// Distinct from [`CameraError::Disconnected`], which means a working
    /// device went away: this one never worked. Reporting it at
    /// [`Camera::open`] keeps a blind camera from passing the booth's startup
    /// self-check and failing in front of a Visitor instead (ADR-0006).
    #[error("camera {device} opened but produced no frames: {message}")]
    NoFrames {
        /// Device the backend named.
        device: String,
        /// What was being attempted.
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
    /// # Contract
    ///
    /// On success the very next [`Camera::grab`] must return a usable frame.
    /// Sensors do not honour this by themselves — AVFoundation hands back an
    /// open stream while the first frames are still pure black — so an
    /// implementation whose device needs to settle must do that settling here;
    /// this crate's `settle` helper is what the in-tree backends use. A Visitor
    /// presses the Shutter once; that first frame is the whole Capture.
    ///
    /// This lives on the trait rather than in one backend because ADR-0006
    /// treats the backend as replaceable: a future implementation must inherit
    /// the obligation, not rediscover the black frame.
    ///
    /// # Errors
    ///
    /// Returns [`CameraError::NotFound`], [`CameraError::Busy`] or
    /// [`CameraError::PermissionDenied`] where the backend allows them to be
    /// told apart, [`CameraError::NoFrames`] if the stream started but yielded
    /// nothing, and [`CameraError::Backend`] otherwise.
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
    use std::path::PathBuf;

    use super::*;
    use crate::testing::FakeCamera;

    #[test]
    fn should_be_object_safe_so_the_backend_can_be_chosen_at_runtime() {
        let mut camera: Box<dyn Camera> = Box::new(FakeCamera::still(2, 2));
        camera.open().expect("fake camera opens");

        assert_eq!(camera.grab().expect("a frame").width(), 2);
    }

    fn is_black(frame: &Frame) -> bool {
        frame.pixels().iter().all(|channel| *channel == 0)
    }

    #[test]
    fn should_hand_out_a_black_frame_when_the_sensor_has_not_settled() {
        // Arrange: the defect itself — a sensor whose first frames carry no
        // signal, opened without a warm-up.
        let mut camera = FakeCamera::still(4, 4).with_black_frames(2);

        // Act
        camera.open().expect("fake camera opens");
        let frame = camera.grab().expect("a frame");

        // Assert
        assert!(
            is_black(&frame),
            "without a warm-up the Visitor's Capture is the unsettled frame"
        );
    }

    #[test]
    fn should_hand_out_a_settled_frame_when_open_warms_the_sensor_up() {
        // Arrange: the same sensor, opened by a backend that honours the
        // contract on `Camera::open`.
        let mut camera = FakeCamera::still(4, 4)
            .with_black_frames(2)
            .with_warmup_frames(2);

        // Act
        camera.open().expect("fake camera opens");
        let frame = camera.grab().expect("a frame");

        // Assert
        assert!(
            !is_black(&frame),
            "open must consume the black frames so the first Capture is real"
        );
    }

    #[test]
    fn should_not_count_warm_up_frames_as_frames_handed_out() {
        let mut camera = FakeCamera::still(4, 4).with_warmup_frames(3);

        camera.open().expect("fake camera opens");

        assert_eq!(camera.grabs(), 0, "frames nobody received are not Captures");
    }

    #[test]
    fn should_refuse_to_open_when_the_sensor_never_yields_a_frame() {
        // A camera that answers every grab with an error is blind, not slow.
        let mut camera = FakeCamera::replaying(Vec::<PathBuf>::new()).with_warmup_frames(4);

        let error = camera.open().expect_err("a blind camera must not open");

        assert!(matches!(error, CameraError::NoFrames { .. }), "{error}");
        assert!(
            !camera.is_open(),
            "a camera that failed to open must not report itself open"
        );
    }

    #[test]
    fn should_discard_the_requested_frames_when_every_grab_succeeds() {
        let mut grabbed = 0;

        let discarded = settle(5, Duration::from_secs(60), || {
            grabbed += 1;
            Ok::<(), ()>(())
        });

        assert_eq!(discarded, 5);
        assert_eq!(grabbed, 5);
    }

    #[test]
    fn should_stop_discarding_when_a_grab_fails() {
        let mut grabbed = 0;

        let discarded = settle(10, Duration::from_secs(60), || {
            grabbed += 1;
            if grabbed < 3 { Ok(()) } else { Err(()) }
        });

        assert_eq!(discarded, 2, "the failing grab is not a discarded frame");
        assert_eq!(grabbed, 3, "warm-up stops rather than retrying");
    }

    #[test]
    fn should_grab_nothing_when_no_warm_up_is_configured() {
        let mut grabbed = 0;

        let discarded = settle(0, Duration::from_secs(60), || {
            grabbed += 1;
            Ok::<(), ()>(())
        });

        assert_eq!(discarded, 0);
        assert_eq!(grabbed, 0);
    }

    #[test]
    fn should_stop_discarding_when_the_budget_runs_out_before_the_frames_do() {
        let mut grabbed = 0;

        // A sensor slow enough that the frame count would take far longer than
        // the booth can wait — the case the budget exists for.
        let discarded = settle(1_000, Duration::from_millis(30), || {
            grabbed += 1;
            std::thread::sleep(Duration::from_millis(10));
            Ok::<(), ()>(())
        });

        assert!(
            discarded < 1_000,
            "the budget must cut the warm-up short, discarded {discarded}"
        );
        assert!(
            discarded > 0,
            "the budget must not prevent warming up at all"
        );
    }

    #[test]
    fn should_render_selector_readably_when_reported_as_not_found() {
        let error = CameraError::NotFound {
            requested: CameraSelector::Index(3),
        };

        assert_eq!(error.to_string(), "no camera matching camera index 3");
    }
}
