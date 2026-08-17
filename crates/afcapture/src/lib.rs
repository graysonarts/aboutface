//! Camera access and the Shutter.
//!
//! Capture is deliberate and opt-in: the Visitor triggers the Shutter
//! themselves, and that gesture is simultaneously the consent and the exposure
//! (ADR-0005). This crate never captures on its own initiative.
//!
//! Hardware is deferred (ADR-0006), so camera access sits behind a trait with
//! `nokhwa` as the initial cross-platform implementation. **No other crate in
//! the workspace imports a camera API** — this is the one place a platform
//! choice is allowed to leak, and containing it here is the whole point of the
//! crate.
//!
//! # Shape
//!
//! - [`Camera`] — open, grab one frame, close, describe. The seam.
//! - [`Frame`] — a plain RGB8 buffer, the only thing that crosses the seam.
//! - [`Shutter`] — one press, one frame, one file on disk.
//! - `NokhwaCamera` — the real device, behind the default-on `nokhwa-backend`
//!   feature.
//! - [`testing::FakeCamera`] — replays photographs from disk, so every later
//!   stage is testable on a machine with no webcam.
//!
//! Swap `FakeCamera` for `NokhwaCamera::new(CameraSelector::Default)` and this
//! is the booth:
//!
//! ```no_run
//! use afcapture::{Shutter, testing::{FakeCamera, sample_path}};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let camera = FakeCamera::replaying([sample_path("1.jpg")]);
//! let mut shutter = Shutter::new(camera, "captures");
//!
//! shutter.open()?;
//! let capture = shutter.press()?; // the Visitor's one Capture
//! println!("wrote {}", capture.path.display());
//! shutter.close()?;
//! # Ok(())
//! # }
//! ```

mod camera;
mod frame;
mod shutter;

pub mod testing;

#[cfg(feature = "nokhwa-backend")]
mod nokhwa_backend;

pub use camera::{Camera, CameraDescription, CameraError, CameraSelector};
pub use frame::{Frame, FrameError};
pub use shutter::{Capture, Shutter, ShutterError};

#[cfg(feature = "nokhwa-backend")]
pub use nokhwa_backend::NokhwaCamera;
