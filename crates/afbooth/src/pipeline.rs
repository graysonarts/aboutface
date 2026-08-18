//! Shutter to Corpus: the one path a Capture takes.
//!
//! This is where the seven crates become one process. Nothing here reaches
//! into another crate's internals: the Shutter hands over a frame, `afvision`
//! turns it into an Embedding and two crops, `afstore` writes them, and
//! `afrender` is handed paths. The wiring lives in the binary because that is
//! the only place allowed to know all of them.
//!
//! Every stage is timed. ADR-0006 makes Stage 1 the hardware evaluation
//! instrument: the numbers a press prints are what one candidate machine is
//! compared against another with, so they are part of the deliverable rather
//! than debugging output.

use std::fmt;
use std::time::{Duration, Instant, SystemTime};

use afcapture::{Camera, CameraDescription, Shutter, ShutterError};
use afcore::{FaceId, GridSpec};
use afrender::Portrait;
use afstore::{Corpus, NewFace, StoreError};
use afvision::{
    DetectError, DisplayCropSpec, EmbedError, FaceDetector, FaceEmbedder, Faces, GeometryError,
    align, display_crop, select_execution_provider,
};
use image::RgbImage;

use crate::config::BoothConfig;

/// Ways the path from Shutter to wall fails.
///
/// A Capture with no face in it is not here: that is an outcome, not a failure
/// (see [`Captured`]).
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// The camera could not be opened, or the frame never arrived.
    #[error("capture failed: {0}")]
    Capture(#[from] ShutterError),

    /// A model file could not be loaded into a Session.
    #[error("cannot load the detector: {0}")]
    Detector(#[source] DetectError),

    /// The embedder could not be loaded.
    #[error("cannot load the embedder: {0}")]
    Embedder(#[source] EmbedError),

    /// Detection failed on a frame that did arrive.
    #[error("detection failed: {0}")]
    Detect(#[source] DetectError),

    /// The landmarks describe no warp — a degenerate detection.
    #[error("alignment failed: {0}")]
    Align(#[from] GeometryError),

    /// Inference failed on an aligned crop.
    #[error("embedding failed: {0}")]
    Embed(#[source] EmbedError),

    /// The Corpus refused the Face, or could not be read.
    #[error("the corpus refused the face: {0}")]
    Store(#[from] StoreError),

    /// A frame arrived with dimensions its own buffer contradicts.
    #[error("the camera returned a {width}x{height} frame with {bytes} bytes")]
    Frame {
        /// Frame width.
        width: u32,
        /// Frame height.
        height: u32,
        /// Bytes actually supplied.
        bytes: usize,
    },
}

/// What one Shutter press produced.
///
/// No face and several faces are reported rather than raised: a Visitor who
/// stepped out of frame, or two friends leaning in together, must leave the
/// Corpus unchanged and the piece running.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Captured {
    /// One face, now in the Corpus.
    Face {
        /// The Face the Corpus assigned.
        id: FaceId,
        /// What each stage cost.
        timings: Timings,
    },

    /// Nobody was in the frame.
    NoFace,

    /// More than one person was in the frame.
    ///
    /// Which face the booth would embed — the largest, the nearest, the
    /// highest-scoring — is an open question the implementation plan leaves to
    /// the wall, so a crowd is declined rather than resolved by accident here.
    Several(usize),
}

/// What one Capture cost, stage by stage.
///
/// Wall-clock per stage, because that is what an operator comparing two
/// machines can act on (ADR-0006).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timings {
    /// Grabbing the frame off the camera.
    pub capture: Duration,
    /// Finding the face and its landmarks.
    pub detect: Duration,
    /// Warping to the aligned crop and cutting the display crop.
    pub align: Duration,
    /// Inference.
    pub embed: Duration,
    /// Writing the row and the three images.
    pub store: Duration,
}

impl Timings {
    /// Everything the press cost.
    pub fn total(&self) -> Duration {
        self.capture + self.detect + self.align + self.embed + self.store
    }
}

impl fmt::Display for Timings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "capture {}  detect {}  align {}  embed {}  store {}  total {}",
            millis(self.capture),
            millis(self.detect),
            millis(self.align),
            millis(self.embed),
            millis(self.store),
            millis(self.total()),
        )
    }
}

fn millis(duration: Duration) -> String {
    format!("{:.1}ms", duration.as_secs_f64() * 1000.0)
}

/// The booth's one path: press, detect, align, embed, store.
///
/// Generic over nothing: the camera is a `Box<dyn Camera>` so the choice
/// between the device and the fake is made at runtime, which is what lets the
/// whole path be exercised in a test with no webcam (ADR-0009).
pub struct Pipeline {
    shutter: Shutter<Box<dyn Camera>>,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    corpus: Corpus,
    framing: DisplayCropSpec,
}

impl Pipeline {
    /// Loads both models, opens the Corpus, and opens the camera.
    ///
    /// The camera is opened here rather than on the first press, so a booth
    /// with no camera fails at startup in front of the operator instead of in
    /// front of a Visitor.
    ///
    /// # Errors
    ///
    /// Returns an error if a model will not load, the Corpus cannot be opened
    /// or migrated, or the camera is absent, busy or permission-denied.
    pub fn open(config: &BoothConfig, camera: Box<dyn Camera>) -> Result<Self, PipelineError> {
        let provider = select_execution_provider();
        let detector =
            FaceDetector::open(config.detector(), provider).map_err(PipelineError::Detector)?;
        let embedder =
            FaceEmbedder::open(config.embedder(), provider).map_err(PipelineError::Embedder)?;
        let corpus = Corpus::open(config.corpus_dir())?;

        // The booth exposes rather than presses: the Corpus retains the
        // original frame itself, so there is no loose capture directory.
        let mut shutter = Shutter::over(camera);
        shutter.open()?;

        Ok(Self {
            shutter,
            detector,
            embedder,
            corpus,
            framing: *config.display_crop(),
        })
    }

    /// Takes one Capture and, if there is exactly one face in it, stores it.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame never arrived, a model failed, or the
    /// Corpus refused the Face. An empty or crowded frame is an [`Captured`]
    /// outcome instead, and leaves the Corpus untouched.
    pub fn capture(&mut self) -> Result<Captured, PipelineError> {
        let started = Instant::now();
        let frame = self.shutter.expose()?;
        let captured_at = SystemTime::now();
        let original = to_image(frame)?;
        let capture = started.elapsed();

        let started = Instant::now();
        let faces = self
            .detector
            .detect(&original)
            .map_err(PipelineError::Detect)?;
        let detect = started.elapsed();

        let face = match faces {
            Faces::One(face) => face,
            Faces::None => return Ok(Captured::NoFace),
            Faces::Many(several) => return Ok(Captured::Several(several.len())),
        };

        let started = Instant::now();
        let aligned = align(&original, face.landmarks())?;
        let display = display_crop(&original, face.bbox(), &self.framing);
        let align_time = started.elapsed();

        let started = Instant::now();
        let embedding = self
            .embedder
            .embed(&aligned)
            .map_err(PipelineError::Embed)?;
        let embed = started.elapsed();

        let started = Instant::now();
        let id = self.corpus.ingest(NewFace::new(
            embedding,
            aligned,
            display,
            original,
            captured_at,
        ))?;
        let store = started.elapsed();

        Ok(Captured::Face {
            id,
            timings: Timings {
                capture,
                detect,
                align: align_time,
                embed,
                store,
            },
        })
    }

    /// The Window the wall should be showing: as many Faces as `grid` has
    /// Cells, ending at the newest.
    ///
    /// The Window sits at the end of the archive rather than its start,
    /// because the Visitor who just pressed the Shutter must find themselves on
    /// the wall — a Corpus larger than the Grid would otherwise show the oldest
    /// Faces forever and swallow every Capture after the first Grid's worth.
    /// Drift moves the Window across the whole archive in Stage 3 (ADR-0004);
    /// until then it is pinned to the end.
    ///
    /// # Errors
    ///
    /// Returns an error if the Corpus cannot be read.
    pub fn portraits(&self, grid: GridSpec) -> Result<Vec<Portrait>, PipelineError> {
        let cells = grid.cell_count();
        let offset = newest_window(self.corpus.count()?, cells);

        Ok(self
            .corpus
            .window(offset, cells)?
            .iter()
            .map(|face| Portrait::new(face.id(), face.display_path()))
            .collect())
    }

    /// How many Faces the Corpus holds.
    ///
    /// # Errors
    ///
    /// Returns an error if the Corpus cannot be read.
    pub fn face_count(&self) -> Result<u64, PipelineError> {
        Ok(self.corpus.count()?)
    }

    /// What camera the booth actually opened, for the startup self-check.
    pub fn camera(&self) -> CameraDescription {
        self.shutter.camera().describe()
    }

    /// Releases the camera.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend refuses to shut down.
    pub fn close(&mut self) -> Result<(), PipelineError> {
        self.shutter.close()?;
        Ok(())
    }
}

/// Where a Window of `cells` Cells sits when it is pinned to the newest Faces.
///
/// A Corpus smaller than the Grid starts at zero: every Face is on the wall and
/// the spare Cells stay empty.
fn newest_window(held: u64, cells: u32) -> u64 {
    held.saturating_sub(u64::from(cells))
}

/// The frame as an image the vision crate can read.
fn to_image(frame: afcapture::Frame) -> Result<RgbImage, PipelineError> {
    let (width, height) = (frame.width(), frame.height());
    let pixels = frame.into_pixels();
    let bytes = pixels.len();

    RgbImage::from_raw(width, height, pixels).ok_or(PipelineError::Frame {
        width,
        height,
        bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_sum_every_stage_when_a_capture_is_timed() {
        let timings = Timings {
            capture: Duration::from_millis(10),
            detect: Duration::from_millis(20),
            align: Duration::from_millis(1),
            embed: Duration::from_millis(40),
            store: Duration::from_millis(5),
        };

        assert_eq!(timings.total(), Duration::from_millis(76));
    }

    #[test]
    fn should_name_every_stage_when_timings_are_reported() {
        // The operator comparing two machines reads this line; a stage missing
        // from it is a stage nobody can compare.
        let line = Timings {
            capture: Duration::from_millis(1),
            detect: Duration::from_millis(2),
            align: Duration::from_millis(3),
            embed: Duration::from_millis(4),
            store: Duration::from_millis(5),
        }
        .to_string();

        for stage in ["capture", "detect", "align", "embed", "store", "total"] {
            assert!(line.contains(stage), "{stage} missing from {line}");
        }
    }

    #[test]
    fn should_start_the_window_at_zero_when_the_corpus_is_smaller_than_the_grid() {
        assert_eq!(newest_window(3, 20), 0);
        assert_eq!(newest_window(20, 20), 0);
    }

    #[test]
    fn should_end_the_window_on_the_newest_face_when_the_corpus_outgrows_the_grid() {
        // Twenty-one Faces, twenty Cells: the Visitor who just pressed the
        // Shutter is the last row, not a Face nobody can reach.
        assert_eq!(newest_window(21, 20), 1);
        assert_eq!(newest_window(1_000, 20), 980);
    }

    #[test]
    fn should_carry_every_pixel_across_when_a_frame_becomes_an_image() {
        let frame = afcapture::Frame::from_rgb8(2, 3, (0..18).collect()).expect("a valid frame");

        let image = to_image(frame).expect("the frame converts");

        assert_eq!(image.dimensions(), (2, 3));
        assert_eq!(image.get_pixel(0, 0).0, [0, 1, 2]);
        assert_eq!(image.get_pixel(1, 2).0, [15, 16, 17]);
    }
}
