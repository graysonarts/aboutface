//! The tracer bullet, end to end, with no camera and no window.
//!
//! An integration test because it crosses every system boundary the piece has:
//! ONNX Runtime, SQLite, the filesystem, and the `Camera` seam. What it cannot
//! cross is the GPU — a window needs a display — so it stops at the portraits
//! the wall would be handed.
//!
//! The ONNX files are fetched, not committed (`docs/models.md`), so these tests
//! report and return when a model is absent; CI builds without them, as in
//! `afvision`'s own integration tests. Run `./scripts/fetch-models.sh` to have
//! them do real work.

use std::path::{Path, PathBuf};

use afbooth::config::BoothConfig;
use afbooth::pipeline::{Captured, Pipeline, PipelineError};
use afcapture::testing::{FakeCamera, sample_path};
use afcapture::{Camera, CameraError, CameraSelector, ShutterError};
use tempfile::TempDir;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn models_dir() -> PathBuf {
    repo_root().join("models")
}

/// Whether both ONNX files have been fetched.
fn models_present() -> bool {
    let present = ["face_detection_yunet_2023mar.onnx", "dinov2-small.onnx"]
        .iter()
        .all(|file| models_dir().join(file).is_file());

    if !present {
        eprintln!(
            "skipping: {} is missing a model — run ./scripts/fetch-models.sh",
            models_dir().display()
        );
    }

    present
}

/// A temporary Corpus and a booth configured to store into it.
///
/// The directory is returned because dropping it deletes the Corpus.
fn fixture() -> Option<(TempDir, BoothConfig)> {
    if !models_present() {
        return None;
    }

    let directory = tempfile::tempdir().expect("a temporary directory");
    let text = format!(
        r#"
        [models]
        dir = "{models}"

        [models.detector]
        file = "face_detection_yunet_2023mar.onnx"

        [models.embedder]
        file = "dinov2-small.onnx"

        [corpus]
        dir = "{corpus}"

        [grid]
        cols = 5
        rows = 4
        "#,
        models = models_dir().display(),
        corpus = directory.path().display(),
    );
    let config = BoothConfig::parse(&text, directory.path().join("booth.toml"))
        .expect("valid configuration");

    Some((directory, config))
}

/// A booth whose camera replays `photographs`.
fn booth(config: &BoothConfig, photographs: Vec<PathBuf>) -> Pipeline {
    let camera: Box<dyn Camera> = Box::new(FakeCamera::replaying(photographs));

    Pipeline::open(config, camera).expect("the booth opens")
}

#[test]
fn should_put_one_new_face_on_the_wall_when_the_shutter_is_pressed_once() {
    let Some((_directory, config)) = fixture() else {
        return;
    };
    let mut booth = booth(&config, vec![sample_path("1.jpg")]);

    let captured = booth.capture().expect("a capture");

    let Captured::Face { id, .. } = captured else {
        panic!("the sample photograph holds one face; got {captured:?}");
    };
    assert_eq!(booth.face_count().expect("a readable corpus"), 1);
    let portraits = booth.portraits(config.grid()).expect("a window");
    assert_eq!(portraits.len(), 1, "one press, one face on the wall");
    assert_eq!(portraits[0].face(), id);
}

#[test]
fn should_report_what_the_capture_cost_when_a_face_is_stored() {
    // Stage 1 is the hardware evaluation instrument (ADR-0006): a Capture that
    // reports nothing cannot be compared between candidate machines.
    let Some((_directory, config)) = fixture() else {
        return;
    };
    let mut booth = booth(&config, vec![sample_path("1.jpg")]);

    let captured = booth.capture().expect("a capture");

    let Captured::Face { timings, .. } = captured else {
        panic!("the sample photograph holds one face; got {captured:?}");
    };
    assert!(timings.embed > std::time::Duration::ZERO);
    assert!(timings.total() >= timings.embed);
}

#[test]
fn should_still_hold_the_face_when_the_booth_is_restarted() {
    let Some((_directory, config)) = fixture() else {
        return;
    };

    let stored = {
        let mut booth = booth(&config, vec![sample_path("1.jpg")]);
        let captured = booth.capture().expect("a capture");
        booth.close().expect("the camera closes");

        match captured {
            Captured::Face { id, .. } => id,
            other => panic!("the sample photograph holds one face; got {other:?}"),
        }
    };

    // A second process against the same Corpus directory: the archive persists
    // between installations and does not depend on the booth staying up.
    let restarted = booth(&config, vec![sample_path("2.jpg")]);

    let portraits = restarted.portraits(config.grid()).expect("a window");
    assert_eq!(portraits.len(), 1);
    assert_eq!(portraits[0].face(), stored);
    assert!(
        portraits[0].display_crop().is_file(),
        "the face the wall draws must still have its display crop on disk"
    );
}

#[test]
fn should_leave_the_corpus_unchanged_when_the_frame_holds_no_face() {
    let Some((_directory, config)) = fixture() else {
        return;
    };
    // A flat frame: the right shape, nobody in it.
    let camera: Box<dyn Camera> = Box::new(FakeCamera::still(640, 480));
    let mut booth = Pipeline::open(&config, camera).expect("the booth opens");

    let captured = booth.capture().expect("a capture with nobody in it");

    assert_eq!(captured, Captured::NoFace);
    assert_eq!(
        booth.face_count().expect("a readable corpus"),
        0,
        "a capture with no face must leave the corpus untouched"
    );
}

#[test]
fn should_keep_taking_captures_when_one_of_them_held_no_face() {
    let Some((_directory, config)) = fixture() else {
        return;
    };
    let camera: Box<dyn Camera> = Box::new(FakeCamera::still(640, 480));
    let mut booth = Pipeline::open(&config, camera).expect("the booth opens");
    booth.capture().expect("a capture with nobody in it");

    // The next Visitor gets their turn: an empty frame does not end the piece.
    assert!(booth.capture().is_ok());
}

#[test]
fn should_report_the_camera_the_booth_actually_opened() {
    let Some((_directory, config)) = fixture() else {
        return;
    };
    let booth = booth(&config, vec![sample_path("1.jpg")]);

    assert_eq!(booth.camera().backend, "fake");
}

#[test]
fn should_refuse_to_start_when_the_camera_is_absent() {
    let Some((_directory, config)) = fixture() else {
        return;
    };
    let camera: Box<dyn Camera> = Box::new(FakeCamera::failing_to_open(CameraError::NotFound {
        requested: CameraSelector::Default,
    }));

    let Err(error) = Pipeline::open(&config, camera) else {
        panic!("a booth with no camera must not start");
    };

    // Absent, busy and disconnected reach the operator as different things:
    // the variant survives the trip out of the pipeline, not just the word.
    assert!(
        matches!(
            error,
            PipelineError::Capture(ShutterError::Camera(CameraError::NotFound { .. }))
        ),
        "{error}"
    );
}
