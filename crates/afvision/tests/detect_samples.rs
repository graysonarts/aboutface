//! The `ort` boundary, exercised against the real ONNX file.
//!
//! Everything else in `afvision` is unit-tested in place; this is the one test
//! that loads the YuNet graph, runs inference, and checks the decode against a
//! photograph. It is an integration test because that is what it is: a system
//! boundary the crate does not own.
//!
//! The ONNX files are fetched, not committed (`docs/models.md`), so these tests
//! report and return when the model is absent — CI builds without it. Run
//! `./scripts/fetch-models.sh` to have them do real work.

use std::path::{Path, PathBuf};

use afvision::{
    ALIGNED_SIZE, DisplayCropSpec, FaceDetector, Faces, ModelRole, ModelSpec,
    select_execution_provider,
};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// The detector, or `None` when the ONNX file has not been fetched.
fn detector() -> Option<FaceDetector> {
    let spec = ModelSpec::new(
        ModelRole::Detector,
        repo_root().join("models"),
        "face_detection_yunet_2023mar.onnx",
        None,
    )
    .expect("valid spec");

    if !spec.exists() {
        eprintln!(
            "skipping: {} is absent — run ./scripts/fetch-models.sh",
            spec.path().display()
        );
        return None;
    }

    Some(FaceDetector::open(&spec, select_execution_provider()).expect("the detector loads"))
}

fn sample(name: &str) -> image::RgbImage {
    image::open(repo_root().join("samples").join(name))
        .expect("the fixture decodes")
        .to_rgb8()
}

#[test]
fn should_find_one_face_with_plausible_landmarks_in_each_sample_portrait() {
    let Some(mut detector) = detector() else {
        return;
    };

    for name in ["1.jpg", "3.jpg", "4.jpg"] {
        let image = sample(name);
        let faces = detector.detect(&image).expect("inference runs");

        let face = faces
            .into_sole()
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let bbox = face.bbox();
        let landmarks = face.landmarks();

        assert!(
            face.score() > 0.9,
            "{name}: weak detection {}",
            face.score()
        );
        assert!(
            bbox.x() >= 0.0
                && bbox.y() >= 0.0
                && bbox.right() <= image.width() as f32
                && bbox.bottom() <= image.height() as f32,
            "{name}: box {bbox:?} leaves the frame"
        );
        // A front-facing portrait: the eyes are the right way round, above the
        // mouth, and inside the box the detector drew.
        assert!(
            landmarks.left_eye.x < landmarks.right_eye.x,
            "{name}: eyes are transposed"
        );
        assert!(
            landmarks.left_eye.y < landmarks.mouth_left.y,
            "{name}: the eye is below the mouth"
        );
        for point in landmarks.as_array() {
            assert!(
                point.x >= bbox.x()
                    && point.x <= bbox.right()
                    && point.y >= bbox.y()
                    && point.y <= bbox.bottom(),
                "{name}: landmark {point:?} outside {bbox:?}"
            );
        }
    }
}

#[test]
fn should_produce_both_crops_from_a_detection() {
    let Some(mut detector) = detector() else {
        return;
    };

    let image = sample("1.jpg");
    let face = detector
        .detect(&image)
        .expect("inference runs")
        .into_sole()
        .expect("one face");

    let aligned = afvision::align(&image, face.landmarks()).expect("alignable landmarks");
    let spec = DisplayCropSpec::default();
    let display = afvision::display_crop(&image, face.bbox(), &spec);

    assert_eq!(aligned.dimensions(), (ALIGNED_SIZE, ALIGNED_SIZE));
    assert_eq!(display.dimensions(), (spec.width(), spec.height()));
}

#[test]
fn should_report_no_face_when_the_frame_is_blank() {
    let Some(mut detector) = detector() else {
        return;
    };

    let blank = image::RgbImage::from_pixel(640, 480, image::Rgb([128, 128, 128]));

    assert_eq!(
        detector.detect(&blank).expect("inference runs"),
        Faces::None
    );
}
