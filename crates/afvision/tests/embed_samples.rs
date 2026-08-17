//! The embedder's `ort` boundary, exercised against the real DINOv2 file.
//!
//! Like `detect_samples`, this is an integration test because it crosses a
//! system boundary the crate does not own. It says nothing about whether the
//! numbers are *meaningful* — that is the sanity check in Stage 1 of the
//! implementation plan, which needs captures of the same person under different
//! lighting. What it does assert is the contract: unit norm, a width that came
//! from the loaded graph, the `ModelId` of the file actually loaded, and a
//! refusal to compare across models.
//!
//! The ONNX files are fetched, not committed (`docs/models.md`), so these tests
//! report and return when the model is absent — CI builds without it. Run
//! `./scripts/fetch-models.sh` to have them do real work.

use std::path::{Path, PathBuf};

use afcore::{Embedding, ModelId};
use afvision::{
    ALIGNED_SIZE, FaceDetector, FaceEmbedder, ModelRole, ModelSpec, align,
    select_execution_provider,
};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn embedder_spec() -> ModelSpec {
    ModelSpec::new(
        ModelRole::Embedder,
        repo_root().join("models"),
        "dinov2-small.onnx",
        None,
    )
    .expect("valid spec")
}

/// The embedder, or `None` when the ONNX file has not been fetched.
fn embedder() -> Option<FaceEmbedder> {
    let spec = embedder_spec();
    if !spec.exists() {
        eprintln!(
            "skipping: {} is absent — run ./scripts/fetch-models.sh",
            spec.path().display()
        );
        return None;
    }

    Some(FaceEmbedder::open(&spec, select_execution_provider()).expect("the embedder loads"))
}

/// The detector, or `None` when its ONNX file has not been fetched.
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

/// The aligned crop of the sole face in a sample photograph.
fn aligned_crop(detector: &mut FaceDetector, name: &str) -> image::RgbImage {
    let image = image::open(repo_root().join("samples").join(name))
        .expect("the fixture decodes")
        .to_rgb8();
    let face = detector
        .detect(&image)
        .expect("inference runs")
        .into_sole()
        .unwrap_or_else(|error| panic!("{name}: {error}"));

    align(&image, face.landmarks()).expect("alignable landmarks")
}

fn norm(embedding: &Embedding) -> f32 {
    embedding
        .as_slice()
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
}

#[test]
fn should_embed_an_aligned_crop_as_a_unit_vector_carrying_the_loaded_model_id() {
    let (Some(mut detector), Some(mut embedder)) = (detector(), embedder()) else {
        return;
    };

    let crop = aligned_crop(&mut detector, "1.jpg");
    assert_eq!(crop.dimensions(), (ALIGNED_SIZE, ALIGNED_SIZE));

    let embedding = embedder.embed(&crop).expect("inference runs");

    assert!(
        (norm(&embedding) - 1.0).abs() < 1e-5,
        "{}",
        norm(&embedding)
    );
    assert_eq!(embedding.model(), embedder_spec().id());
    assert_eq!(embedding.dim(), embedder.dim());
}

#[test]
fn should_discover_the_width_from_the_loaded_graph_rather_than_assume_one() {
    let Some(embedder) = embedder() else {
        return;
    };

    // ViT-S/14 is 384 wide — the width of the file `embedder_spec` names, and
    // not a number anything in the crate assumes. Swapping the file for base
    // (768) or large (1024) must change this without any code changing.
    assert_eq!(embedder.dim(), 384);
}

#[test]
fn should_return_the_same_embedding_when_the_same_crop_is_embedded_twice() {
    let (Some(mut detector), Some(mut embedder)) = (detector(), embedder()) else {
        return;
    };

    let crop = aligned_crop(&mut detector, "1.jpg");
    let first = embedder.embed(&crop).expect("inference runs");
    let second = embedder.embed(&crop).expect("inference runs");

    let distance = first.cosine_distance(&second).expect("comparable");
    assert!(distance < 1e-5, "distance was {distance}");
}

#[test]
fn should_compare_two_photographs_as_a_distance_rather_than_a_failure() {
    let (Some(mut detector), Some(mut embedder)) = (detector(), embedder()) else {
        return;
    };

    let one = embedder
        .embed(&aligned_crop(&mut detector, "1.jpg"))
        .expect("inference runs");
    let three = embedder
        .embed(&aligned_crop(&mut detector, "3.jpg"))
        .expect("inference runs");

    let distance = one.cosine_distance(&three).expect("comparable");
    assert!(
        distance.is_finite() && (0.0..=2.0).contains(&distance),
        "distance was {distance}"
    );
    // Two different photographs are not the same photograph. Nothing stronger
    // is claimed here; whether the number tracks resemblance is Stage 1's
    // sanity check.
    assert!(distance > 1e-4, "distance was {distance}");
}

#[test]
fn should_refuse_to_compare_a_real_embedding_against_another_model() {
    let (Some(mut detector), Some(mut embedder)) = (detector(), embedder()) else {
        return;
    };

    let real = embedder
        .embed(&aligned_crop(&mut detector, "1.jpg"))
        .expect("inference runs");
    let impostor = Embedding::new(ModelId::new("some-other-model"), real.as_slice().to_vec())
        .expect("valid embedding");

    assert!(
        real.cosine_distance(&impostor).is_err(),
        "comparing across models must fail, not return a number"
    );
}

#[test]
fn should_refuse_to_compare_a_real_embedding_against_a_narrower_one() {
    let (Some(mut detector), Some(mut embedder)) = (detector(), embedder()) else {
        return;
    };

    // What a stale Corpus entry from a smaller ViT would look like: same model
    // name, fewer dimensions.
    let real = embedder
        .embed(&aligned_crop(&mut detector, "1.jpg"))
        .expect("inference runs");
    let stale = Embedding::new(
        real.model().clone(),
        real.as_slice()[..real.dim() / 2].to_vec(),
    )
    .expect("valid embedding");

    assert!(
        real.cosine_distance(&stale).is_err(),
        "comparing across widths must fail, not return a number"
    );
}
