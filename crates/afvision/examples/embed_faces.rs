//! Point it at two photographs; it embeds both and prints the distance.
//!
//! The whole `detect -> align -> embed` path, off files on disk, with no booth
//! and no camera:
//!
//! ```text
//! cargo run -p afvision --example embed_faces -- samples/1.jpg samples/3.jpg
//! cargo run -p afvision --example embed_faces -- a.jpg b.jpg --embedder models/dinov2-base.onnx
//! ```
//!
//! It prints each Embedding's norm — 1.0, because Embeddings are normalized —
//! its width, the `ModelId` recorded on it, and the cosine distance between the
//! two. Whether that distance tracks resemblance is the Stage 1 sanity check,
//! not something this tool claims.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use afcore::Embedding;
use afvision::{
    FaceDetector, FaceEmbedder, Faces, ModelRole, ModelSpec, align, select_execution_provider,
};

fn main() -> ExitCode {
    let options = match Options::parse(std::env::args().skip(1)) {
        Ok(options) => options,
        Err(message) => {
            eprintln!("{message}\n\n{USAGE}");
            return ExitCode::FAILURE;
        }
    };

    match run(&options) {
        Ok(()) => ExitCode::SUCCESS,
        Err(report) => {
            eprintln!("{report}");
            ExitCode::FAILURE
        }
    }
}

const USAGE: &str =
    "usage: embed_faces <photograph> <photograph> [--detector FILE] [--embedder FILE]";

struct Options {
    photographs: Vec<PathBuf>,
    detector: PathBuf,
    embedder: PathBuf,
}

impl Options {
    fn parse(arguments: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut photographs = Vec::new();
        let mut detector = PathBuf::from("models/face_detection_yunet_2023mar.onnx");
        let mut embedder = PathBuf::from("models/dinov2-small.onnx");

        let mut arguments = arguments;
        while let Some(argument) = arguments.next() {
            let mut value = |name: &str| {
                arguments
                    .next()
                    .ok_or_else(|| format!("{name} needs a value"))
            };
            match argument.as_str() {
                "--detector" => detector = PathBuf::from(value("--detector")?),
                "--embedder" => embedder = PathBuf::from(value("--embedder")?),
                other if other.starts_with("--") => {
                    return Err(format!("unknown option {other}"));
                }
                other => photographs.push(PathBuf::from(other)),
            }
        }

        if photographs.len() != 2 {
            return Err(format!(
                "two photographs are needed to compare two Faces; got {}",
                photographs.len()
            ));
        }

        Ok(Self {
            photographs,
            detector,
            embedder,
        })
    }
}

fn run(options: &Options) -> Result<(), String> {
    let provider = select_execution_provider();
    println!("execution provider: {provider}");

    let detector_spec = spec(ModelRole::Detector, &options.detector)?;
    let embedder_spec = spec(ModelRole::Embedder, &options.embedder)?;

    let mut detector =
        FaceDetector::open(&detector_spec, provider).map_err(|error| error.to_string())?;
    let mut embedder =
        FaceEmbedder::open(&embedder_spec, provider).map_err(|error| error.to_string())?;
    println!(
        "embedder: {} ({} dimensions, {}x{} input)",
        embedder.model(),
        embedder.dim(),
        embedder.input_size(),
        embedder.input_size()
    );

    let mut embeddings = Vec::new();
    for photograph in &options.photographs {
        let embedding = embed(&mut detector, &mut embedder, photograph)?;
        println!(
            "{}: norm {:.6}, {} dimensions, model {}",
            photograph.display(),
            norm(&embedding),
            embedding.dim(),
            embedding.model()
        );
        embeddings.push(embedding);
    }

    let distance = embeddings[0]
        .cosine_distance(&embeddings[1])
        .map_err(|error| error.to_string())?;
    println!("cosine distance: {distance:.6}  (0 identical, 1 unrelated, 2 opposite)");

    Ok(())
}

fn spec(role: ModelRole, file: &Path) -> Result<ModelSpec, String> {
    let spec = ModelSpec::new(role, ".", file, None).map_err(|error| error.to_string())?;
    spec.ensure_present().map_err(|error| error.to_string())?;

    Ok(spec)
}

/// Detect, align, embed — the whole path over one photograph.
///
/// No face and several faces stop the tool rather than being papered over: the
/// multi-face policy at the Shutter is still an open question, and picking one
/// here would settle it by accident.
fn embed(
    detector: &mut FaceDetector,
    embedder: &mut FaceEmbedder,
    photograph: &Path,
) -> Result<Embedding, String> {
    let image = image::open(photograph)
        .map_err(|error| format!("cannot read {}: {error}", photograph.display()))?
        .to_rgb8();

    let face = match detector.detect(&image).map_err(|error| error.to_string())? {
        Faces::One(face) => face,
        Faces::None => return Err(format!("no face in {}", photograph.display())),
        Faces::Many(faces) => {
            return Err(format!(
                "{} faces in {}; this tool embeds one",
                faces.len(),
                photograph.display()
            ));
        }
    };

    let aligned = align(&image, face.landmarks()).map_err(|error| error.to_string())?;

    embedder
        .embed(&aligned)
        .map_err(|error| format!("{}: {error}", photograph.display()))
}

fn norm(embedding: &Embedding) -> f32 {
    embedding
        .as_slice()
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
}
