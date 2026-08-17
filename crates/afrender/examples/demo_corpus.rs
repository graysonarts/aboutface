//! Fills a Corpus with photographs so the wall has something to draw.
//!
//! A fixture builder, not part of the piece: it ingests whole photographs as
//! though they were display crops, with arbitrary Embeddings, so the renderer
//! can be worked on with no camera and no models on the machine. Nothing here
//! detects a face, and the Embeddings it writes mean nothing — the wall does
//! not read them, and Stage 2's layout must not be pointed at this.
//!
//! ```text
//! cargo run -p afrender --example demo_corpus -- corpus samples/*.jpg
//! cargo run -p afrender --example wall -- corpus
//! ```

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::SystemTime;

use afcore::{Embedding, ModelId};
use afstore::{Corpus, NewFace};
use image::RgbImage;

fn main() -> ExitCode {
    let mut arguments = std::env::args().skip(1);
    let Some(root) = arguments.next().map(PathBuf::from) else {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    };
    let photographs: Vec<PathBuf> = arguments.map(PathBuf::from).collect();
    if photographs.is_empty() {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    }

    match run(&root, &photographs) {
        Ok(count) => {
            println!("{count} faces in {}", root.display());
            ExitCode::SUCCESS
        }
        Err(report) => {
            eprintln!("{report}");
            ExitCode::FAILURE
        }
    }
}

const USAGE: &str = "usage: demo_corpus <corpus-dir> <photograph>...";

/// The identifier these fixtures claim their Embeddings came from.
///
/// Deliberately not a real model name: a Corpus seeded by this tool must not
/// be mistaken for one a booth produced.
const MODEL: &str = "demo-fixture";

fn run(root: &PathBuf, photographs: &[PathBuf]) -> Result<u64, String> {
    let mut corpus = Corpus::open(root).map_err(|error| error.to_string())?;

    for (index, path) in photographs.iter().enumerate() {
        let photograph = image::open(path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?
            .to_rgb8();
        let embedding = Embedding::new(ModelId::new(MODEL), direction(index))
            .map_err(|error| error.to_string())?;

        corpus
            .ingest(NewFace::new(
                embedding,
                RgbImage::new(112, 112),
                photograph.clone(),
                photograph,
                SystemTime::now(),
            ))
            .map_err(|error| error.to_string())?;
    }

    corpus.count().map_err(|error| error.to_string())
}

/// A direction that differs per fixture, so two of them are not identical.
fn direction(index: usize) -> Vec<f32> {
    let angle = index as f32;
    vec![angle.cos(), angle.sin(), 0.5, -0.25]
}
