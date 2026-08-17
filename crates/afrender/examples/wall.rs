//! Point it at a Corpus; it puts the Faces on a wall.
//!
//! No camera and no models are involved: this draws what `afstore` already
//! holds, which is how the renderer is exercised before there is a booth to run
//! it in.
//!
//! ```text
//! cargo run -p afrender --example wall -- corpus
//! cargo run -p afrender --example wall -- corpus --cols 8 --rows 5 --offset 40
//! ```
//!
//! A Corpus with more Faces than the Grid has Cells shows a Window onto it;
//! `--offset` is where that Window sits, which is what Drift will move in
//! Stage 3.

use std::path::PathBuf;
use std::process::ExitCode;

use afcore::GridSpec;
use afrender::{Portrait, WallSpec, show};
use afstore::Corpus;

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

const USAGE: &str = "usage: wall <corpus-dir> [--cols N] [--rows N] [--offset N]";

struct Options {
    corpus: PathBuf,
    cols: u32,
    rows: u32,
    offset: u64,
}

impl Options {
    fn parse(arguments: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut corpus = None;
        let mut cols = 5;
        let mut rows = 4;
        let mut offset = 0;
        let mut arguments = arguments.peekable();

        while let Some(argument) = arguments.next() {
            let mut value = |flag: &str| {
                arguments
                    .next()
                    .ok_or_else(|| format!("{flag} needs a value"))
            };

            match argument.as_str() {
                "--cols" => cols = parse(&value("--cols")?, "--cols")?,
                "--rows" => rows = parse(&value("--rows")?, "--rows")?,
                "--offset" => offset = parse(&value("--offset")?, "--offset")?,
                flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}")),
                path => corpus = Some(PathBuf::from(path)),
            }
        }

        Ok(Self {
            corpus: corpus.ok_or("name the corpus directory")?,
            cols,
            rows,
            offset,
        })
    }
}

fn parse<T: std::str::FromStr>(value: &str, flag: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("{flag} cannot read {value}"))
}

fn run(options: &Options) -> Result<(), String> {
    let grid = GridSpec::new(options.cols, options.rows).map_err(|error| error.to_string())?;
    let corpus = Corpus::open(&options.corpus).map_err(|error| error.to_string())?;

    let faces = corpus
        .window(options.offset, grid.cell_count())
        .map_err(|error| error.to_string())?;
    println!(
        "{} of {} faces, in a {}x{} grid",
        faces.len(),
        corpus.count().map_err(|error| error.to_string())?,
        grid.cols(),
        grid.rows()
    );

    let portraits = faces
        .iter()
        .map(|face| Portrait::new(face.id(), face.display_path()))
        .collect();

    show(WallSpec::new(grid), portraits).map_err(|error| error.to_string())
}
