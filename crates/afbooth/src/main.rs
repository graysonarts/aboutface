//! The About:Face installation binary.
//!
//! Wires the crates together, owns configuration, and performs the startup
//! self-check. Grid size, Drift rate, map resolution and the model assets are
//! configuration rather than constants, so the piece can be scaled to the
//! machine it is running on — and pointed at a different DINOv2 ViT size —
//! without code changes (ADR-0006).
//!
//! Stage 1 is the whole piece in one line: press the Shutter, and the Face
//! joins the wall. Ordering, Assignment and motion are Stage 2; Consent
//! Records and Receipt Codes are Stage 4.

use std::process::ExitCode;

use afbooth::config::BoothConfig;
use afbooth::pipeline::{Captured, Pipeline};
use afbooth::selfcheck::SelfCheck;
use afcapture::{Camera, NokhwaCamera};
use afrender::{Portrait, WallSpec, show_live};
use anyhow::Context;

fn main() -> ExitCode {
    let config = match BoothConfig::load() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("configuration error: {error}");
            return ExitCode::FAILURE;
        }
    };

    let check = SelfCheck::run(&config);
    if !check.is_ready() {
        eprintln!();
        for problem in check.missing() {
            eprintln!("{problem}");
        }
        return ExitCode::FAILURE;
    }

    match run(&config) {
        Ok(()) => ExitCode::SUCCESS,
        // `{error:#}` prints the whole source chain: an operator standing in a
        // gallery needs "camera failed: no camera at the default index", not
        // the outermost sentence on its own.
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::FAILURE
        }
    }
}

/// Opens the camera, the models and the Corpus, then runs the wall until it is
/// closed.
fn run(config: &BoothConfig) -> anyhow::Result<()> {
    let camera: Box<dyn Camera> = Box::new(NokhwaCamera::new(config.camera().clone()));
    let mut booth = Pipeline::open(config, camera).context("the booth could not start")?;

    println!("camera opened: {}", booth.camera());
    println!("corpus holds {} faces", booth.face_count()?);
    println!("press SPACE to capture, ESC to quit");

    let grid = config.grid();
    let portraits = booth.portraits(grid)?;

    // The wall asks; this answers. A Capture that produced no Face — nobody in
    // the frame, or a crowd — answers with the Faces already on the wall, so
    // the Corpus and the screen both stay as they were.
    let mut shown = portraits.clone();
    let result = show_live(WallSpec::new(grid), portraits, || {
        shown = capture(&mut booth, grid, &shown);
        shown.clone()
    })
    .context("the wall failed");

    // The camera is released whether the wall closed cleanly or failed: a booth
    // that quits holding the device cannot be restarted without unplugging it.
    if let Err(error) = booth.close() {
        eprintln!("camera did not close cleanly: {error}");
    }

    result
}

/// One press: capture, report, and hand back the Window to show.
fn capture(booth: &mut Pipeline, grid: afcore::GridSpec, shown: &[Portrait]) -> Vec<Portrait> {
    match booth.capture() {
        Ok(Captured::Face { id, timings }) => {
            println!("{id}: {timings}");
            match booth.portraits(grid) {
                Ok(portraits) => portraits,
                Err(error) => {
                    eprintln!("cannot read the corpus: {error}");
                    shown.to_vec()
                }
            }
        }
        Ok(Captured::NoFace) => {
            println!("no face in the frame — nothing captured");
            shown.to_vec()
        }
        Ok(Captured::Several(count)) => {
            println!("{count} faces in the frame — one visitor at a time");
            shown.to_vec()
        }
        // A failed Capture must not close the booth: the next Visitor gets
        // their turn, and the operator gets the reason on stderr.
        Err(error) => {
            eprintln!("capture failed: {:#}", anyhow::Error::new(error));
            shown.to_vec()
        }
    }
}
