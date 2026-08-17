//! The About:Face installation binary.
//!
//! Wires the crates together, owns configuration, and performs the startup
//! self-check. Grid size, Drift rate, map resolution and the model assets are
//! configuration rather than constants, so the piece can be scaled to the
//! machine it is running on — and pointed at a different DINOv2 ViT size —
//! without code changes (ADR-0006).

mod config;
mod selfcheck;

use std::process::ExitCode;

use afcore::{GridSpec, MAX_CELLS, MIN_CELLS};

use crate::config::BoothConfig;
use crate::selfcheck::SelfCheck;

fn main() -> ExitCode {
    let config = match BoothConfig::load() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("configuration error: {error}");
            return ExitCode::FAILURE;
        }
    };

    let check = SelfCheck::run(&config);

    println!("grid range: {MIN_CELLS}..={MAX_CELLS} cells");
    match GridSpec::new(4, 4) {
        Ok(spec) => println!(
            "example grid: {}x{} = {} cells",
            spec.cols(),
            spec.rows(),
            spec.cell_count()
        ),
        Err(error) => eprintln!("invalid grid: {error}"),
    }

    if !check.is_ready() {
        eprintln!();
        for problem in check.missing() {
            eprintln!("{problem}");
        }
        return ExitCode::FAILURE;
    }

    // Stage 1 replaces this with the tracer bullet — Shutter to wall.
    println!("no capture pipeline yet — see docs/implementation-plan.md, Stage 1");
    ExitCode::SUCCESS
}
