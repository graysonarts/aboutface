//! The About:Face installation binary.
//!
//! Wires the crates together, owns configuration, and performs the startup
//! self-check. Grid size, Drift rate, and map resolution are configuration
//! rather than constants, so the piece can be scaled to the machine it is
//! running on without code changes (ADR-0006).

use afcore::{GridSpec, MAX_CELLS, MIN_CELLS};

fn main() {
    // Stage 0: the workspace stands up and the pieces are wired together.
    // Stage 1 replaces this with the tracer bullet — Shutter to wall.
    println!("About:Face {}", env!("CARGO_PKG_VERSION"));
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

    println!("no capture pipeline yet — see docs/implementation-plan.md, Stage 1");
}
