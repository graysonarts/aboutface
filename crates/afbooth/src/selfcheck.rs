//! The startup self-check.
//!
//! A permanent fixture, not scaffolding: the hardware is not chosen, the CPU
//! path must stay viable, and Stage 1 is meant to be run on every candidate
//! machine (ADR-0006). "Which provider did I actually get, and are the models
//! here" is therefore an answer the operator needs at every launch, before
//! anything else happens.

use afvision::ModelSpec;

use crate::config::BoothConfig;

/// What the self-check found.
///
/// Only the problems are carried forward; the machine's provider and runtime
/// have already been reported by the time this exists. Stage 1 will want the
/// selected provider back out of here to build its Sessions with.
pub struct SelfCheck {
    missing: Vec<String>,
}

impl SelfCheck {
    /// Runs the check and prints its report to stdout.
    ///
    /// Nothing here panics or bails early: a booth with a missing model should
    /// still tell the operator everything it knows about the machine.
    pub fn run(config: &BoothConfig) -> Self {
        let provider = afvision::select_execution_provider();
        let runtime = afvision::runtime_info();

        println!("About:Face {}", env!("CARGO_PKG_VERSION"));
        println!("config: {}", config.source().display());
        println!("models dir: {}", config.models_dir().display());
        println!("corpus dir: {}", config.corpus_dir().display());
        println!("execution provider: {provider}");
        println!("onnx runtime: {runtime}");
        println!(
            "grid: {}x{} = {} cells",
            config.grid().cols(),
            config.grid().rows(),
            config.grid().cell_count()
        );

        let mut missing = Vec::new();
        match afrender::adapter_report() {
            Some(adapter) => println!("render backend: {adapter}"),
            None => {
                println!("render backend: NONE");
                missing.push(
                    "no GPU adapter this build can draw on — the wall cannot open".to_owned(),
                );
            }
        }
        // Which camera resolved is only knowable once it is opened, which the
        // booth does next: this is the device it will ask for.
        println!("camera requested: {}", config.camera());
        let crop = config.display_crop();
        println!(
            "display crop: {}x{} (margin {:.2}, bias {:.2})",
            crop.width(),
            crop.height(),
            crop.margin(),
            crop.vertical_bias()
        );
        println!("models:");

        for model in config.models() {
            println!("  {}", describe(model));
            if let Err(error) = model.ensure_present() {
                missing.push(error.to_string());
            }
        }

        Self { missing }
    }

    /// Whether every configured model file is on disk.
    pub fn is_ready(&self) -> bool {
        self.missing.is_empty()
    }

    /// One actionable message per absent model file.
    pub fn missing(&self) -> &[String] {
        &self.missing
    }
}

/// One model's line in the report: role, presence, path, identifier.
fn describe(model: &ModelSpec) -> String {
    let presence = if model.exists() { "present" } else { "MISSING" };
    format!(
        "{role:<8} {presence:<7} {path} (model id: {id})",
        role = model.role(),
        path = model.path().display(),
        id = model.id(),
    )
}
