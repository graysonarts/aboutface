//! Which ONNX Runtime execution provider this machine gets.
//!
//! The hardware is not chosen (ADR-0006), so the provider is discovered rather
//! than assumed, and CPU is a supported answer rather than a failure. The
//! startup self-check reports the result at every launch, so a machine that
//! quietly fell back to CPU says so instead of being mysteriously slow.

use std::fmt;

use ort::ep::ExecutionProvider as _;

/// An execution provider this build can ask ONNX Runtime for.
///
/// Only the providers compiled into this build appear here. CUDA and TensorRT
/// join the list when there is a Linux GPU machine to test them on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProviderKind {
    /// Apple's CoreML, on macOS builds.
    CoreMl,
    /// The always-available fallback. Correct on CPU is the baseline
    /// (ADR-0006); accelerators are an optimization.
    Cpu,
}

impl ExecutionProviderKind {
    /// The provider's identifier as ONNX Runtime itself names it.
    pub fn name(self) -> &'static str {
        match self {
            #[cfg(target_os = "macos")]
            Self::CoreMl => ort::ep::CoreML::default().name(),
            #[cfg(not(target_os = "macos"))]
            Self::CoreMl => "CoreMLExecutionProvider",
            Self::Cpu => ort::ep::CPU::default().name(),
        }
    }

    /// Whether this build's ONNX Runtime reports the provider as available.
    ///
    /// Availability is a property of the runtime binary, not of a Session: a
    /// model whose operators the provider cannot handle still falls back
    /// per-node at inference time.
    pub fn is_available(self) -> bool {
        match self {
            #[cfg(target_os = "macos")]
            Self::CoreMl => ort::ep::CoreML::default().is_available().unwrap_or(false),
            #[cfg(not(target_os = "macos"))]
            Self::CoreMl => false,
            Self::Cpu => true,
        }
    }
}

impl fmt::Display for ExecutionProviderKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Preference order. The first available one wins; CPU always terminates it.
const PREFERENCE: &[ExecutionProviderKind] =
    &[ExecutionProviderKind::CoreMl, ExecutionProviderKind::Cpu];

/// Picks the execution provider this machine will run inference on.
///
/// Never fails: an unaccelerated machine gets [`ExecutionProviderKind::Cpu`],
/// which is a supported configuration, not a degraded one.
pub fn select_execution_provider() -> ExecutionProviderKind {
    PREFERENCE
        .iter()
        .copied()
        .find(|provider| provider.is_available())
        .unwrap_or(ExecutionProviderKind::Cpu)
}

/// The ONNX Runtime build backing this binary — version, commit, flags.
pub fn runtime_info() -> String {
    ort::info().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_always_offer_cpu_when_no_accelerator_exists() {
        assert!(ExecutionProviderKind::Cpu.is_available());
    }

    #[test]
    fn should_end_the_preference_order_with_cpu() {
        assert_eq!(PREFERENCE.last(), Some(&ExecutionProviderKind::Cpu));
    }

    #[test]
    fn should_select_an_available_provider_when_asked() {
        let selected = select_execution_provider();
        assert!(
            selected.is_available(),
            "selected {selected} is unavailable"
        );
    }

    #[test]
    fn should_name_the_provider_as_onnx_runtime_does() {
        assert_eq!(ExecutionProviderKind::Cpu.name(), "CPUExecutionProvider");
    }
}
