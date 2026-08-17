//! Where the ONNX files live and which [`ModelId`] each one carries.
//!
//! The files are fetched, never committed (`docs/models.md`). Which DINOv2 ViT
//! size is still open (ADR-0006, ADR-0007), so neither the file name nor the
//! `ModelId` may be a constant: both come from configuration, and the
//! `ModelId` is derived from the file name when configuration does not state
//! one.

use std::fmt;
use std::path::{Path, PathBuf};

use afcore::ModelId;

/// What a model does in the pipeline.
///
/// Two roles, settled in ADR-0007: YuNet (MIT) detects and aligns, DINOv2
/// (Apache 2.0) embeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ModelRole {
    /// Finds the face and the five landmarks alignment needs.
    Detector,
    /// Turns an aligned crop into an [`afcore::Embedding`].
    Embedder,
}

impl ModelRole {
    /// The role's name as it appears in configuration and in the self-check.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Detector => "detector",
            Self::Embedder => "embedder",
        }
    }
}

impl fmt::Display for ModelRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Ways a model asset can be unusable.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ModelError {
    /// The configured file name carries no stem to derive a `ModelId` from.
    #[error("the {role} model file name {path:?} has no stem to derive a model id from")]
    UnnamedFile {
        /// Role whose file name is unusable.
        role: ModelRole,
        /// The offending file name.
        path: PathBuf,
    },

    /// Configuration supplied a blank `model_id`.
    #[error("the {role} model id is empty; remove it to derive one from the file name")]
    EmptyModelId {
        /// Role whose configured id is blank.
        role: ModelRole,
    },

    /// The ONNX file is not on disk. Fetching it is a documented procedure.
    #[error(
        "the {role} model is missing: {path} does not exist — fetch it with \
         `scripts/fetch-models.sh` (see docs/models.md)"
    )]
    Missing {
        /// Role whose file is absent.
        role: ModelRole,
        /// Where the file was expected.
        path: PathBuf,
    },
}

/// One model asset: where its ONNX file is and which [`ModelId`] it carries.
///
/// The `ModelId` is part of the store schema, not a runtime detail — changing
/// models invalidates every Embedding in the Corpus (ADR-0006).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    role: ModelRole,
    path: PathBuf,
    id: ModelId,
}

impl ModelSpec {
    /// Resolves a model's file against `dir` and settles its [`ModelId`].
    ///
    /// An absolute `file` is used as given; a relative one is joined onto
    /// `dir`. `id` overrides the derived identifier, which otherwise comes
    /// from the file's stem — so swapping `dinov2-small.onnx` for
    /// `dinov2-large.onnx` changes the `ModelId` too, and the Corpus can tell
    /// the two apart.
    ///
    /// # Errors
    ///
    /// Returns an error if `id` is blank, or if it is absent and `file` has no
    /// stem to derive from.
    pub fn new(
        role: ModelRole,
        dir: impl AsRef<Path>,
        file: impl AsRef<Path>,
        id: Option<&str>,
    ) -> Result<Self, ModelError> {
        let file = file.as_ref();
        let path = if file.is_absolute() {
            file.to_path_buf()
        } else {
            dir.as_ref().join(file)
        };

        let id = match id {
            Some(id) if id.trim().is_empty() => return Err(ModelError::EmptyModelId { role }),
            Some(id) => ModelId::new(id.trim()),
            None => derive_model_id(role, file)?,
        };

        Ok(Self { role, path, id })
    }

    /// What this model does in the pipeline.
    pub fn role(&self) -> ModelRole {
        self.role
    }

    /// The resolved path to the ONNX file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The identifier stored alongside every Embedding this model produces.
    pub fn id(&self) -> &ModelId {
        &self.id
    }

    /// Whether the ONNX file is on disk.
    pub fn exists(&self) -> bool {
        self.path.is_file()
    }

    /// Checks the file is present.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::Missing`] — carrying the fetch instructions — if
    /// the file is absent. A missing model is an operator's problem to fix, not
    /// a reason to panic.
    pub fn ensure_present(&self) -> Result<(), ModelError> {
        if self.exists() {
            Ok(())
        } else {
            Err(ModelError::Missing {
                role: self.role,
                path: self.path.clone(),
            })
        }
    }
}

/// Derives a [`ModelId`] from a model file name: its stem, unchanged.
///
/// Deliberately mechanical. The ViT size lives in the file name and therefore
/// in the identifier, and nothing in the code has to know which sizes exist.
fn derive_model_id(role: ModelRole, file: &Path) -> Result<ModelId, ModelError> {
    file.file_stem()
        .and_then(|stem| stem.to_str())
        .map(str::trim)
        .filter(|stem| !stem.is_empty())
        .map(ModelId::new)
        .ok_or_else(|| ModelError::UnnamedFile {
            role,
            path: file.to_path_buf(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_derive_model_id_from_file_stem_when_config_names_none() {
        let spec = ModelSpec::new(ModelRole::Embedder, "models", "dinov2-small.onnx", None)
            .expect("valid spec");

        assert_eq!(spec.id().as_str(), "dinov2-small");
    }

    #[test]
    fn should_change_model_id_when_vit_size_changes() {
        let small = ModelSpec::new(ModelRole::Embedder, "models", "dinov2-small.onnx", None)
            .expect("valid spec");
        let large = ModelSpec::new(ModelRole::Embedder, "models", "dinov2-large.onnx", None)
            .expect("valid spec");

        assert_ne!(small.id(), large.id());
    }

    #[test]
    fn should_prefer_configured_model_id_when_one_is_given() {
        let spec = ModelSpec::new(
            ModelRole::Embedder,
            "models",
            "model.onnx",
            Some("dinov2-vitb14"),
        )
        .expect("valid spec");

        assert_eq!(spec.id().as_str(), "dinov2-vitb14");
    }

    #[test]
    fn should_trim_configured_model_id_when_it_has_surrounding_space() {
        let spec = ModelSpec::new(
            ModelRole::Detector,
            "models",
            "model.onnx",
            Some("  yunet-2023mar  "),
        )
        .expect("valid spec");

        assert_eq!(spec.id().as_str(), "yunet-2023mar");
    }

    #[test]
    fn should_reject_blank_model_id_when_config_gives_one() {
        let error = ModelSpec::new(ModelRole::Detector, "models", "model.onnx", Some("   "))
            .expect_err("blank id");

        assert_eq!(
            error,
            ModelError::EmptyModelId {
                role: ModelRole::Detector
            }
        );
    }

    #[test]
    fn should_reject_file_name_when_it_has_no_stem() {
        let error = ModelSpec::new(ModelRole::Detector, "models", "..", None).expect_err("no stem");

        assert!(matches!(error, ModelError::UnnamedFile { .. }));
    }

    #[test]
    fn should_resolve_relative_file_against_the_models_dir() {
        let spec = ModelSpec::new(ModelRole::Detector, "/opt/af/models", "yunet.onnx", None)
            .expect("valid spec");

        assert_eq!(spec.path(), Path::new("/opt/af/models/yunet.onnx"));
    }

    #[test]
    fn should_keep_absolute_file_when_config_gives_one() {
        let spec = ModelSpec::new(
            ModelRole::Detector,
            "/opt/af/models",
            "/srv/shared/yunet.onnx",
            None,
        )
        .expect("valid spec");

        assert_eq!(spec.path(), Path::new("/srv/shared/yunet.onnx"));
    }

    #[test]
    fn should_report_missing_when_the_onnx_file_is_absent() {
        let spec = ModelSpec::new(
            ModelRole::Embedder,
            "/nonexistent",
            "dinov2-small.onnx",
            None,
        )
        .expect("valid spec");

        assert!(!spec.exists());
        let error = spec.ensure_present().expect_err("absent file");
        let message = error.to_string();
        assert!(
            message.contains("docs/models.md"),
            "message must say how to fix it, got: {message}"
        );
    }

    #[test]
    fn should_report_present_when_the_file_is_on_disk() {
        // Any file on disk will do; this test is about presence, not contents.
        let spec = ModelSpec::new(
            ModelRole::Embedder,
            env!("CARGO_MANIFEST_DIR"),
            "Cargo.toml",
            None,
        )
        .expect("valid spec");

        assert!(spec.exists());
        assert!(spec.ensure_present().is_ok());
    }
}
