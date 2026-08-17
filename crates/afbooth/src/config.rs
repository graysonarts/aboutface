//! The booth's configuration file.
//!
//! Model paths and identifiers are configuration, not constants: which DINOv2
//! ViT size the piece runs follows the hardware decision, which is still open
//! (ADR-0006, ADR-0007). Nothing here knows that DINOv2 has sizes at all — it
//! knows only that a file name and an optional identifier arrive from a file.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use afvision::{ModelError, ModelRole, ModelSpec};
use serde::Deserialize;

/// Environment variable naming an explicit configuration file.
pub const CONFIG_ENV: &str = "AFBOOTH_CONFIG";

/// File loaded from the working directory when `AFBOOTH_CONFIG` is unset.
pub const DEFAULT_CONFIG_FILE: &str = "booth.toml";

/// Ways configuration can fail to produce a usable set of models.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// No configuration file at the resolved location.
    #[error(
        "no configuration file at {path} — copy the repository's `booth.toml`, \
         or point {CONFIG_ENV} at one"
    )]
    NotFound {
        /// Where the file was looked for.
        path: PathBuf,
    },

    /// The file exists but could not be read.
    #[error("cannot read the configuration file {path}: {source}")]
    Unreadable {
        /// The file that could not be read.
        path: PathBuf,
        /// The underlying I/O failure.
        source: std::io::Error,
    },

    /// The file is not valid TOML, or does not match the expected shape.
    #[error("cannot parse the configuration file {path}: {source}")]
    Invalid {
        /// The file that failed to parse.
        path: PathBuf,
        /// The underlying parse failure.
        source: toml::de::Error,
    },

    /// The file parsed, but a model entry is unusable.
    #[error(transparent)]
    Model(#[from] ModelError),
}

/// One model entry in the configuration file.
#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ModelEntry {
    /// ONNX file name, relative to the models directory unless absolute.
    file: PathBuf,
    /// Overrides the identifier derived from the file name.
    #[serde(default)]
    id: Option<String>,
}

/// The `[models]` table.
#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ModelsTable {
    /// Where the fetched ONNX files live; relative paths resolve against the
    /// configuration file's own directory, so a booth can be moved wholesale.
    #[serde(default = "default_models_dir")]
    dir: PathBuf,
    detector: ModelEntry,
    embedder: ModelEntry,
}

fn default_models_dir() -> PathBuf {
    PathBuf::from("models")
}

/// The configuration file as written on disk.
#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ConfigFile {
    models: ModelsTable,
}

/// The booth's resolved configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoothConfig {
    source: PathBuf,
    models_dir: PathBuf,
    detector: ModelSpec,
    embedder: ModelSpec,
}

impl BoothConfig {
    /// Loads configuration from `AFBOOTH_CONFIG`, or `booth.toml` in the
    /// working directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is absent, unreadable, malformed, or names
    /// a model it cannot turn into a [`ModelSpec`].
    pub fn load() -> Result<Self, ConfigError> {
        Self::load_from(resolve_config_path())
    }

    /// Loads configuration from an explicit path.
    ///
    /// # Errors
    ///
    /// As [`BoothConfig::load`].
    pub fn load_from(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(ConfigError::NotFound {
                path: path.to_path_buf(),
            });
        }

        let text = fs::read_to_string(path).map_err(|source| ConfigError::Unreadable {
            path: path.to_path_buf(),
            source,
        })?;

        Self::parse(&text, path)
    }

    /// Parses configuration text as if it had been read from `source`.
    ///
    /// Relative paths inside resolve against `source`'s parent directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the text is not valid configuration, or names a
    /// model it cannot turn into a [`ModelSpec`].
    pub fn parse(text: &str, source: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let source = source.as_ref();
        let file: ConfigFile =
            toml::from_str(text).map_err(|source_error| ConfigError::Invalid {
                path: source.to_path_buf(),
                source: source_error,
            })?;

        let base = source.parent().unwrap_or(Path::new("."));
        let models_dir = if file.models.dir.is_absolute() {
            file.models.dir.clone()
        } else {
            base.join(&file.models.dir)
        };

        let detector = spec(ModelRole::Detector, &models_dir, &file.models.detector)?;
        let embedder = spec(ModelRole::Embedder, &models_dir, &file.models.embedder)?;

        Ok(Self {
            source: source.to_path_buf(),
            models_dir,
            detector,
            embedder,
        })
    }

    /// The file this configuration came from.
    pub fn source(&self) -> &Path {
        &self.source
    }

    /// Where the fetched ONNX files are expected.
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    /// Every model the booth needs, in self-check order.
    pub fn models(&self) -> [&ModelSpec; 2] {
        [&self.detector, &self.embedder]
    }
}

impl fmt::Display for BoothConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.source.display())
    }
}

fn spec(role: ModelRole, dir: &Path, entry: &ModelEntry) -> Result<ModelSpec, ModelError> {
    ModelSpec::new(role, dir, &entry.file, entry.id.as_deref())
}

/// Where configuration will be read from.
///
/// An empty `AFBOOTH_CONFIG` counts as unset — a shell that exports the
/// variable blank should get the default, not a lookup of the empty path.
fn resolve_config_path() -> PathBuf {
    config_path_from(std::env::var_os(CONFIG_ENV))
}

fn config_path_from(configured: Option<std::ffi::OsString>) -> PathBuf {
    configured
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CONFIG_FILE))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"
        [models.detector]
        file = "face_detection_yunet_2023mar.onnx"

        [models.embedder]
        file = "dinov2-small.onnx"
    "#;

    fn parse(text: &str) -> Result<BoothConfig, ConfigError> {
        BoothConfig::parse(text, "/opt/af/booth.toml")
    }

    #[test]
    fn should_derive_both_model_ids_when_config_names_none() {
        let config = parse(MINIMAL).expect("valid config");
        let [detector, embedder] = config.models();

        assert_eq!(detector.id().as_str(), "face_detection_yunet_2023mar");
        assert_eq!(embedder.id().as_str(), "dinov2-small");
    }

    #[test]
    fn should_default_the_models_dir_when_config_omits_it() {
        let config = parse(MINIMAL).expect("valid config");

        assert_eq!(config.models_dir(), Path::new("/opt/af/models"));
    }

    #[test]
    fn should_resolve_relative_models_dir_against_the_config_file() {
        let config = parse(
            r#"
            [models]
            dir = "assets/onnx"

            [models.detector]
            file = "yunet.onnx"

            [models.embedder]
            file = "dinov2-base.onnx"
            "#,
        )
        .expect("valid config");

        assert_eq!(config.models_dir(), Path::new("/opt/af/assets/onnx"));
        assert_eq!(
            config.models()[1].path(),
            Path::new("/opt/af/assets/onnx/dinov2-base.onnx")
        );
    }

    #[test]
    fn should_keep_absolute_models_dir_when_config_gives_one() {
        let config = parse(
            r#"
            [models]
            dir = "/srv/models"

            [models.detector]
            file = "yunet.onnx"

            [models.embedder]
            file = "dinov2-large.onnx"
            "#,
        )
        .expect("valid config");

        assert_eq!(config.models_dir(), Path::new("/srv/models"));
    }

    #[test]
    fn should_use_the_configured_model_id_when_one_is_given() {
        let config = parse(
            r#"
            [models.detector]
            file = "yunet.onnx"
            id = "yunet-2023mar"

            [models.embedder]
            file = "model.onnx"
            id = "dinov2-vitl14"
            "#,
        )
        .expect("valid config");

        assert_eq!(config.models()[0].id().as_str(), "yunet-2023mar");
        assert_eq!(config.models()[1].id().as_str(), "dinov2-vitl14");
    }

    #[test]
    fn should_track_the_vit_size_when_the_embedder_file_changes() {
        let small = parse(MINIMAL).expect("valid config");
        let large = parse(
            r#"
            [models.detector]
            file = "face_detection_yunet_2023mar.onnx"

            [models.embedder]
            file = "dinov2-large.onnx"
            "#,
        )
        .expect("valid config");

        assert_ne!(small.models()[1].id(), large.models()[1].id());
    }

    #[test]
    fn should_reject_config_when_the_embedder_is_missing() {
        let error = parse(
            r#"
            [models.detector]
            file = "yunet.onnx"
            "#,
        )
        .expect_err("incomplete config");

        assert!(matches!(error, ConfigError::Invalid { .. }), "{error}");
    }

    #[test]
    fn should_reject_config_when_a_key_is_misspelled() {
        let error = parse(
            r#"
            [models.detector]
            path = "yunet.onnx"

            [models.embedder]
            file = "dinov2-small.onnx"
            "#,
        )
        .expect_err("misspelled key");

        assert!(matches!(error, ConfigError::Invalid { .. }), "{error}");
    }

    #[test]
    fn should_reject_config_when_a_model_id_is_blank() {
        let error = parse(
            r#"
            [models.detector]
            file = "yunet.onnx"
            id = "  "

            [models.embedder]
            file = "dinov2-small.onnx"
            "#,
        )
        .expect_err("blank id");

        assert!(matches!(error, ConfigError::Model(_)), "{error}");
    }

    #[test]
    fn should_use_the_env_var_when_it_names_a_path() {
        let path = config_path_from(Some("/etc/af/booth.toml".into()));

        assert_eq!(path, Path::new("/etc/af/booth.toml"));
    }

    #[test]
    fn should_fall_back_to_the_working_directory_when_the_env_var_is_unset_or_blank() {
        assert_eq!(config_path_from(None), Path::new(DEFAULT_CONFIG_FILE));
        assert_eq!(
            config_path_from(Some("".into())),
            Path::new(DEFAULT_CONFIG_FILE)
        );
    }

    #[test]
    fn should_report_not_found_when_the_config_file_is_absent() {
        let error =
            BoothConfig::load_from("/nonexistent/booth.toml").expect_err("no such config file");

        assert!(matches!(error, ConfigError::NotFound { .. }), "{error}");
        assert!(error.to_string().contains(CONFIG_ENV));
    }

    #[test]
    fn should_load_the_repository_config_when_asked_for_it() {
        // The shipped `booth.toml` is the file `cargo run -p afbooth` picks up;
        // if it stops parsing, the booth stops starting.
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(DEFAULT_CONFIG_FILE);

        let config = BoothConfig::load_from(&repo_root).expect("repository config parses");

        assert_eq!(config.models().len(), 2);
    }
}
