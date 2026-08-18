//! The booth's configuration file.
//!
//! Model paths and identifiers are configuration, not constants: which DINOv2
//! ViT size the piece runs follows the hardware decision, which is still open
//! (ADR-0006, ADR-0007). Nothing here knows that DINOv2 has sizes at all — it
//! knows only that a file name and an optional identifier arrive from a file.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use afcapture::CameraSelector;
use afcore::{GridSpec, GridSpecError};
use afvision::{DisplayCropError, DisplayCropSpec, ModelError, ModelRole, ModelSpec};
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

    /// The file parsed, but the display framing describes no rectangle.
    #[error(transparent)]
    DisplayCrop(#[from] DisplayCropError),

    /// The file parsed, but the Grid it asks for is not one the piece shows.
    #[error(transparent)]
    Grid(#[from] GridSpecError),
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

/// The `[display]` table: how the wall's crop is framed around a face.
///
/// Whether display crops are square or portrait, and how tightly framed, is an
/// open question in the implementation plan — it is meant to be settled by eye
/// on the wall. The defaults here are a starting point, not an answer.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct DisplayTable {
    /// Fraction of the detection box added on every side.
    #[serde(default = "default_margin")]
    margin: f32,
    /// Width over height: 1.0 is square, 0.8 is a 4:5 portrait.
    #[serde(default = "default_aspect_ratio")]
    aspect_ratio: f32,
    /// Rendered width in pixels.
    #[serde(default = "default_crop_width")]
    width: u32,
    /// Lifts the frame by this fraction of its own height.
    #[serde(default = "default_vertical_bias")]
    vertical_bias: f32,
}

impl Default for DisplayTable {
    fn default() -> Self {
        Self {
            margin: default_margin(),
            aspect_ratio: default_aspect_ratio(),
            width: default_crop_width(),
            vertical_bias: default_vertical_bias(),
        }
    }
}

// The house framing lives in `afvision`, with the code that applies it; an
// omitted key here means "whatever the piece's default framing is", not a
// number this file gets to choose independently.
fn default_margin() -> f32 {
    DisplayCropSpec::default().margin()
}

fn default_aspect_ratio() -> f32 {
    DisplayCropSpec::default().aspect_ratio()
}

fn default_crop_width() -> u32 {
    DisplayCropSpec::default().width()
}

fn default_vertical_bias() -> f32 {
    DisplayCropSpec::default().vertical_bias()
}

/// The `[corpus]` table: where the archive lives.
#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CorpusTable {
    /// Relative paths resolve against the configuration file's directory, so a
    /// booth and its archive can be moved together.
    #[serde(default = "default_corpus_dir")]
    dir: PathBuf,
}

impl Default for CorpusTable {
    fn default() -> Self {
        Self {
            dir: default_corpus_dir(),
        }
    }
}

fn default_corpus_dir() -> PathBuf {
    PathBuf::from("corpus")
}

/// The `[grid]` table: how many Cells the wall shows.
///
/// Fixed for the run. The Grid breathing between [`afcore::MIN_CELLS`] and
/// [`afcore::MAX_CELLS`] on the piece's own clock is Stage 3 (ADR-0004).
#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct GridTable {
    #[serde(default = "default_cols")]
    cols: u32,
    #[serde(default = "default_rows")]
    rows: u32,
}

impl Default for GridTable {
    fn default() -> Self {
        Self {
            cols: default_cols(),
            rows: default_rows(),
        }
    }
}

// Twenty Cells: above `afcore::MIN_CELLS`, and small enough that a booth on an
// unknown machine comes up without asking it for a thousand textures.
fn default_cols() -> u32 {
    5
}

fn default_rows() -> u32 {
    4
}

/// The `[camera]` table: which device the Shutter opens.
#[derive(Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CameraTable {
    /// Device index as the platform enumerates them; omitted means the
    /// platform's first camera.
    #[serde(default)]
    index: Option<u32>,
}

/// The configuration file as written on disk.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ConfigFile {
    models: ModelsTable,
    #[serde(default)]
    display: DisplayTable,
    #[serde(default)]
    corpus: CorpusTable,
    #[serde(default)]
    grid: GridTable,
    #[serde(default)]
    camera: CameraTable,
}

/// The booth's resolved configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct BoothConfig {
    source: PathBuf,
    models_dir: PathBuf,
    corpus_dir: PathBuf,
    detector: ModelSpec,
    embedder: ModelSpec,
    display_crop: DisplayCropSpec,
    grid: GridSpec,
    camera: CameraSelector,
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
        let models_dir = beside(base, &file.models.dir);
        let corpus_dir = beside(base, &file.corpus.dir);
        let grid = GridSpec::new(file.grid.cols, file.grid.rows)?;
        let camera = match file.camera.index {
            Some(index) => CameraSelector::Index(index),
            None => CameraSelector::Default,
        };

        let detector = spec(ModelRole::Detector, &models_dir, &file.models.detector)?;
        let embedder = spec(ModelRole::Embedder, &models_dir, &file.models.embedder)?;
        let display_crop = DisplayCropSpec::new(
            file.display.margin,
            file.display.aspect_ratio,
            file.display.width,
        )?
        .with_vertical_bias(file.display.vertical_bias);

        Ok(Self {
            source: source.to_path_buf(),
            models_dir,
            corpus_dir,
            detector,
            embedder,
            display_crop,
            grid,
            camera,
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

    /// How the wall's crop is framed around a detected face.
    pub fn display_crop(&self) -> &DisplayCropSpec {
        &self.display_crop
    }

    /// Where the Corpus lives.
    pub fn corpus_dir(&self) -> &Path {
        &self.corpus_dir
    }

    /// The Grid the wall shows.
    pub fn grid(&self) -> GridSpec {
        self.grid
    }

    /// Which camera the Shutter opens.
    pub fn camera(&self) -> &CameraSelector {
        &self.camera
    }

    /// Every model the booth needs, in self-check order.
    pub fn models(&self) -> [&ModelSpec; 2] {
        [&self.detector, &self.embedder]
    }

    /// The detector, for building a Session.
    pub fn detector(&self) -> &ModelSpec {
        &self.detector
    }

    /// The embedder, for building a Session.
    pub fn embedder(&self) -> &ModelSpec {
        &self.embedder
    }
}

/// Resolves a configured directory, which may be absolute, against the
/// configuration file's own directory.
fn beside(base: &Path, dir: &Path) -> PathBuf {
    if dir.is_absolute() {
        dir.to_path_buf()
    } else {
        base.join(dir)
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
    fn should_frame_a_portrait_display_crop_when_config_omits_the_table() {
        let config = parse(MINIMAL).expect("valid config");
        let crop = config.display_crop();

        assert_eq!((crop.width(), crop.height()), (512, 640));
        assert!(
            crop.margin() > 0.0,
            "the display crop must be looser than the detection box"
        );
    }

    #[test]
    fn should_use_the_configured_framing_when_the_display_table_is_present() {
        let config = parse(
            r#"
            [display]
            margin = 0.6
            aspect_ratio = 1.0
            width = 256
            vertical_bias = 0.1

            [models.detector]
            file = "yunet.onnx"

            [models.embedder]
            file = "dinov2-small.onnx"
            "#,
        )
        .expect("valid config");
        let crop = config.display_crop();

        assert_eq!((crop.width(), crop.height()), (256, 256));
        assert!((crop.margin() - 0.6).abs() < f32::EPSILON);
        assert!((crop.vertical_bias() - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn should_reject_config_when_the_display_framing_is_impossible() {
        let error = parse(
            r#"
            [display]
            aspect_ratio = 0.0

            [models.detector]
            file = "yunet.onnx"

            [models.embedder]
            file = "dinov2-small.onnx"
            "#,
        )
        .expect_err("impossible framing");

        assert!(matches!(error, ConfigError::DisplayCrop(_)), "{error}");
    }

    #[test]
    fn should_put_the_corpus_beside_the_config_when_it_names_no_directory() {
        let config = parse(MINIMAL).expect("valid config");

        assert_eq!(config.corpus_dir(), Path::new("/opt/af/corpus"));
    }

    #[test]
    fn should_keep_an_absolute_corpus_directory_when_config_gives_one() {
        let config =
            parse(&format!("{MINIMAL}\n[corpus]\ndir = \"/srv/faces\"\n")).expect("valid config");

        assert_eq!(config.corpus_dir(), Path::new("/srv/faces"));
    }

    #[test]
    fn should_show_the_configured_grid_when_the_grid_table_is_present() {
        let config =
            parse(&format!("{MINIMAL}\n[grid]\ncols = 8\nrows = 5\n")).expect("valid config");

        assert_eq!(config.grid().cell_count(), 40);
    }

    #[test]
    fn should_reject_config_when_the_grid_is_outside_the_piece_s_range() {
        // A Grid the piece will not show is a configuration error at startup,
        // not a renderer failure after the window opens (ADR-0004).
        let error = parse(&format!("{MINIMAL}\n[grid]\ncols = 1\nrows = 1\n"))
            .expect_err("a grid below MIN_CELLS");

        assert!(matches!(error, ConfigError::Grid(_)), "{error}");
    }

    #[test]
    fn should_open_the_default_camera_when_config_names_no_index() {
        let config = parse(MINIMAL).expect("valid config");

        assert_eq!(config.camera(), &CameraSelector::Default);
    }

    #[test]
    fn should_open_the_configured_camera_when_config_names_an_index() {
        let config = parse(&format!("{MINIMAL}\n[camera]\nindex = 2\n")).expect("valid config");

        assert_eq!(config.camera(), &CameraSelector::Index(2));
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
