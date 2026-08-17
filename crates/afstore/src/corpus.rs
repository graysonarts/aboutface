//! The Corpus: the database, the image files, and the path between them.

use std::path::{Path, PathBuf};

use afcore::{Embedding, FaceId, ModelId};
use image::RgbImage;
use rusqlite::Connection;

use crate::error::StoreError;
use crate::face::{NewFace, StoredFace, from_unix_millis, to_unix_millis};
use crate::paths::{ImageKind, face_dir, in_dir};
use crate::{blob, schema};

/// The database file inside a Corpus directory.
const DATABASE_FILE: &str = "corpus.db";

/// A `FaceId` as SQLite stores it.
///
/// SQLite has no unsigned integer: a rowid is an `i64`, always positive, and
/// the round trip through `u64` is why every identifier crossing this boundary
/// goes through these two functions rather than an inline cast.
fn rowid(face: FaceId) -> i64 {
    face.0 as i64
}

/// A rowid as a `FaceId`.
fn face_id(rowid: i64) -> FaceId {
    FaceId(rowid as u64)
}

/// How far a stored Embedding's magnitude may sit from 1.0 and still be read.
///
/// Wide enough for the rounding a few thousand `f32` squares accumulate,
/// narrow enough that a blob holding something other than an Embedding is
/// refused rather than quietly rescaled.
const UNIT_TOLERANCE: f32 = 1e-3;

/// Every Face ever retained, on one machine's disk.
///
/// A Corpus is a directory: one SQLite database holding Faces and their
/// Embeddings, and a tree of image files holding the crops and the original
/// frames. The database is the index; the images are too large to want inside
/// it and the renderer reads them as files.
///
/// The Corpus never leaves the installation machine.
pub struct Corpus {
    root: PathBuf,
    connection: Connection,
    schema_version: u32,
}

impl Corpus {
    /// Opens the Corpus rooted at `root`, creating and migrating it if needed.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created, the database
    /// cannot be opened, or the schema is newer than this build understands.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, StoreError> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root).map_err(|source| StoreError::file(&root, source))?;

        let mut connection = Connection::open(root.join(DATABASE_FILE))?;
        // A half-written Face is worse than a refused Capture, and the wall
        // reads while the booth writes.
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "foreign_keys", true)?;

        let schema_version = schema::migrate(&mut connection)?;

        Ok(Self {
            root,
            connection,
            schema_version,
        })
    }

    /// The directory this Corpus lives in.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The schema version the database is at.
    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Writes one Capture into the Corpus and reports its [`FaceId`].
    ///
    /// All of it or none of it: the row is inserted inside a transaction, the
    /// three images are written while it is open, and the transaction commits
    /// only once every file is on disk. A failure part-way rolls the row back
    /// and removes whatever was written, so the Corpus never holds a Face with
    /// a missing photograph.
    ///
    /// # Errors
    ///
    /// Returns an error if the database refuses the row, an image cannot be
    /// written, or the identifier's directory is already occupied — see
    /// [`StoreError::FaceDirOccupied`], which is refused rather than
    /// overwritten.
    pub fn ingest(&mut self, face: NewFace) -> Result<FaceId, StoreError> {
        let transaction = self.connection.transaction()?;
        let embedding = face.embedding();

        transaction.execute(
            "INSERT INTO face (captured_at, model_id, dim, values_le)
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![
                to_unix_millis(face.captured_at()),
                embedding.model().as_str(),
                embedding.dim() as i64,
                blob::encode(embedding.as_slice()),
            ],
        )?;
        // INVARIANT: SQLite assigns a positive rowid to the row just inserted.
        let id = face_id(transaction.last_insert_rowid());
        let dir = face_dir(&self.root, id);

        match write_images(&dir, &face).and_then(|()| Ok(transaction.commit()?)) {
            Ok(()) => Ok(id),
            Err(error) => {
                // Dropping the transaction rolls the row back and frees the
                // rowid for the next Capture, so this attempt's images go with
                // it: a directory left behind would refuse that next Capture,
                // and images with no row are not a Face anyone can delete.
                //
                // Except when the directory was already occupied — those images
                // are somebody else's, which is exactly why the write refused.
                if !matches!(error, StoreError::FaceDirOccupied { .. }) {
                    let _ = std::fs::remove_dir_all(&dir);
                }
                Err(error)
            }
        }
    }

    /// Reads one Face back, or `None` if the Corpus does not hold it.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be read, or if the stored
    /// Embedding disagrees with its recorded width or model.
    pub fn face(&self, id: FaceId) -> Result<Option<StoredFace>, StoreError> {
        let mut statement = self
            .connection
            .prepare("SELECT id, captured_at, model_id, dim, values_le FROM face WHERE id = ?1")?;
        let mut rows = statement.query(rusqlite::params![rowid(id)])?;

        match rows.next()? {
            Some(row) => self.stored_face(row).map(Some),
            None => Ok(None),
        }
    }

    /// Every Embedding produced by `model`, with the Face it belongs to.
    ///
    /// This is the query the layout stage runs: it orders the Corpus by
    /// Embedding and never needs an image to do it. Embeddings from other
    /// models are not returned — they describe a different space, and
    /// `afcore` refuses to compare across models rather than returning a
    /// degraded number (ADR-0006).
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be read, or if a stored
    /// Embedding disagrees with its recorded width.
    pub fn embeddings_for_model(
        &self,
        model: &ModelId,
    ) -> Result<Vec<(FaceId, Embedding)>, StoreError> {
        let mut statement = self
            .connection
            .prepare("SELECT id, dim, values_le FROM face WHERE model_id = ?1 ORDER BY id")?;
        let mut rows = statement.query(rusqlite::params![model.as_str()])?;

        let mut embeddings = Vec::new();
        while let Some(row) = rows.next()? {
            let id = face_id(row.get(0)?);
            let dim: i64 = row.get(1)?;
            let values: Vec<u8> = row.get(2)?;
            embeddings.push((id, decode_embedding(id, model.clone(), dim, &values)?));
        }

        Ok(embeddings)
    }

    /// How many Faces the Corpus holds.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be read.
    pub fn count(&self) -> Result<u64, StoreError> {
        let count: i64 = self
            .connection
            .query_row("SELECT COUNT(*) FROM face", [], |row| row.get(0))?;
        // INVARIANT: a COUNT is never negative.
        Ok(count.unsigned_abs())
    }

    fn stored_face(&self, row: &rusqlite::Row<'_>) -> Result<StoredFace, StoreError> {
        let id = face_id(row.get(0)?);
        let captured_at = from_unix_millis(row.get(1)?);
        let model = ModelId::new(row.get::<_, String>(2)?);
        let dim: i64 = row.get(3)?;
        let values: Vec<u8> = row.get(4)?;

        Ok(StoredFace::new(
            id,
            decode_embedding(id, model, dim, &values)?,
            captured_at,
            face_dir(&self.root, id),
        ))
    }
}

/// Writes the three images of `face` into `dir`.
///
/// An occupied directory is refused rather than cleared. SQLite hands out
/// `max(rowid) + 1`, so a Corpus whose database was restored from a backup
/// while `faces/` survived would offer an identifier whose images already
/// exist — and those images belong to a Visitor who is still in the archive.
fn write_images(dir: &Path, face: &NewFace) -> Result<(), StoreError> {
    if dir.exists() {
        return Err(StoreError::FaceDirOccupied {
            path: dir.to_path_buf(),
        });
    }
    std::fs::create_dir_all(dir).map_err(|source| StoreError::file(dir, source))?;

    for (kind, image) in [
        (ImageKind::Original, face.original()),
        (ImageKind::Aligned, face.aligned()),
        (ImageKind::Display, face.display()),
    ] {
        write_png(&in_dir(dir, kind), image)?;
    }

    Ok(())
}

/// Encodes one image losslessly at `path`.
fn write_png(path: &Path, image: &RgbImage) -> Result<(), StoreError> {
    image
        .save_with_format(path, image::ImageFormat::Png)
        .map_err(|source| StoreError::image(path, source))
}

/// Rebuilds an Embedding from its stored parts, refusing a row that no longer
/// describes one.
fn decode_embedding(
    id: FaceId,
    model: ModelId,
    dim: i64,
    values: &[u8],
) -> Result<Embedding, StoreError> {
    let dim = usize::try_from(dim).map_err(|_| StoreError::Corrupt {
        face: id,
        detail: format!("recorded width {dim} is not a width"),
    })?;

    let values = blob::decode(values, dim).ok_or_else(|| StoreError::Corrupt {
        face: id,
        detail: format!(
            "embedding blob is {} bytes, expected {} for width {dim}",
            values.len(),
            dim * size_of::<f32>()
        ),
    })?;

    // The values went in already normalized, so they come back out unchanged
    // rather than through a second division by a magnitude that is only
    // approximately 1.0 in `f32`. A Face read back is the Face that was
    // written; the tolerance catches a corrupt blob, it does not rescale one.
    Embedding::from_unit(model, values, UNIT_TOLERANCE).map_err(|source| StoreError::Corrupt {
        face: id,
        detail: source.to_string(),
    })
}
