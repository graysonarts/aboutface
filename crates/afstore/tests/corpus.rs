//! The Corpus over a real SQLite database in a temporary directory.
//!
//! An integration test rather than a unit test because it crosses a system
//! boundary the crate does not own: SQLite and the filesystem. Nothing here is
//! mocked — the Corpus is the artwork's memory, and "it worked against a fake"
//! is not evidence that a Visitor's Face survives a restart.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use afcore::{Embedding, ModelId};
use afstore::{Corpus, NewFace};
use image::RgbImage;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// A `samples/` photograph, standing in for the frame a camera hands over.
fn sample_photograph() -> RgbImage {
    image::open(repo_root().join("samples").join("1.jpg"))
        .expect("the sample photograph decodes")
        .to_rgb8()
}

fn model() -> ModelId {
    ModelId::new("dinov2-small")
}

/// An Embedding with a direction that depends on `seed`, so two fixtures are
/// distinguishable without running a model.
fn embedding(seed: f32) -> Embedding {
    Embedding::new(model(), vec![seed, 1.0, 0.5, -2.0]).expect("valid embedding")
}

fn solid(width: u32, height: u32, colour: [u8; 3]) -> RgbImage {
    RgbImage::from_pixel(width, height, image::Rgb(colour))
}

/// One Capture's worth of material: the sample photograph as the original
/// frame, with the two crops and the Embedding that would have come off it.
fn new_face(seed: f32) -> NewFace {
    NewFace::new(
        embedding(seed),
        solid(112, 112, [10, 20, 30]),
        solid(512, 640, [40, 50, 60]),
        sample_photograph(),
        UNIX_EPOCH + Duration::from_millis(1_755_000_000_000),
    )
}

#[test]
fn should_run_migrations_when_the_database_is_empty() {
    let root = tempfile::tempdir().expect("a temporary directory");

    let corpus = Corpus::open(root.path()).expect("the corpus opens");

    assert_eq!(corpus.schema_version(), afstore::SCHEMA_VERSION);
}

#[test]
fn should_leave_the_schema_alone_when_the_corpus_is_reopened() {
    let root = tempfile::tempdir().expect("a temporary directory");
    drop(Corpus::open(root.path()).expect("the corpus opens"));

    let reopened = Corpus::open(root.path()).expect("the corpus reopens");

    assert_eq!(reopened.schema_version(), afstore::SCHEMA_VERSION);
}

#[test]
fn should_write_a_face_and_its_three_images_when_a_photograph_is_ingested() {
    let root = tempfile::tempdir().expect("a temporary directory");
    let mut corpus = Corpus::open(root.path()).expect("the corpus opens");
    let original = sample_photograph();

    let id = corpus.ingest(new_face(1.0)).expect("the face is ingested");

    let face = corpus
        .face(id)
        .expect("the face reads back")
        .expect("the face is there");
    assert_eq!(corpus.count().expect("the corpus counts"), 1);
    assert_eq!(face.id(), id);
    assert_eq!(
        image::open(face.original_path())
            .expect("the original decodes")
            .to_rgb8()
            .dimensions(),
        original.dimensions()
    );
    assert_eq!(
        image::open(face.aligned_path())
            .expect("the aligned crop decodes")
            .to_rgb8()
            .dimensions(),
        (112, 112)
    );
    assert_eq!(
        image::open(face.display_path())
            .expect("the display crop decodes")
            .to_rgb8()
            .dimensions(),
        (512, 640)
    );
}

#[test]
fn should_return_a_comparable_embedding_when_a_face_is_read_back() {
    let root = tempfile::tempdir().expect("a temporary directory");
    let mut corpus = Corpus::open(root.path()).expect("the corpus opens");
    let written = embedding(1.0);

    let id = corpus.ingest(new_face(1.0)).expect("the face is ingested");

    let read = corpus
        .face(id)
        .expect("the face reads back")
        .expect("the face is there")
        .embedding()
        .clone();
    assert_eq!(read.model(), written.model());
    assert_eq!(read.dim(), written.dim());
    assert!(
        read.cosine_distance(&written).expect("comparable") < 1e-6,
        "the stored embedding pointed somewhere else"
    );
}

#[test]
fn should_keep_the_face_when_the_corpus_is_closed_and_reopened() {
    let root = tempfile::tempdir().expect("a temporary directory");
    let id = {
        let mut corpus = Corpus::open(root.path()).expect("the corpus opens");
        corpus.ingest(new_face(1.0)).expect("the face is ingested")
    };

    let corpus = Corpus::open(root.path()).expect("the corpus reopens");

    let face = corpus
        .face(id)
        .expect("the face reads back")
        .expect("the face survived the restart");
    assert!(face.original_path().exists(), "the original frame is gone");
}

#[test]
fn should_return_nothing_when_the_corpus_does_not_hold_that_face() {
    let root = tempfile::tempdir().expect("a temporary directory");
    let corpus = Corpus::open(root.path()).expect("the corpus opens");

    let face = corpus.face(afcore::FaceId(404)).expect("the query runs");

    assert!(face.is_none());
}

#[test]
fn should_return_only_that_model_when_embeddings_are_loaded_for_layout() {
    let root = tempfile::tempdir().expect("a temporary directory");
    let mut corpus = Corpus::open(root.path()).expect("the corpus opens");
    let first = corpus.ingest(new_face(1.0)).expect("the face is ingested");
    let second = corpus.ingest(new_face(2.0)).expect("the face is ingested");
    let other_model = NewFace::new(
        Embedding::new(ModelId::new("dinov2-large"), vec![1.0, 0.0, 0.0, 0.0]).expect("valid"),
        solid(112, 112, [0, 0, 0]),
        solid(512, 640, [0, 0, 0]),
        solid(64, 64, [0, 0, 0]),
        SystemTime::UNIX_EPOCH,
    );
    corpus.ingest(other_model).expect("the face is ingested");

    let loaded = corpus
        .embeddings_for_model(&model())
        .expect("the embeddings load");

    assert_eq!(
        loaded.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        vec![first, second]
    );
    assert!(loaded.iter().all(|(_, e)| e.model() == &model()));
}

#[test]
fn should_return_faces_in_capture_order_when_a_window_is_read() {
    let root = tempfile::tempdir().expect("a temporary directory");
    let mut corpus = Corpus::open(root.path()).expect("the corpus opens");
    let first = corpus.ingest(new_face(1.0)).expect("the face is ingested");
    let second = corpus.ingest(new_face(2.0)).expect("the face is ingested");

    let window = corpus.window(0, 10).expect("the window reads");

    assert_eq!(
        window.iter().map(|face| face.id()).collect::<Vec<_>>(),
        vec![first, second]
    );
    assert!(
        window[0].display_path().is_file(),
        "the wall needs the display crop the window points at"
    );
}

#[test]
fn should_return_only_as_many_faces_as_the_grid_has_cells_when_a_window_is_read() {
    let root = tempfile::tempdir().expect("a temporary directory");
    let mut corpus = Corpus::open(root.path()).expect("the corpus opens");
    for seed in 0..5 {
        corpus
            .ingest(new_face(seed as f32))
            .expect("the face is ingested");
    }

    let window = corpus.window(0, 2).expect("the window reads");

    assert_eq!(window.len(), 2);
}

#[test]
fn should_move_across_the_corpus_when_a_window_is_read_at_an_offset() {
    let root = tempfile::tempdir().expect("a temporary directory");
    let mut corpus = Corpus::open(root.path()).expect("the corpus opens");
    let ids: Vec<_> = (0..4)
        .map(|seed| {
            corpus
                .ingest(new_face(seed as f32))
                .expect("the face is ingested")
        })
        .collect();

    let window = corpus.window(2, 2).expect("the window reads");

    assert_eq!(
        window.iter().map(|face| face.id()).collect::<Vec<_>>(),
        ids[2..].to_vec()
    );
}

#[test]
fn should_return_nothing_when_a_window_is_read_past_the_end_of_the_corpus() {
    let root = tempfile::tempdir().expect("a temporary directory");
    let mut corpus = Corpus::open(root.path()).expect("the corpus opens");
    corpus.ingest(new_face(1.0)).expect("the face is ingested");

    let window = corpus.window(10, 10).expect("the window reads");

    assert!(window.is_empty());
}

#[test]
fn should_leave_no_face_behind_when_ingest_fails_part_way() {
    let root = tempfile::tempdir().expect("a temporary directory");
    let mut corpus = Corpus::open(root.path()).expect("the corpus opens");
    // The first Face's image directory, occupied by a file, so writing its
    // images fails after the row has been inserted.
    let blocked = root.path().join("faces").join("0000");
    std::fs::create_dir_all(&blocked).expect("the shard directory is made");
    std::fs::write(blocked.join("000001"), b"in the way").expect("the blocker is written");

    let failure = corpus.ingest(new_face(1.0));

    assert!(failure.is_err(), "ingest reported success: {failure:?}");
    assert_eq!(corpus.count().expect("the corpus counts"), 0);
    assert_eq!(
        std::fs::read(blocked.join("000001")).expect("the blocker is still there"),
        b"in the way"
    );
}

#[test]
fn should_store_the_original_frame_pixel_for_pixel() {
    // Re-embedding after a model change reads the originals back (ADR-0006),
    // so the stored frame is the frame, not a re-encoded likeness of it.
    let root = tempfile::tempdir().expect("a temporary directory");
    let mut corpus = Corpus::open(root.path()).expect("the corpus opens");
    let original = sample_photograph();

    let id = corpus.ingest(new_face(1.0)).expect("the face is ingested");

    let face = corpus
        .face(id)
        .expect("the face reads back")
        .expect("the face is there");
    let stored = image::open(face.original_path())
        .expect("the original decodes")
        .to_rgb8();
    assert_eq!(stored, original);
}

#[test]
fn should_refuse_when_the_face_directory_is_already_occupied() {
    // SQLite hands out max(rowid) + 1, so a database restored from a backup
    // while `faces/` survived would point at another Visitor's images. Refuse
    // rather than overwrite: the original frame is what a re-embed reads back
    // (ADR-0006).
    let root = tempfile::tempdir().expect("a temporary directory");
    let mut corpus = Corpus::open(root.path()).expect("the corpus opens");
    let dir = root.path().join("faces").join("0000").join("000001");
    std::fs::create_dir_all(&dir).expect("the face directory is made");
    std::fs::write(dir.join("original.png"), b"someone else's frame").expect("written");

    let failure = corpus.ingest(new_face(1.0));

    assert!(failure.is_err(), "ingest overwrote a face: {failure:?}");
    assert_eq!(corpus.count().expect("the corpus counts"), 0);
    assert_eq!(
        std::fs::read(dir.join("original.png")).expect("the frame is still there"),
        b"someone else's frame"
    );
}

#[test]
fn should_leave_no_images_behind_when_a_face_directory_is_rolled_back() {
    let root = tempfile::tempdir().expect("a temporary directory");
    let mut corpus = Corpus::open(root.path()).expect("the corpus opens");
    // A display crop that cannot be encoded: a zero-width image is not a PNG.
    let unwritable = NewFace::new(
        embedding(1.0),
        solid(112, 112, [0, 0, 0]),
        RgbImage::new(0, 0),
        sample_photograph(),
        SystemTime::UNIX_EPOCH,
    );

    let failure = corpus.ingest(unwritable);

    assert!(failure.is_err(), "ingest reported success: {failure:?}");
    assert_eq!(corpus.count().expect("the corpus counts"), 0);
    assert!(
        !root
            .path()
            .join("faces")
            .join("0000")
            .join("000001")
            .exists(),
        "a rolled-back face left its directory behind"
    );
    // The freed rowid is usable again, which it would not be if the directory
    // had survived.
    corpus
        .ingest(new_face(1.0))
        .expect("the next face is ingested");
}

#[test]
fn should_return_the_identical_embedding_when_a_face_is_read_back() {
    // Not "close enough": the Corpus is the artwork's memory, and a value that
    // shifts a little on every read is an archive that drifts.
    let root = tempfile::tempdir().expect("a temporary directory");
    let mut corpus = Corpus::open(root.path()).expect("the corpus opens");
    let written = embedding(1.0);

    let id = corpus.ingest(new_face(1.0)).expect("the face is ingested");

    let read = corpus
        .face(id)
        .expect("the face reads back")
        .expect("the face is there");
    assert_eq!(read.embedding(), &written);
}

#[test]
fn should_report_corruption_when_a_stored_embedding_disagrees_with_its_width() {
    let root = tempfile::tempdir().expect("a temporary directory");
    let id = {
        let mut corpus = Corpus::open(root.path()).expect("the corpus opens");
        corpus.ingest(new_face(1.0)).expect("the face is ingested")
    };
    let database = rusqlite::Connection::open(root.path().join("corpus.db")).expect("opens");
    database
        .execute("UPDATE face SET dim = dim + 1", [])
        .expect("the row is tampered with");
    drop(database);

    let corpus = Corpus::open(root.path()).expect("the corpus reopens");

    assert!(matches!(
        corpus.face(id),
        Err(afstore::StoreError::Corrupt { .. })
    ));
}

#[test]
fn should_refuse_to_open_when_the_corpus_is_from_a_newer_build() {
    let root = tempfile::tempdir().expect("a temporary directory");
    drop(Corpus::open(root.path()).expect("the corpus opens"));
    let database = rusqlite::Connection::open(root.path().join("corpus.db")).expect("opens");
    database
        .pragma_update(None, "user_version", afstore::SCHEMA_VERSION + 1)
        .expect("the version is bumped");
    drop(database);

    assert!(matches!(
        Corpus::open(root.path()),
        Err(afstore::StoreError::SchemaTooNew { .. })
    ));
}
