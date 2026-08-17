//! Migrations.
//!
//! The Corpus persists between installations and grows without bound
//! (`CONTEXT.md`), so its schema is versioned from the first commit rather than
//! retrofitted once there are Faces in it worth not losing.
//!
//! The version lives in SQLite's own `user_version` pragma: it is written
//! inside the same transaction as the statements it describes, so a migration
//! either lands with its version bump or not at all.

use rusqlite::Connection;

use crate::error::StoreError;

/// The schema version this build migrates to.
pub const SCHEMA_VERSION: u32 = 1;

/// Every migration, in order. Index `n` moves the schema from version `n` to
/// `n + 1`. Migrations are append-only: an existing entry is never edited,
/// because some installation's Corpus has already run it.
const MIGRATIONS: &[&str] = &[
    // 0 -> 1: Faces and their Embeddings.
    //
    // Embedding values are a blob of little-endian `f32`s rather than a table
    // of rows: they are read whole or not at all, and the layout stage loads
    // every one of them at once. `model_id` and `dim` are stored beside the
    // blob because changing models invalidates every Embedding (ADR-0006) and
    // the width follows the ViT size (ADR-0007) — neither may be inferred.
    //
    // A Face has no consent columns. Consent Records are Stage 4 and arrive as
    // their own table keyed on `face.id`, which this schema leaves room for.
    "CREATE TABLE face (
        id           INTEGER PRIMARY KEY,
        captured_at  INTEGER NOT NULL,
        model_id     TEXT    NOT NULL,
        dim          INTEGER NOT NULL,
        values_le    BLOB    NOT NULL
    );
    CREATE INDEX face_model_id ON face (model_id);",
];

/// Brings `connection` up to [`SCHEMA_VERSION`], and reports the version.
///
/// Idempotent: a database already at the current version runs no statements.
///
/// # Errors
///
/// Returns [`StoreError::SchemaTooNew`] when the database was written by a
/// newer build, and [`StoreError::Database`] when a migration fails — in which
/// case nothing from that migration is left behind.
pub(crate) fn migrate(connection: &mut Connection) -> Result<u32, StoreError> {
    let version: u32 = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;

    if version > SCHEMA_VERSION {
        return Err(StoreError::SchemaTooNew {
            found: version,
            understood: SCHEMA_VERSION,
        });
    }

    for (index, migration) in MIGRATIONS.iter().enumerate().skip(version as usize) {
        let next = index as u32 + 1;
        let transaction = connection.transaction()?;
        transaction.execute_batch(migration)?;
        // PRAGMA does not accept a bound parameter, and `next` is an integer
        // this module produced, not anything a caller supplied.
        transaction.pragma_update(None, "user_version", next)?;
        transaction.commit()?;
    }

    Ok(SCHEMA_VERSION)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_stamp_the_current_version_when_the_database_is_empty() {
        let mut connection = Connection::open_in_memory().expect("an in-memory database");

        let version = migrate(&mut connection).expect("migrations run");

        assert_eq!(version, SCHEMA_VERSION);
    }

    #[test]
    fn should_run_nothing_when_the_database_is_already_current() {
        let mut connection = Connection::open_in_memory().expect("an in-memory database");
        migrate(&mut connection).expect("migrations run");

        // A second pass over the same connection would fail on `CREATE TABLE`
        // if any migration ran twice.
        let version = migrate(&mut connection).expect("migrations are idempotent");

        assert_eq!(version, SCHEMA_VERSION);
    }

    #[test]
    fn should_refuse_when_the_database_is_from_a_newer_build() {
        let mut connection = Connection::open_in_memory().expect("an in-memory database");
        connection
            .pragma_update(None, "user_version", SCHEMA_VERSION + 1)
            .expect("the pragma is set");

        assert!(matches!(
            migrate(&mut connection),
            Err(StoreError::SchemaTooNew { .. })
        ));
    }

    #[test]
    fn should_have_a_migration_for_every_version() {
        assert_eq!(MIGRATIONS.len(), SCHEMA_VERSION as usize);
    }
}
