# The Corpus is one SQLite database beside a tree of image files

The Corpus stores each Face as a row in a bundled-SQLite database — capture time, model
identifier, embedding width, and the embedding itself as a blob of little-endian `f32`s —
and its three images as PNG files under `faces/<shard>/<id>/`. The database is the index;
the photographs are not in it. `rusqlite`'s `bundled` feature compiles the SQLite
amalgamation in, so an installation machine's own sqlite is never part of the piece
(ADR-0006 keeps the hardware target open).

Images stay out of the database because the renderer streams up to a thousand of them on
its own schedule and the layout stage wants every Embedding without touching one, while the
original full-quality frames — retained because re-embedding reads them back (ADR-0006) —
would otherwise dominate a file that has to stay backup-able. Embeddings stay *in* it
because they are read whole, at once, by the SOM: a row per dimension buys nothing and
brute-force cosine over a few thousand blobs is not a performance problem at this scale
(implementation plan, "Key technical choices").

The split has one consequence worth stating: the two halves can disagree. A database
restored from a backup while `faces/` survives will hand out an identifier whose images
already exist. Ingest refuses that Capture rather than overwriting, because the file it
would destroy is some Visitor's original frame.
