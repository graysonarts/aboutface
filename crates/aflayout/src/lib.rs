//! Ordering and placement.
//!
//! Layout is two separable stages (ADR-0003):
//!
//! 1. **Ordering** — a self-organizing map with rectangular lattice topology,
//!    trained on the Corpus's Embeddings. Its Nodes give the Grid a stable,
//!    spatially coherent arrangement, and unlike UMAP it updates incrementally
//!    as Faces arrive rather than being re-fit from scratch. Nodes are not
//!    Cells, and a Node is not a person.
//!
//! 2. **Placement** — the Window's Faces are assigned bijectively to Cells by
//!    solving a linear assignment problem (LAPJV), which runs in tens of
//!    milliseconds across the whole supported Grid range.
//!
//! The assignment cost blends similarity with a movement penalty:
//!
//! ```text
//! cost(f, c) = cosine_distance(embedding(f), prototype(node_at(c)))
//!            + λ · movement_penalty(current_cell(f), c)
//! ```
//!
//! The movement penalty is essential, not an optimization. Without it an
//! optimal-by-similarity solution will teleport a Face across the wall to save a
//! trivial distance, and the animation reads as chaos rather than as the crowd
//! reorganizing. λ is an artistic dial, expected to be tuned by eye.
//!
//! This crate also resamples the Node lattice when the Grid resizes, rather than
//! retraining, so that a resize looks like a breath and not a glitch (ADR-0004).

#![allow(dead_code, unused_imports)] // Stage 0 scaffold.
