# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**About:Face** is an art project (early development): a booth photographs consenting visitors and displays them alongside similar-looking people, on a wall that rearranges itself so people who look alike sit near each other.

## Current state: Stage 1 in progress

A Cargo workspace of seven crates, CI, and the domain types in `afcore`. `afcapture` grabs frames behind the `Camera` trait; `afvision` runs the whole `detect -> align -> embed` path off files on disk; `afstore` holds a migrated Corpus that ingests a Face and reads it back. **Nothing lays out or renders yet**, and `afbooth` still prints its configuration and exits.

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
./scripts/fetch-models.sh                             # ONNX files; not committed
cargo run -p afbooth                                  # startup self-check
cargo run -p afvision --example detect_face -- samples/1.jpg   # detect + both crops
cargo run -p afvision --example embed_faces -- samples/1.jpg samples/3.jpg  # detect -> align -> embed
```

`cargo run -p afbooth` reads `booth.toml` and reports each model's path, presence and `ModelId` plus the ONNX Runtime execution provider selected. Missing model files exit non-zero with the fetch command. Where the weights come from: [`docs/models.md`](docs/models.md).

The original C++/OpenCV "annotator" tools (a 2015 prototype of face *detection*) were deleted on 2026-08-17 per [ADR-0001](docs/adr/0001-retire-opencv-annotators-for-learned-embeddings.md). They remain in git history — `git log -- annotators/` — and are not a reference implementation. Do not resurrect them.

## Crates

| Crate | Owns |
| --- | --- |
| `afcore` | Domain types. No I/O, no GPU, no inference. Everything depends on it; it depends on `thiserror`. |
| `afvision` | Detect, align, embed. Owns the `ModelId` and the execution-provider choice. |
| `afstore` | The Corpus: Faces, Embeddings, Consent Records, deletion. |
| `aflayout` | SOM ordering, LAPJV placement, lattice resampling on resize. |
| `afcapture` | `Camera` trait and Shutter. **The only crate allowed to import a camera API.** |
| `afrender` | wgpu + winit. Re-solve and Drift are visually distinct motions. |
| `afbooth` | The binary: wiring, config, startup self-check. |

Two invariants are already encoded in `afcore` and should stay that way: Embeddings from different models or widths **refuse** to compare rather than returning a degraded number, and an `Assignment` refuses to place one Face in two Cells.

## Read before doing design work

- **[`CONTEXT.md`](CONTEXT.md)** — the project's vocabulary. Use these terms exactly; the glossary also lists terms deliberately avoided.
- **[`docs/adr/`](docs/adr/)** — the decisions and their reasoning. These win over any other document in a conflict.
- **[`docs/implementation-plan.md`](docs/implementation-plan.md)** — the staged plan and the intended crate layout.
- **[`docs/models.md`](docs/models.md)** — where the ONNX weights come from, their licences, and how `booth.toml` points at them.

## Direction in brief

Everything below is settled in the ADRs; this is a summary, not the source of truth.

- **All Rust**, one process, a Cargo workspace of seven crates. No openFrameworks — there never was any in this project, and the renderer is `wgpu` + `winit` (ADR-0002).
- **Similarity is a learned embedding** of an aligned crop, cosine distance, behind a YuNet (MIT) detector that emits the five landmarks needed for alignment (ADR-0001). The model is **DINOv2 (Apache 2.0)**, and the piece is committed to *apparent* resemblance rather than identity resemblance. Face-recognition weights are dropped — every mainstream set is research-only and the piece is expected to be commercial (ADR-0007). Embedding width is not fixed; it follows the ViT size.
- **Layout is two stages**: a self-organizing map supplies ordering, and an LAPJV linear assignment places one Face per Cell with a movement penalty so re-solves animate coherently (ADR-0003).
- **The Grid is a Window** onto a much larger persistent Corpus, sized on the piece's own clock and drifting slowly across the archive (ADR-0004).
- **Capture is opt-in** and the Corpus persists, so Consent Records, Receipt Codes, and a tested deletion path are components with tickets (ADR-0005).
- **Hardware is deferred**, so camera access sits behind a trait and the CPU inference path must stay viable. The model identifier is part of the store schema because changing models invalidates every Embedding (ADR-0006).

## Next actionable work

Stage 1 in the implementation plan: a fixed grid of stored Faces on screen in `afrender`, then the `afcapture -> afvision -> afstore -> afrender` wiring in `afbooth`. `afstore`'s schema and write path is done — `Corpus::ingest` takes an Embedding, the two crops and the original frame; `embeddings_for_model` is what the layout stage will read. The Stage 1 sanity check on what the Embedding is actually keyed on (ADR-0007) needs a camera and is still outstanding.

## `samples/`

Four face photographs kept as test fixtures for the new pipeline. They were sample inputs for the deleted C++ tools; the CMake install rules that referenced them are gone.

## Agent skills

The `mattpocock-skills` plugin is enabled for this repo via `.claude/settings.json`. Note that upstream renamed `/grill-me` to `/grill-with-docs`.

### Issue tracker

Issues live in this repo's GitHub Issues (`graysonarts/aboutface`), driven by the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, each using its own name as the label string. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and one `docs/adr/` at the repo root, both now populated. See `docs/agents/domain.md`.
