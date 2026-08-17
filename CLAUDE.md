# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**About:Face** is an art project (early development): a booth photographs consenting visitors and displays them alongside similar-looking people, on a wall that rearranges itself so people who look alike sit near each other.

## Current state: no implementation

The repository holds **documentation and sample images only**. There is no build, no code, and no test suite.

The original C++/OpenCV "annotator" tools (a 2015 prototype of face *detection*) were deleted on 2026-08-17 per [ADR-0001](docs/adr/0001-retire-opencv-annotators-for-learned-embeddings.md). They remain in git history — `git log -- annotators/` — and are not a reference implementation. Do not resurrect them.

## Read before doing design work

- **[`CONTEXT.md`](CONTEXT.md)** — the project's vocabulary. Use these terms exactly; the glossary also lists terms deliberately avoided.
- **[`docs/adr/`](docs/adr/)** — the decisions and their reasoning. These win over any other document in a conflict.
- **[`docs/implementation-plan.md`](docs/implementation-plan.md)** — the staged plan and the intended crate layout.

## Direction in brief

Everything below is settled in the ADRs; this is a summary, not the source of truth.

- **All Rust**, one process, a Cargo workspace of seven crates. No openFrameworks — there never was any in this project, and the renderer is `wgpu` + `winit` (ADR-0002).
- **Similarity is a learned embedding** of an aligned crop, cosine distance, behind a YuNet (MIT) detector that emits the five landmarks needed for alignment (ADR-0001). The embedding model is **pluggable and its dimensionality is not fixed** — DINOv2 (Apache 2.0) ships, ArcFace is built for evaluation only, because the piece is expected to be commercial and every mainstream face-recognition weight set is research-only (ADR-0007).
- **Layout is two stages**: a self-organizing map supplies ordering, and an LAPJV linear assignment places one Face per Cell with a movement penalty so re-solves animate coherently (ADR-0003).
- **The Grid is a Window** onto a much larger persistent Corpus, sized on the piece's own clock and drifting slowly across the archive (ADR-0004).
- **Capture is opt-in** and the Corpus persists, so Consent Records, Receipt Codes, and a tested deletion path are components with tickets (ADR-0005).
- **Hardware is deferred**, so camera access sits behind a trait and the CPU inference path must stay viable. The model identifier is part of the store schema because changing models invalidates every Embedding (ADR-0006).

## Next actionable work

Stage 0 in the implementation plan: stand up the Cargo workspace with the seven crates and CI. The model-licensing question that previously blocked it is resolved in ADR-0007.

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
