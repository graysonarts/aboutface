# ADR-0008: Model assets are fetched, configured, and identified by file name

- **Status:** Accepted
- **Date:** 2026-08-16
- **Refines:** ADR-0006, ADR-0007

## Context

The ViT size of the DINOv2 embedder is still open and follows the hardware
decision (ADR-0006), yet the `ModelId` is part of the store schema because
changing models invalidates every Embedding in the Corpus. A constant anywhere —
a path, an identifier, an embedding width — would either bake in a size or let
two different models share one identifier.

The ONNX files themselves are 227 KB (YuNet) and 84 MB–1.2 GB (DINOv2 by size),
so they do not belong in git.

## Decision

- **Fetched, not vendored.** `scripts/fetch-models.sh` downloads both files into
  a git-ignored `models/`, pinned to a source revision and verified against a
  recorded SHA-256. Sources and licences are documented in `docs/models.md`.
- **Configured, not constant.** `booth.toml` names the models directory, each
  model's file, and optionally its identifier. `afbooth` parses it; `afvision`
  turns it into `ModelSpec`s and owns the `ModelId`.
- **Identified by file name.** When configuration states no identifier, the
  `ModelId` is the file's stem — `dinov2-small.onnx` becomes `dinov2-small`.
  The ViT size therefore lives in the file name and flows into the schema
  without any code knowing that DINOv2 has sizes at all.
- **The self-check reports and refuses.** Every launch prints each model's
  resolved path, presence and `ModelId`, plus the ONNX Runtime execution
  provider selected and the runtime build. A missing file is a non-zero exit
  carrying the fetch command, never a panic.
- **CoreML is the only accelerator compiled in so far.** Selection probes ONNX
  Runtime's own availability check in preference order and falls back to CPU,
  which ADR-0006 requires to stay viable. CUDA and TensorRT join the probe list
  when there is a Linux GPU machine to test them on; adding them is a feature
  flag and a line in the preference order, not a redesign.

## Consequences

- Renaming a model file silently changes the `ModelId`, and therefore which
  Embeddings the Corpus considers comparable. That is the intended behaviour —
  it fails closed via the existing `ModelId` mismatch refusal in `afcore` —
  but it means model files must not be renamed casually.
- `booth.toml` is committed and names ViT-S/14 as today's starting point. That
  is configuration, not the settled answer to the open ViT-size question.
- Adding `ort` to `afvision` pulls an ONNX Runtime binary download into the
  build. The alternative, requiring a system ONNX Runtime, would make a fresh
  checkout fail to build on a machine that has never done inference before.
