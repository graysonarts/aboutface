# ADR-0006: Stay portable across hardware; abstract capture, keep inference CPU-viable

- **Status:** Accepted
- **Date:** 2026-08-17

## Context

The installation machine is not chosen. The candidates — Apple silicon, a Linux
box with a discrete GPU, or a small ARM board inside a sculptural enclosure —
differ in the three things that matter to this system: the ONNX execution
provider, the wgpu backend, and the camera API.

Of those, wgpu already abstracts the graphics backend and `ort` already abstracts
the execution provider. Camera capture does not abstract itself, and is the one
place where a platform choice leaks into ordinary code if it is not contained.

## Decision

No hardware commitment. Instead:

- `afcapture` exposes a `Camera` trait. `nokhwa` is the initial cross-platform
  implementation; a platform-specific backend can be swapped behind the trait if
  it proves inadequate. No other crate imports a camera API.
- `afvision` selects an ONNX Runtime execution provider at startup — CoreML,
  CUDA/TensorRT, or CPU — and **the CPU path must remain viable**. Model choice is
  therefore constrained: if a full ArcFace R100 is too slow on CPU, a smaller
  MobileFaceNet-class model is used instead. Correct-on-CPU is the baseline;
  accelerators are an optimization.
- Grid size, Drift rate, and SOM resolution are configuration, not constants, so
  the piece can be scaled down for a weaker machine without code changes.
- A startup self-check reports which providers, backend, and camera resolved, so
  that a misconfigured install is obvious rather than mysteriously slow.

## Consequences

- More abstraction than a committed single-target build would need. That is the
  price of deferring, and it is deliberately confined to two seams.
- Performance work cannot be finished until hardware is chosen. Budgets should be
  measured on the real machine, and the tracer bullet (Stage 1) should be run on
  every candidate before committing.
- Embeddings are model-specific. Changing the model — including swapping to a
  smaller one for a weaker machine — **invalidates the entire Corpus's
  Embeddings**. The store must record which model produced each Embedding, and a
  re-embed migration path must exist from day one. This is the single most
  expensive consequence of deferring the hardware decision, and it is why the
  model identifier is part of the schema rather than an afterthought.
- The original photographs must be retained at full quality precisely so that
  re-embedding is possible.
