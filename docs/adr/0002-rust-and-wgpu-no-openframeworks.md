# ADR-0002: Rust and wgpu for the whole system; no openFrameworks

- **Status:** Accepted
- **Date:** 2026-08-17

## Context

The project was described as having a C++ image-analysis half and an
openFrameworks display half, with an intent to "update openFrameworks to the
latest version" and "migrate to Rust where it makes sense."

Investigation found **no openFrameworks code anywhere in this repository**, and
none in the maintainer's hands to bring in. There is nothing to update.

That matters, because openFrameworks' strongest argument in a project like this
is inheritance: existing sketch code, existing addons wired up, an existing feel
for the tool. With no such code, oF has to win on merits alone, against a stated
preference for Rust — and while carrying the cost of a C ABI seam between a Rust
core and a C++ renderer, two toolchains, and two build systems.

The actual rendering demand is modest: up to ~1000 textured quads on a grid, with
animated per-quad transforms. This is comfortably within wgpu's range and does
not need a creative-coding framework.

## Decision

The entire system is Rust. Rendering is `wgpu` with `winit`. openFrameworks is
not used, and the "upgrade openFrameworks" work item is dropped.

The system is a single process, organized as a Cargo workspace:

```
crates/
  afcore/     domain types — Face, Embedding, GridSpec, Assignment, Window
  afvision/   detect + align + embed (ort / ONNX Runtime)
  afstore/    the Corpus — SQLite records, image files, Consent Records
  aflayout/   self-organizing map + linear assignment
  afcapture/  camera abstraction and Shutter handling
  afrender/   wgpu + winit display
  afbooth/    the binary that wires them together
```

## Consequences

- One language, one build, one artifact to deploy. No FFI.
- The crate boundaries carry forward the old architecture's discipline — one job
  per stage, explicit types between stages — without paying for separate
  processes.
- Camera capture is the real cost of dropping oF. It is the one thing the
  framework would have handed over for free. `afcapture` exists specifically to
  isolate that pain behind a trait (see ADR-0006).
- `afrender` owns windowing, fullscreen, and display timing, which oF would also
  have provided. This is a known, bounded chunk of work.
- Shader-driven visual experimentation is somewhat less immediate than in oF.
  Accepted; the piece's visual language is layout and motion, not shader effects.
- The Rust ecosystem for the numeric work is adequate but thin in places. The
  self-organizing map is expected to be written directly rather than pulled from
  a crate — it is a few hundred lines and the growth behaviour is bespoke.
