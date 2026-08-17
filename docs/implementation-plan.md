# About:Face — Implementation Plan

Derived from a grilling session, 2026-08-17. Vocabulary is defined in
[`CONTEXT.md`](../CONTEXT.md); decisions and their reasoning are in
[`docs/adr/`](adr/). This document is the *how* and the *order*; the ADRs are the
*why*, and they win in a conflict.

## The system in one paragraph

A Visitor presses the Shutter. A camera frame is captured; a detector finds the
face and its five landmarks; the frame is warped to an aligned 112×112 crop; an
ArcFace-family ONNX model produces a 512-dimension Embedding. The Face, its
Embedding, its display crop, and its Consent Record are written to the Corpus,
and the Visitor is handed a Receipt Code. A self-organizing map trained over the
Corpus supplies a spatial ordering; a linear assignment solve places the Window's
Faces bijectively into the Grid's Cells; every Face animates to its new Cell. The
Grid's size breathes between ~10 and ~1000 Cells on the piece's own clock, and the
Window Drifts slowly across the whole Corpus.

## Workspace

```
crates/
  afcore/     Face, Embedding, GridSpec, Window, Assignment. No I/O, no OpenCV,
              no GPU. Everything else depends on this and it depends on nothing.
  afvision/   detect → align → embed, over `ort`. Owns the model identifier.
  afstore/    SQLite + image files + Consent Records + deletion. Owns migrations.
  aflayout/   SOM training and resampling; LAPJV assignment with movement penalty.
  afcapture/  `Camera` trait, `nokhwa` implementation, Shutter input.
  afrender/   wgpu + winit; texture management; animation.
  afbooth/    the binary; config; startup self-check; operator controls.
```

The old C++ tree (`annotators/`, `common/`, `contrib/`, `data/`, `CMake/`) is
retired per ADR-0001. It stays in git history. Deleting it from the working tree
is a task in Stage 0, not a prerequisite for anything.

## Key technical choices

| Concern | Choice | Note |
| --- | --- | --- |
| Detect + align | SCRFD or YuNet | one step, emits bbox + 5 landmarks |
| Embed | ArcFace-family ONNX, 512-D, L2-normalized | cosine distance |
| Inference runtime | `ort` (ONNX Runtime) | CPU path must stay viable — ADR-0006 |
| Store | SQLite (`rusqlite`), images on disk | brute-force cosine is fine at this scale |
| Ordering | self-organizing map, rectangular lattice | written directly, ~300 lines |
| Placement | LAPJV linear assignment | ~tens of ms at n=1000 |
| Camera | `nokhwa` behind a trait | the one real platform seam |
| Render | wgpu + winit, instanced textured quads | ≤1000 quads is undemanding |

Two crops per Face, and they are not interchangeable: the tight aligned 112×112
crop that feeds the model, and a looser portrait crop for display. Retain the
original full-quality frame as well — re-embedding depends on it (ADR-0006).

## Stages

Each stage is a tracer bullet: it ends with something that runs end to end, not
with a finished layer.

### Stage 0 — Ground

- Land `CONTEXT.md` and the ADRs. *(done — this commit)*
- Update `README.md` to the booth framing. *(done — this commit)*
- Decide the model (SCRFD/YuNet + ArcFace variant) and confirm its license
  permits installation use. **Blocks Stage 1.**
- Delete the retired C++ tree, or explicitly decide to leave it in place.
- Stand up the Cargo workspace with the seven empty crates and CI that builds
  them.

### Stage 1 — Tracer bullet: shutter to wall

The thinnest possible end-to-end path. No SOM, no assignment, no animation:
Faces are placed in arbitrary order.

- `afcapture`: open a camera, grab a frame on a keypress.
- `afvision`: detect, align, embed. Print the Embedding's norm to prove it works.
- `afstore`: schema and write path — Face, Embedding (with model identifier),
  crops, original frame.
- `afrender`: a fixed grid of stored Faces on screen, filling in arbitrary order.
- `afbooth`: wire it together; startup self-check reports provider, backend,
  camera.

**Done when:** pressing a key adds your face to a grid on screen, and it is still
there after a restart.

**Run this stage on every candidate machine before choosing hardware (ADR-0006).**

### Stage 2 — Layout

- SOM training over the Corpus's Embeddings; persist training state.
- LAPJV assignment with the movement-penalty cost from ADR-0003.
- Animated transitions: Faces interpolate from old Cell to new Cell on Re-solve.
- Expose λ (movement penalty weight) in config and tune it by eye.

**Done when:** adding a Face visibly reorganizes the wall, and people who look
alike are adjacent. This is the first stage where the piece is recognizably
itself.

### Stage 3 — Breath and Drift

- Clock-driven Grid resize between ~10 and ~1000 Cells; SOM lattice resampled to
  the current dimensions rather than retrained (ADR-0004).
- Window Drift across the Corpus, with its own transition treatment for Faces
  entering and leaving.
- Try the on-Capture Neighborhood jump described as an open question in ADR-0004,
  and decide it by eye.
- Settle the Drift rate and whether Drift is a walk, a sweep, or chronological.

**Done when:** the wall breathes and wanders unattended for an hour and looks
intentional the whole time.

### Stage 4 — The booth

- Shutter interaction and on-screen consent text, versioned.
- Consent Record written with every Face.
- Receipt Code generation and hand-off.
- Deletion tool: code in, Face and Embedding and crops and Consent Record out,
  SOM updated. **Tested, not improvised.**
- Signage and consent copy drafted for legal review (ADR-0005).

**Done when:** a stranger can be photographed, receive a code, and have
themselves removed — with the wall reflecting the removal.

### Stage 5 — Install hardening

- Kiosk startup, crash recovery, corpus backup and restore.
- Operator controls: pause capture, hide a Face, force a Re-solve.
- Bad-capture handling: no face detected, multiple faces in frame, motion blur.
  (`face_isolator`'s old `faces[0]` bug is the cautionary tale — decide the
  multi-face rule explicitly and test it.)
- Lighting, camera mounting, and the physical Shutter.
- Re-embed migration path, exercised at least once end to end.

**Done when:** it survives a full unattended day.

## Open questions

Tracked here, not resolved:

- Drift rate and Drift mode (walk / sweep / chronological) — ADR-0004.
- Whether Capture briefly jumps the Window to the new Face's Neighborhood, to
  give each Visitor a moment of recognition — ADR-0004.
- What the first showing opens with, given an empty Corpus — ADR-0005.
- Hardware, and therefore the final model size — ADR-0006.
- Multi-face-in-frame policy at the Shutter.
- Whether display crops are square or portrait, and how tightly framed.

## The risk worth restating

Drift means a Visitor may press the Shutter and never see their own Face while
standing there. The README promises to display people "alongside similar looking
people," and Drift does not guarantee that promise is kept to the person who just
participated. This was chosen deliberately — contemplative archive over
interactive mirror — and the open question in ADR-0004 exists to revisit it once
there is something to look at.
