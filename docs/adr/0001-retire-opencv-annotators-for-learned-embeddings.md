# ADR-0001: Retire the OpenCV annotator pipeline in favour of a learned face embedding

- **Status:** Accepted
- **Date:** 2026-08-17

## Context

The repository contains three independent C++ executables built on OpenCV 2.x:

- `face_isolator` — Haar cascade detection (frontal, falling back to profile).
- `skin_color_picker` — an HSV skin mask reduced to a single scalar.
- `feature_extractor` — 75-point ASM landmark fitting via vendored
  `asmlib-opencv`.

They are composed by hand through stdout using the `X,Y+WxH` string format in
`af::common::Rectangle`. The last substantive commit is from around 2015.

Between them they yield on the order of three to five usable dimensions. The
piece requires a space rich enough that "who is near whom" is meaningful to a
human looking at the wall. It is not close.

Three concrete problems with the existing code, beyond dimensionality:

1. `face_isolator.cpp` returns `faces[0]` — an arbitrary detection, not the
   largest or most central. With more than one person in frame the result is
   undefined.
2. `skin_hue_averager.cpp:70` computes `cv::mean(skinMask)` over the **binary
   mask**, not over the hue channel. The reported "average hue" is actually
   `255 × (fraction of pixels classified as skin)`. The annotator does not
   measure what its name and output label claim.
3. The code uses OpenCV 2.x C-era constants (`CV_BGR2GRAY`, `CV_RGB`,
   `CV_BGR2HSV_FULL`). It will not compile against OpenCV 4 without edits.

Modern face-similarity models take a *5-point-aligned* 112×112 crop, which means
detection and landmark alignment collapse into a single step. That step is served
by SCRFD or YuNet, not by a Haar cascade plus a separate ASM fit.

## Decision

Similarity is defined by a learned face embedding: a 512-dimension L2-normalized
vector from an ArcFace-family ONNX model, compared by cosine distance.

The detection/alignment front end is SCRFD or YuNet, producing a bounding box and
five landmarks, which are warped by similarity transform to the 112×112 crop the
embedding model expects.

`face_isolator`, `skin_color_picker`, and `feature_extractor` are retired. The
vendored `asmlib-opencv`, the Haar cascades in `data/`, and TCLAP are retired with
them.

## Consequences

- This is a rewrite, not a migration. Effectively none of the existing C++
  survives into the running piece.
- What *does* survive is architectural: one clear responsibility per stage, and a
  single well-defined data type carried between stages. That idea moves into the
  Rust crate split (ADR-0002) rather than into separate processes.
- The `X,Y+WxH` wire format is retired. Nothing crosses a process boundary any
  more, so a Rust type replaces it. `CLAUDE.md`'s description of it as "the
  inter-process wire format" becomes historical.
- The C++ tree is kept in git history and may be deleted from the working tree.
  It is not maintained, not built, and not a reference implementation.
- A model file must now be vendored or fetched, with a license that permits
  installation use. Model choice and provenance are a task, not an assumption.
- Two different crops are needed per Face and must not be confused: the tight
  aligned 112×112 crop that feeds the model, and a looser portrait crop for
  display. Only the first is geometrically constrained.
