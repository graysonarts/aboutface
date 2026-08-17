# ADR-0010: The Embedding is DINOv2's CLS token, fed a 224×224 crop

- **Status:** Accepted
- **Date:** 2026-08-17
- **Refines:** ADR-0007

## Context

DINOv2's ONNX export ends at `last_hidden_state` — one token per patch plus a
leading CLS token — so something has to pool it into a single vector, and the
input resolution has to be chosen because the export leaves height and width
symbolic.

## Decision

The Embedding is the CLS token: the image-level summary the model was trained to
produce, and what DINOv2's own linear evaluations read. The aligned 112×112 crop
is upscaled to 224×224 — the resolution the model was trained at, and a whole
number of 14px patches.

Preprocessing follows `facebook/dinov2-*`'s image processor: bicubic resampling,
`/255`, ImageNet channel means and standard deviations, RGB, NCHW. One step is
deliberately dropped — the reference resizes the shortest edge to 256 and
centre-crops to 224, which here would cut the face off a crop that is already
tightly framed by the landmark template.

## Considered Options

Mean-pooling the patch tokens, or concatenating CLS with the patch mean, are the
common alternatives and are reported to help on dense tasks. They are not chosen
here: the patch tokens over an aligned crop carry background and hair as much as
face, and the piece is already at risk of keying on the room rather than on
people (Stage 1's sanity check). Concatenation would also double the width for
no argued benefit.

## Consequences

Pooling and preprocessing are part of what produced an Embedding, exactly as the
model file is: changing either invalidates every Embedding in the Corpus
(ADR-0006) but does *not* change the `ModelId`, which is derived from the file
name (ADR-0008). If either is revisited, the `ModelId` must be overridden in
`booth.toml` to mark the break.
