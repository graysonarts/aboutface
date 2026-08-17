# ADR-0007: Similarity is model-pluggable; the shipping model must be permissively licensed

- **Status:** Accepted
- **Date:** 2026-08-17
- **Refines:** ADR-0001

## Context

ADR-0001 assumed the Embedding would come from an ArcFace-family model. Checking
the licenses before committing turned up two problems, one legal and one
conceptual.

**The legal problem is ecosystem-wide, not repo-specific.** Verified against
upstream sources on 2026-08-17:

- YuNet (OpenCV Zoo) is **MIT**. Detection and alignment are unencumbered.
- InsightFace's *code* is MIT with "no limitation for both academic and
  commercial usage," but its *pretrained models* are "available for
  non-commercial research purposes only," with `buffalo_l` requiring direct
  contact for other use.
- facenet-pytorch publishes no explicit weight license and its weights derive
  from VGGFace2 and CASIA-WebFace.
- DINOv2's "code and model weights are released under the Apache License 2.0."

Every mainstream face-recognition model is trained on MS-Celeb-1M, Glint360K,
VGGFace2, or CASIA-WebFace. All are research-only; two have been withdrawn. This
cannot be solved by choosing a different repository.

**The conceptual problem is the more interesting one.** ArcFace is trained to be
*invariant* to hairstyle, glasses, expression, lighting, age and pose — it
recovers identity through those nuisances. But "people who look alike," as a
visitor standing at the wall means it, is substantially *composed of* those
things. A face-recognition embedding discards much of what a human calls
resemblance and preserves what a passport check calls it. A general visual
embedding such as DINOv2 does close to the opposite.

These are two different artworks, and it is not obvious which one this is.

The installation is expected to be commercial — galleries, commissions,
admission — so the non-commercial research exemption is not available to the
shipping system.

## Decision

**The similarity model is a pluggable component.** `afvision` exposes the
embedding step behind a trait, and the store records which model produced each
Embedding (already required by ADR-0006).

**Two implementations are built:**

1. **DINOv2 (Apache 2.0)** — apparent resemblance. This is the **shipping
   default**, and the only model permitted in a commercial installation unless
   the position changes.
2. **ArcFace-family (InsightFace weights)** — identity resemblance. Built for
   **evaluation only**. It informs the design decision; it does not ship.

The two walls are compared by eye during Stage 2, over the same Corpus.

Detection and alignment are **YuNet (MIT)** for both paths. Both consume the same
aligned crop; only the resize target and normalization differ.

## Consequences

- **The comparison has an asymmetric outcome, and this is the main thing to
  understand about this decision.** If DINOv2 wins, it ships. If ArcFace wins,
  no decision has been made — a licensing negotiation with InsightFace has been
  triggered, and it gates release. Budget time for that possibility rather than
  discovering it on the night the ArcFace wall looks better.
- Embedding dimensionality is now model-dependent — 512 for ArcFace, 384/768/1024
  for DINOv2 depending on the ViT size. Nothing may assume 512. `CONTEXT.md` is
  updated accordingly.
- Both Embeddings can be stored per Face during evaluation; at ~2–4 KB per Face
  this is free at any plausible Corpus size, and it means the comparison does not
  require re-capturing anyone.
- **Known risk with DINOv2:** a general visual embedding may cluster on lighting,
  background, and framing rather than on people. Mitigations, in order of
  preference: a tight aligned crop that excludes background, consistent booth
  lighting, and background masking if the first two are insufficient. If the wall
  is visibly organizing by shirt colour, this is why.
- The choice interacts with ADR-0005. An identity-capable embedding is
  unambiguously a biometric identity template. A general visual embedding is a
  weaker identifier, which likely reduces exposure — though the My Health My Data
  Act's definition of biometric data is broad and this should not be treated as
  an exemption without a lawyer saying so.
- YuNet being MIT means the detector never blocks anything. If the project ever
  needs a fallback, the licensing risk sits entirely in the embedding stage.
