# ADR-0007: Similarity is apparent resemblance, from permissively-licensed weights

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
cannot be solved by choosing a different repository. The installation is expected
to be commercial (ADR-0005), so the research exemption is unavailable.

**The conceptual problem is the more interesting one.** ArcFace is trained to be
*invariant* to hairstyle, glasses, expression, lighting, age and pose — it
recovers identity through those nuisances. But "people who look alike," as a
visitor standing at the wall means it, is substantially *composed of* those
things. A face-recognition embedding discards much of what a human calls
resemblance and preserves what a passport check calls it. A general visual
embedding such as DINOv2 does close to the opposite.

An earlier version of this ADR proposed building both and choosing by eye in
Stage 2. That was rejected: the comparison's outcomes were asymmetric — DINOv2
winning would settle the question, while ArcFace winning would only open a
licensing negotiation that gates release. Paying for a second implementation to
obtain a result that could not be acted on is not worth it.

## Decision

**Similarity means apparent resemblance**, not identity resemblance. This is an
artistic commitment, not merely a licensing workaround: the wall pairs people who
look alike the way a stranger means it — colouring, hair, glasses, expression,
the set of the face — rather than people a recognition system would judge to be
the same person.

- **Detection and alignment: YuNet (MIT).**
- **Embedding: DINOv2 (Apache 2.0).** This is the only similarity model built.
- **ArcFace and every other face-recognition weight set are dropped**, and are
  not to be reintroduced without revisiting this ADR and ADR-0005 together.

The embedding step still sits behind a trait in `afvision`, and the store still
records which model produced each Embedding (ADR-0006). The reason is no longer
an A/B against ArcFace — it is that DINOv2 ships in several ViT sizes with
different dimensionalities, and the hardware decision is still open.

## Consequences

- No licensing exposure anywhere in the pipeline. Nothing gates release.
- One embedding implementation instead of two, and no bake-off stage.
- **Accepted loss:** it will never be known whether an identity-similarity wall
  was the better artwork. The cost of finding out was a licensing negotiation
  with release-blocking risk, and that price was judged too high.
- Embedding dimensionality is model-dependent — 384 / 768 / 1024 / 1536 across
  the DINOv2 ViT sizes. Nothing may assume a fixed width, and the ViT size is a
  live variable until hardware is chosen (ADR-0006).
- **Known risk, and the main technical one now:** a general visual embedding may
  cluster on lighting, background, and framing rather than on people. If the wall
  visibly organizes by shirt colour or by which side the light falls on, this is
  the cause. Mitigations in order of preference: a tight aligned crop that
  excludes background, consistent booth lighting, then background masking.
  Because there is no second model to fall back on, **this risk must be tested
  early — in Stage 1, on real captures, not in Stage 2.**
- A general visual embedding is a weaker identifier than an identity template,
  which likely reduces exposure under ADR-0005. The My Health My Data Act's
  definition of biometric data is broad, so this is not an exemption without a
  lawyer saying so.
