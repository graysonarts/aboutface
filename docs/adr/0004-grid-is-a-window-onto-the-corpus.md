# ADR-0004: The Grid is a Window onto the Corpus, sized on a clock

- **Status:** Accepted
- **Date:** 2026-08-17

## Context

Two requirements were stated that appear compatible and are not:

- One Cell holds exactly one Face, always.
- The Grid resizes periodically on a fixed rhythm, ranging from about 10 Cells to
  about 1000.

If Cells are people, Grid size is a function of headcount and a clock has no say.
Concretely: 200 Faces in the Corpus, the clock calls for 10 Cells. Either 190
people disappear from the wall, or Cells are no longer people.

A third requirement settles it — the Corpus persists and grows without bound
(ADR-0005). An unbounded Corpus can never fit on a bounded Grid regardless of
rhythm, so *something* must always be off-screen.

## Decision

The Grid is a **Window** onto the Corpus.

- Cells hold exactly one Face. This is preserved absolutely.
- Grid dimensions are driven by a clock — a fixed rhythm belonging to the piece,
  independent of how many people are present — sweeping between roughly 10 and
  roughly 1000 Cells.
- The Window holds exactly as many Faces as there are Cells. The rest of the
  Corpus is present but unseen.
- Which Faces are in the Window is decided by **Drift**: the Window moves slowly
  and continuously through the Corpus, so that over hours the whole archive
  surfaces.

Grid size and Window membership are therefore two independent slow processes: one
controls *how many* faces are visible, the other controls *which*.

For the self-organizing map, the underlying Node lattice is trained at a fixed
resolution over the Corpus and is **resampled** to the current Grid dimensions,
rather than being retrained on every resize. Retraining a fresh SOM at each tick
of the clock would discard learned structure and make the resize look like a
glitch instead of a breath.

## Consequences

- The piece has two distinct scales that mean different things. At ~10 Cells the
  portraits are large and intimate and the wall is a statement about a handful of
  people. At ~1000 it is a crowd. The work changes character as it breathes, and
  that is intended.
- **Known risk, accepted:** with Drift as the selection rule, a Visitor may
  trigger the Shutter and never see their own Face on the wall while standing
  there. That is in tension with the project's own description — "displays them
  alongside similar looking people" — which promises a personal payoff. Drift was
  chosen deliberately over a "newest arrival's Neighborhood" rule, in favour of a
  contemplative, archive-like reading over an interactive one.
- **Open question:** whether to mitigate that risk with a brief exception — on
  Capture, the Window jumps to the new Face's Neighborhood for a short interval,
  then resumes Drifting. This preserves the contemplative default while giving
  each Visitor one moment of recognition. Not decided; to be tried in the layout
  prototype and judged by eye.
- **Open question:** the Drift rate, and whether Drift is a random walk through
  Embedding space, an ordered sweep, or chronological. Each produces a very
  different sense of the archive.
- Only Window-resident Faces need decoded textures in GPU memory, so texture
  budget is bounded by Grid size, not Corpus size. This is a significant relief at
  scale.
- Faces entering and leaving the Window on Drift need their own transition
  treatment, distinct from Cell-to-Cell movement during Re-solve.
