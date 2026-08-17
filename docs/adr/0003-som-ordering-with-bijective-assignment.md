# ADR-0003: Self-organizing map for ordering, linear assignment for placement

- **Status:** Accepted
- **Date:** 2026-08-17

## Context

Faces must be laid out on a 2D Grid so that similar Faces are adjacent, and when
a new Face arrives the Grid re-solves and every Face animates to its new Cell.

Two properties are in tension.

**Projection instability.** UMAP and t-SNE are not stable under re-fit. Adding one
point and re-running rotates, reflects, and reshuffles the entire map. Everyone
would fly across the screen on every Capture — motion that reads as noise rather
than as the crowd reorganizing. Anchoring is required, whatever the method.

**Nodes are not people.** A self-organizing map's units are prototype vectors, and
many Faces map to the same Node. A SOM alone therefore cannot satisfy
"one Cell holds exactly one Face" — at 10 Nodes with 200 Faces, twenty Faces
would claim each Node.

The cost of exactness was checked rather than assumed. The Hungarian algorithm is
O(n³), which sounds prohibitive at n=1000, but LAPJV on a 1000×1000 dense cost
matrix completes in tens of milliseconds on ordinary hardware. Exact optimal
assignment is affordable across the entire Grid range.

## Decision

Layout is two stages:

1. **Ordering.** A self-organizing map with rectangular lattice topology is
   trained on the Corpus's Embeddings. Its Node grid supplies a stable, spatially
   coherent arrangement of the Embedding space, and unlike UMAP it updates
   incrementally as Faces arrive rather than being re-fit from scratch.

2. **Placement.** The Window's Faces are assigned bijectively to Grid Cells by
   solving a linear assignment problem (LAPJV) over a cost matrix.

The assignment cost for placing Face *f* in Cell *c* is:

```
cost(f, c) = cosine_distance(embedding(f), prototype(node_at(c)))
           + λ · movement_penalty(current_cell(f), c)
```

The movement penalty is essential and is the reason not to place Faces greedily.
Without it, an optimal-by-similarity solution will happily teleport a Face across
the wall to save a trivial amount of distance, and the animation reads as chaos.
λ is a tuning dial for how sticky the layout feels, and is expected to be tuned
by eye during installation.

## Consequences

- Ordering and placement are separable and separately testable. The SOM can be
  evaluated on quantization/topographic error; the assignment on total cost and
  total movement.
- Re-solve is cheap enough to run on every Capture at any Grid size in range.
- λ becomes an artistic control, not just a performance knob. It is worth exposing
  in an operator config.
- The SOM must be persisted alongside the Corpus so a restart does not reshuffle
  the wall. Training state is part of the installation's saved state.
- A SOM is less faithful to true Embedding distances than UMAP. Accepted: grid
  topology and incremental stability matter more here than metric fidelity.
- Grid resizing interacts with SOM training and is handled in ADR-0004.
