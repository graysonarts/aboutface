# About:Face — Context

The shared vocabulary for this project. When an issue, ADR, module name, or test
name refers to a concept below, use **this** term, not a synonym.

## What the piece is

About:Face is a **booth**. A visitor steps up, triggers the shutter themselves,
and their portrait joins a growing corpus of everyone who has ever done so. A
wall of portraits arranges itself so that people who look alike sit next to each
other. The wall slowly breathes between showing ten faces and showing a
thousand, and slowly drifts across the whole accumulated crowd.

It is *not* a surveillance piece. Capture is deliberate and opt-in. (This is a
change from the original README framing of "an automated camera takes pictures of
people" — see ADR-0005.)

## Glossary

**Visitor** — a person physically present at the installation. Not "user."

**Capture** — the act of taking one portrait. Always visitor-initiated.

**Shutter** — the physical or on-screen control a Visitor presses to consent to
and trigger a Capture. The Shutter *is* the consent gesture; there is no separate
consent step.

**Face** — one captured portrait of one Visitor at one moment, plus everything
derived from it (Embedding, crops, Consent Record). A Visitor who returns twice
produces two Faces. The system does not attempt to recognize that they are the
same person, and deliberately does not link them.

**Corpus** — every Face ever retained, across every showing of the piece. It
persists between installations and grows without bound. The Corpus is the
artwork's memory.

**Embedding** — the 512-dimension L2-normalized vector produced from an aligned
Face crop by the face-recognition model. Distance between Embeddings is cosine
distance. This is what "similar looking" means operationally, and it is legally a
biometric identifier — treat it with the same care as the photograph.

**Neighborhood** — the set of Faces nearest a given Face in Embedding space.

**Grid** — the rectangular lattice of Cells currently on screen. Its dimensions
change over the run of the piece (see Window).

**Cell** — one position in the Grid. A Cell holds exactly one Face. Cells are
never shared, stacked, or cycled.

**Window** — the subset of the Corpus currently displayed. Because Cells hold
exactly one Face and the Grid ranges from ~10 to ~1000 Cells, the Grid is a
Window onto a much larger Corpus, not a view of all of it.

**Drift** — the slow, continuous movement of the Window across the Corpus, so
that over hours the whole archive surfaces.

**Node** / **Prototype** — a unit of the self-organizing map and its associated
vector. Nodes give the Grid its *ordering*. Nodes are not Cells, and a Node is
not a person.

**Assignment** — the bijective mapping of the Window's Faces onto the Grid's
Cells, solved as a linear assignment problem. Ordering comes from the map;
placement comes from Assignment.

**Re-solve** — recomputing the Assignment and animating every Face to its new
Cell. The visible reorganization of the crowd is a feature of the piece, not a
side effect.

**Consent Record** — the stored evidence that a Visitor triggered the Shutter:
timestamp, the exact consent text shown, and the Receipt Code issued.

**Receipt Code** — a short code handed to the Visitor at Capture. Presenting it
later causes that Face, its Embedding, its image files, and its Consent Record to
be destroyed.

## Terms we deliberately avoid

- **"Annotator."** The retired C++ tools were annotators. The Rust pipeline has
  stages, not annotators. Reusing the word implies the old architecture survived;
  it did not (ADR-0001).
- **"User."** Say Visitor.
- **"Cluster."** Cells hold individuals, not clusters. If you mean a group of
  similar Faces, say Neighborhood.
- **"Recognition."** The piece measures similarity. It never identifies anyone,
  never names anyone, and never links two Faces to one person.
