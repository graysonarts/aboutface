*This project is in initial development.  More information will be coming later*

**About:Face** is an art project. A visitor steps up to a booth and triggers the
shutter themselves; their portrait joins a growing collection of everyone who has
ever done so. A wall of portraits arranges itself so that people who look alike
sit next to each other, slowly breathing between showing a handful of faces and
showing a thousand, and slowly drifting across the whole accumulated crowd.

Capture is deliberate and opt-in — you choose to be photographed, and you are
given a code that lets you have yourself removed at any time.

**I'm looking for collaborators** on this project, so if interested in collaborating,
please open an issue.  Local to Seattle prefered since there is hardware involved.

## Where things stand

The direction is documented and the build is being restarted:

- [`CONTEXT.md`](CONTEXT.md) — the project's vocabulary.
- [`docs/adr/`](docs/adr/) — the decisions and why they were made.
- [`docs/implementation-plan.md`](docs/implementation-plan.md) — the staged plan.

The C++ / OpenCV tools currently in this repository are a 2015 prototype of face
*detection*. They are retired and are being replaced by a Rust implementation
built on learned face embeddings — see
[ADR-0001](docs/adr/0001-retire-opencv-annotators-for-learned-embeddings.md).
