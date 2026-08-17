# ADR-0009: The camera seam ships a fake, and frames cross it as plain RGB8

- **Status:** Accepted
- **Date:** 2026-08-16

## Context

ADR-0006 decided *that* camera access sits behind a trait with `nokhwa` behind
it. It did not decide what crosses the seam, or how the rest of the system is
supposed to be exercised on a machine with no webcam — which is every machine
the piece is developed and tested on, and CI.

Three things forced a choice while building the trait:

- `nokhwa` hands back a backend `Buffer` and can decode into an `image` type. If
  either escaped, `afcapture` would stop being the only crate that knows a
  camera API exists, and the swap ADR-0006 reserves the right to make would stop
  being cheap.
- `nokhwa`'s `input-native` feature pulls every platform's bindings on every
  platform, and its `NokhwaError` flattens all platform failures into message
  strings — there is no error code to switch on.
- A fake camera is needed by tests in `afvision`, `afstore` and `afbooth`, not
  just by `afcapture`'s own tests.

## Decision

- **Frames cross the seam as `afcapture::Frame`** — a validated, tightly packed
  RGB8 buffer with its dimensions, and nothing else. No backend type and no
  `image` type appears in the trait's signature.
- **The fake camera is a published part of the library** (`afcapture::testing`),
  compiled unconditionally rather than behind a `testing` feature. Cargo unifies
  features across a workspace, so a feature enabled by one crate's
  dev-dependencies is enabled for every build of `afcapture` in that graph
  anyway; the gate would buy no isolation and would add a way for the workspace
  to fail to compile. It replays photographs from disk and scripts its failures
  rather than waiting for them to occur.
- **`CameraError` names absent, busy and disconnected separately.** Absent is
  decided on the evidence of the device list before the open is attempted; busy
  and permission-denied are recognised by what the platform wrote, falling back
  to a reported backend error rather than a panic or a guess. An operator in a
  gallery needs to be told which of the three happened.
- **`nokhwa` is an optional, per-target dependency behind a default-on
  `nokhwa-backend` feature**, so each platform builds only its own bindings and
  a machine that cannot build them can still build the workspace against the
  fake.
- **Captures are written as PNG**, losslessly, because re-embedding the Corpus
  after a model change depends on the original frame (ADR-0006).

## Consequences

- Every frame is converted to RGB8 and copied at the seam. At one Capture per
  Visitor this cost is irrelevant, and it is the price of the backend staying
  swappable.
- Classifying "busy" and "permission denied" from message text is fragile across
  platforms and `nokhwa` versions. It degrades to a reported backend error, so
  the failure mode is a less specific message, never a crash — but the string
  lists will need revisiting when hardware is chosen.
- The booth binary links a small amount of image-decoding code it never uses, so
  that the fake is available to every crate's tests.
