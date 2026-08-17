# ADR-0005: A booth with opt-in Capture and a persistent Corpus

- **Status:** Accepted
- **Date:** 2026-08-17

## Context

The README describes About:Face as a piece where "an automated camera takes
pictures of people." Automatic capture of the public, combined with retaining
biometric data across showings, is the most legally and ethically loaded
configuration available.

The Corpus was chosen to persist and grow across installations, because the
accumulating crowd is the substance of the work. An ephemeral corpus would mean
starting empty every night and would gut the concept.

Persistence plus automatic capture would mean operating a standing biometric
database of members of the public who never agreed to anything.

## Decision

**Capture is deliberate and opt-in.** The Visitor triggers the Shutter
themselves. That gesture is simultaneously the consent and the exposure. The
piece is a **booth**, not a surveillance apparatus, and the README is updated to
say so.

**The Corpus persists** and grows across every showing.

Consent and deletion are components with tickets, not policy footnotes:

- A **Consent Record** is written with every Face: timestamp, the exact consent
  text displayed at that moment, and the Receipt Code issued.
- A **Receipt Code** is handed to the Visitor at Capture. Presenting it later
  destroys that Face, its Embedding, its image files, and its Consent Record.
- Deletion is a real operation with a real tool, tested, not a manual database
  edit under pressure at an opening.
- The Corpus never leaves the installation machine. No cloud storage, no
  telemetry, no network egress. This is a design constraint on every crate.

## Consequences

- The piece changes meaning. Surveillance becomes participation. Both are valid
  works; this is the one being built, and the framing in the README, artist
  statement, and signage must all match.
- Photo quality improves substantially — people face the camera on purpose — and
  the Corpus becomes self-selecting toward people willing to be photographed.
  That bias is real and worth naming in the artist statement.
- Cold start is solved by persistence: the second showing inherits the first.
  The very first showing still opens with a nearly empty wall, and that needs a
  deliberate answer (seed the Corpus with consenting collaborators, or let the
  emptiness be the opening).
- Deleting a Face requires retraining or at least updating the SOM, and
  invalidates cached layout state. Deletion is not just a row removal.
- **Legal, and explicitly not legal advice:** an installation in Seattle should
  have a Washington-licensed lawyer review this before opening. The relevant
  shapes are RCW 19.375 (biometric identifiers, which hinges partly on whether
  enrollment is for a "commercial purpose") and RCW 19.373, the My Health My Data
  Act, which covers biometric data and carries a private right of action. Opt-in
  capture with a working deletion path is a much stronger starting position than
  automatic capture, but it is not a substitute for that review.
- **The piece is expected to be commercial** — galleries, commissions, admission
  (decided 2026-08-17). That single fact has two consequences that are easy to
  treat as unrelated and are not: it removes the "commercial purpose" escape from
  the RCW 19.375 analysis above, and it removes the non-commercial-research
  exemption from the model licensing in ADR-0007. Both should be revisited
  together if that expectation ever changes.
- Consent text is versioned. If it changes, existing Consent Records must still
  record what was actually shown at the time.
