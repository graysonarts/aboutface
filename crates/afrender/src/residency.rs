//! Which Face's portrait lives in which texture slot.
//!
//! The GPU holds a fixed set of slots — one per Cell — and Faces move through
//! them as the Window Drifts (ADR-0004). A Face that stays on the wall keeps
//! its slot and is never decoded or uploaded twice; only arrivals cost an
//! upload.
//!
//! The policy is separate from the device so it can be tested on a machine with
//! no GPU.

use std::collections::HashMap;

use afcore::FaceId;

/// A portrait that needs decoding and uploading into a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Upload {
    /// The Face whose display crop is wanted.
    pub face: FaceId,
    /// The texture slot to put it in.
    pub slot: u32,
}

/// The slot each resident Face occupies.
#[derive(Debug, Clone, PartialEq)]
pub struct Residency {
    slots: Vec<Option<FaceId>>,
    by_face: HashMap<FaceId, u32>,
}

impl Residency {
    /// Creates a residency over `capacity` texture slots.
    pub fn with_capacity(capacity: u32) -> Self {
        Self {
            slots: vec![None; capacity as usize],
            by_face: HashMap::new(),
        }
    }

    /// Where `face`'s portrait sits, if it is resident.
    pub fn slot_of(&self, face: FaceId) -> Option<u32> {
        self.by_face.get(&face).copied()
    }

    /// Makes exactly `faces` resident, and reports what must be uploaded.
    ///
    /// Faces already resident keep their slot and are absent from the result:
    /// a re-solve that reshuffles the same crowd, or a resize of the OS window,
    /// costs no texture traffic at all. Faces that have left are evicted, and
    /// their slots are reused in index order.
    ///
    /// Faces beyond the slot count are dropped — the Window is never larger
    /// than the Grid, so this is a guard rather than a policy.
    pub fn sync(&mut self, faces: impl IntoIterator<Item = FaceId>) -> Vec<Upload> {
        let wanted: Vec<FaceId> = faces.into_iter().take(self.slots.len()).collect();
        let keep: HashMap<FaceId, u32> = wanted
            .iter()
            .filter_map(|face| self.slot_of(*face).map(|slot| (*face, slot)))
            .collect();

        for slot in &mut self.slots {
            if slot.is_some_and(|face| !keep.contains_key(&face)) {
                *slot = None;
            }
        }
        self.by_face = keep;

        let mut uploads = Vec::new();
        let mut free = 0;
        for face in wanted {
            if self.by_face.contains_key(&face) {
                continue;
            }
            // INVARIANT: `wanted` is no longer than the slot count and every
            // Face already resident was retained, so a free slot exists.
            while self.slots[free].is_some() {
                free += 1;
            }

            self.slots[free] = Some(face);
            self.by_face.insert(face, free as u32);
            uploads.push(Upload {
                face,
                slot: free as u32,
            });
        }

        uploads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upload(face: u64, slot: u32) -> Upload {
        Upload {
            face: FaceId(face),
            slot,
        }
    }

    #[test]
    fn should_upload_every_face_when_the_wall_first_fills() {
        let mut residency = Residency::with_capacity(4);

        let uploads = residency.sync([FaceId(1), FaceId(2)]);

        assert_eq!(uploads, vec![upload(1, 0), upload(2, 1)]);
        assert_eq!(residency.slot_of(FaceId(2)), Some(1));
    }

    #[test]
    fn should_upload_nothing_when_the_same_faces_are_shown_again() {
        let mut residency = Residency::with_capacity(4);
        residency.sync([FaceId(1), FaceId(2)]);

        let uploads = residency.sync([FaceId(2), FaceId(1)]);

        assert_eq!(uploads, vec![]);
        assert_eq!(residency.slot_of(FaceId(1)), Some(0));
        assert_eq!(residency.slot_of(FaceId(2)), Some(1));
    }

    #[test]
    fn should_reuse_the_slot_of_a_face_that_has_drifted_out_of_the_window() {
        let mut residency = Residency::with_capacity(4);
        residency.sync([FaceId(1), FaceId(2)]);

        let uploads = residency.sync([FaceId(2), FaceId(3)]);

        assert_eq!(uploads, vec![upload(3, 0)]);
        assert_eq!(residency.slot_of(FaceId(1)), None);
        assert_eq!(residency.slot_of(FaceId(3)), Some(0));
    }

    #[test]
    fn should_evict_everything_when_the_window_empties() {
        let mut residency = Residency::with_capacity(4);
        residency.sync([FaceId(1), FaceId(2)]);

        assert_eq!(residency.sync([]), vec![]);
        assert_eq!(residency.slot_of(FaceId(1)), None);
    }

    #[test]
    fn should_drop_faces_that_do_not_fit_when_more_arrive_than_there_are_slots() {
        let mut residency = Residency::with_capacity(2);

        let uploads = residency.sync([FaceId(1), FaceId(2), FaceId(3)]);

        assert_eq!(uploads, vec![upload(1, 0), upload(2, 1)]);
        assert_eq!(residency.slot_of(FaceId(3)), None);
    }
}
