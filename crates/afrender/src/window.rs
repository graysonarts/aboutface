//! Which Faces the Grid is currently showing.
//!
//! The Corpus grows without bound and the Grid does not, so the Grid is a
//! Window onto it (ADR-0004). Stage 1 has no Drift: the Window is the first
//! Cell-count Faces it is offered, in whatever order they arrive.

use afcore::{Assignment, CellIndex, FaceId, GridSpec};

/// The Faces on screen, one per Cell.
///
/// Built on [`afcore::Assignment`], so a Face occupying two Cells is refused by
/// construction rather than by the caller remembering not to ask. Faces beyond
/// the Cell count stay in the Corpus, unseen; Cells beyond the Face count stay
/// empty.
#[derive(Debug, Clone, PartialEq)]
pub struct Window {
    assignment: Assignment,
}

impl Window {
    /// Opens a Window of `spec`'s size onto `faces`, filling Cells in order.
    ///
    /// Faces past the last Cell are not shown, and a Face offered twice is
    /// placed once — the wall never shows one Visitor's Face in two Cells.
    pub fn onto(spec: GridSpec, faces: impl IntoIterator<Item = FaceId>) -> Self {
        let mut assignment = Assignment::new(spec);
        let mut next = 0;

        for face in faces {
            if next >= spec.cell_count() {
                break;
            }
            // A repeat is skipped rather than propagated: the only error
            // `place` can raise here is the bijection this loop is enforcing.
            if assignment.place(CellIndex(next), face).is_ok() {
                next += 1;
            }
        }

        Self { assignment }
    }

    /// The Grid this Window fills.
    pub fn spec(&self) -> GridSpec {
        self.assignment.spec()
    }

    /// The Face in `cell`, if the Window reaches that far.
    pub fn face_at(&self, cell: CellIndex) -> Option<FaceId> {
        self.assignment.face_at(cell)
    }

    /// How many Cells hold a Face.
    pub fn occupied(&self) -> usize {
        self.assignment.occupied()
    }

    /// Every Face on screen, in Cell order.
    ///
    /// This is the set that needs a decoded texture, and nothing else does —
    /// the texture budget follows Grid size, not Corpus size (ADR-0004).
    pub fn resident(&self) -> impl Iterator<Item = FaceId> + '_ {
        (0..self.spec().cell_count()).filter_map(|cell| self.face_at(CellIndex(cell)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> GridSpec {
        GridSpec::new(4, 4).expect("valid grid")
    }

    fn faces(count: u64) -> Vec<FaceId> {
        (1..=count).map(FaceId).collect()
    }

    #[test]
    fn should_fill_cells_in_order_when_the_corpus_is_smaller_than_the_grid() {
        let window = Window::onto(spec(), faces(3));

        assert_eq!(window.face_at(CellIndex(0)), Some(FaceId(1)));
        assert_eq!(window.face_at(CellIndex(2)), Some(FaceId(3)));
        assert_eq!(window.occupied(), 3);
    }

    #[test]
    fn should_leave_cells_empty_when_the_corpus_is_smaller_than_the_grid() {
        let window = Window::onto(spec(), faces(3));

        assert_eq!(window.face_at(CellIndex(3)), None);
        assert_eq!(window.face_at(CellIndex(15)), None);
    }

    #[test]
    fn should_show_a_window_onto_the_corpus_when_it_holds_more_faces_than_cells() {
        let window = Window::onto(spec(), faces(100));

        assert_eq!(window.occupied(), 16);
        assert_eq!(window.face_at(CellIndex(15)), Some(FaceId(16)));
        assert_eq!(window.resident().count(), 16);
    }

    #[test]
    fn should_place_a_face_once_when_it_is_offered_twice() {
        let window = Window::onto(spec(), [FaceId(1), FaceId(1), FaceId(2)]);

        assert_eq!(window.face_at(CellIndex(0)), Some(FaceId(1)));
        assert_eq!(window.face_at(CellIndex(1)), Some(FaceId(2)));
        assert_eq!(window.occupied(), 2);
    }

    #[test]
    fn should_hold_no_faces_when_the_corpus_is_empty() {
        let window = Window::onto(spec(), []);

        assert_eq!(window.occupied(), 0);
        assert_eq!(window.resident().count(), 0);
    }
}
