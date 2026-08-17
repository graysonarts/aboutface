//! The Grid, its Cells, and the Assignment of Faces onto them.

use crate::FaceId;

/// Smallest Grid the piece is designed to show (ADR-0004).
pub const MIN_CELLS: u32 = 10;

/// Largest Grid the piece is designed to show (ADR-0004).
pub const MAX_CELLS: u32 = 1000;

/// Ways a [`GridSpec`] can be invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GridSpecError {
    /// Neither dimension may be zero.
    #[error("grid dimensions must be non-zero, got {cols}x{rows}")]
    ZeroDimension {
        /// Requested columns.
        cols: u32,
        /// Requested rows.
        rows: u32,
    },

    /// The Grid is outside the range the piece is designed for.
    #[error("grid of {cells} cells is outside the supported range {MIN_CELLS}..={MAX_CELLS}")]
    OutOfRange {
        /// Requested cell count.
        cells: u32,
    },
}

/// The dimensions of the Grid currently on screen.
///
/// The Grid is a Window onto a much larger Corpus, and its size is driven by the
/// piece's own clock rather than by headcount (ADR-0004).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridSpec {
    cols: u32,
    rows: u32,
}

impl GridSpec {
    /// Creates a Grid specification.
    ///
    /// # Errors
    ///
    /// Returns an error if either dimension is zero, or if the resulting cell
    /// count falls outside [`MIN_CELLS`]..=[`MAX_CELLS`].
    pub fn new(cols: u32, rows: u32) -> Result<Self, GridSpecError> {
        if cols == 0 || rows == 0 {
            return Err(GridSpecError::ZeroDimension { cols, rows });
        }

        let cells = cols
            .checked_mul(rows)
            .ok_or(GridSpecError::OutOfRange { cells: u32::MAX })?;

        if !(MIN_CELLS..=MAX_CELLS).contains(&cells) {
            return Err(GridSpecError::OutOfRange { cells });
        }

        Ok(Self { cols, rows })
    }

    /// Number of columns.
    pub fn cols(&self) -> u32 {
        self.cols
    }

    /// Number of rows.
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Total number of Cells, which is also the size of the Window.
    pub fn cell_count(&self) -> u32 {
        self.cols * self.rows
    }

    /// The column and row of `cell`, or `None` if it is outside this Grid.
    pub fn position_of(&self, cell: CellIndex) -> Option<(u32, u32)> {
        (cell.0 < self.cell_count()).then(|| (cell.0 % self.cols, cell.0 / self.cols))
    }
}

/// One position in the Grid. A Cell holds exactly one Face — never a cluster,
/// never a stack (`CONTEXT.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CellIndex(pub u32);

/// Ways an [`Assignment`] can be violated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AssignmentError {
    /// The Cell does not exist in this Grid.
    #[error("cell {cell} is outside a grid of {cells} cells")]
    CellOutOfRange {
        /// The offending cell index.
        cell: u32,
        /// Size of the Grid.
        cells: u32,
    },

    /// Placement is bijective: a Face may occupy at most one Cell.
    #[error("{face} is already placed in cell {existing}")]
    FaceAlreadyPlaced {
        /// The Face that is already on the Grid.
        face: FaceId,
        /// Where it already sits.
        existing: u32,
    },
}

/// A bijective placement of Faces onto Grid Cells.
///
/// Ordering comes from the self-organizing map; placement comes from solving a
/// linear assignment problem over this structure (ADR-0003).
#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    spec: GridSpec,
    cells: Vec<Option<FaceId>>,
}

impl Assignment {
    /// Creates an empty Assignment for `spec`.
    pub fn new(spec: GridSpec) -> Self {
        Self {
            spec,
            cells: vec![None; spec.cell_count() as usize],
        }
    }

    /// The Grid this Assignment places Faces onto.
    pub fn spec(&self) -> GridSpec {
        self.spec
    }

    /// Places `face` in `cell`, replacing whatever Face was there.
    ///
    /// # Errors
    ///
    /// Returns an error if `cell` is outside the Grid, or if `face` already
    /// occupies a different Cell — a Face may appear on the wall only once.
    pub fn place(&mut self, cell: CellIndex, face: FaceId) -> Result<(), AssignmentError> {
        let cells = self.spec.cell_count();
        if cell.0 >= cells {
            return Err(AssignmentError::CellOutOfRange {
                cell: cell.0,
                cells,
            });
        }

        if let Some(existing) = self.cell_of(face)
            && existing != cell
        {
            return Err(AssignmentError::FaceAlreadyPlaced {
                face,
                existing: existing.0,
            });
        }

        self.cells[cell.0 as usize] = Some(face);
        Ok(())
    }

    /// The Face in `cell`, if any.
    pub fn face_at(&self, cell: CellIndex) -> Option<FaceId> {
        self.cells.get(cell.0 as usize).copied().flatten()
    }

    /// The Cell `face` occupies, if it is on the Grid.
    pub fn cell_of(&self, face: FaceId) -> Option<CellIndex> {
        self.cells
            .iter()
            .position(|occupant| *occupant == Some(face))
            .map(|index| CellIndex(index as u32))
    }

    /// Number of occupied Cells.
    pub fn occupied(&self) -> usize {
        self.cells.iter().flatten().count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> GridSpec {
        GridSpec::new(4, 4).expect("valid grid")
    }

    #[test]
    fn accepts_grids_across_the_designed_range() {
        assert_eq!(
            GridSpec::new(5, 2).expect("valid grid").cell_count(),
            MIN_CELLS
        );
        assert_eq!(
            GridSpec::new(40, 25).expect("valid grid").cell_count(),
            MAX_CELLS
        );
    }

    #[test]
    fn rejects_grids_outside_the_designed_range() {
        assert_eq!(
            GridSpec::new(3, 3),
            Err(GridSpecError::OutOfRange { cells: 9 })
        );
        assert_eq!(
            GridSpec::new(41, 25),
            Err(GridSpecError::OutOfRange { cells: 1025 })
        );
    }

    #[test]
    fn rejects_zero_dimensions() {
        assert_eq!(
            GridSpec::new(0, 10),
            Err(GridSpecError::ZeroDimension { cols: 0, rows: 10 })
        );
    }

    #[test]
    fn maps_cells_to_positions_in_row_major_order() {
        let spec = spec();
        assert_eq!(spec.position_of(CellIndex(0)), Some((0, 0)));
        assert_eq!(spec.position_of(CellIndex(5)), Some((1, 1)));
        assert_eq!(spec.position_of(CellIndex(15)), Some((3, 3)));
        assert_eq!(spec.position_of(CellIndex(16)), None);
    }

    #[test]
    fn places_and_finds_a_face() {
        let mut assignment = Assignment::new(spec());
        let face = FaceId(7);

        assignment.place(CellIndex(3), face).expect("placed");

        assert_eq!(assignment.face_at(CellIndex(3)), Some(face));
        assert_eq!(assignment.cell_of(face), Some(CellIndex(3)));
        assert_eq!(assignment.occupied(), 1);
    }

    #[test]
    fn refuses_to_place_one_face_in_two_cells() {
        let mut assignment = Assignment::new(spec());
        let face = FaceId(7);
        assignment.place(CellIndex(3), face).expect("placed");

        assert_eq!(
            assignment.place(CellIndex(4), face),
            Err(AssignmentError::FaceAlreadyPlaced { face, existing: 3 })
        );
    }

    #[test]
    fn replacing_a_face_in_its_own_cell_is_allowed() {
        let mut assignment = Assignment::new(spec());
        let face = FaceId(7);
        assignment.place(CellIndex(3), face).expect("placed");

        assignment.place(CellIndex(3), face).expect("still placed");
        assert_eq!(assignment.occupied(), 1);
    }

    #[test]
    fn rejects_cells_outside_the_grid() {
        let mut assignment = Assignment::new(spec());

        assert_eq!(
            assignment.place(CellIndex(16), FaceId(1)),
            Err(AssignmentError::CellOutOfRange {
                cell: 16,
                cells: 16
            })
        );
    }
}
