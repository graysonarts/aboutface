//! Where each Cell sits on the surface.
//!
//! Pure arithmetic over a [`GridSpec`] and a surface size, with no GPU and no
//! window: the wall is re-laid on every resize, and a resize must not touch a
//! single texture. Keeping the arithmetic here is what makes that testable on a
//! machine with no display.

use afcore::{CellIndex, GridSpec};

/// One Cell's rectangle in surface pixels, measured from the top-left corner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellRect {
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width.
    pub width: f32,
    /// Height.
    pub height: f32,
}

/// How the Grid is framed inside the surface.
///
/// The portrait aspect belongs to the display crop `afvision` produced; the
/// Grid does not restretch a Visitor's face to fill a Cell, so a surface whose
/// shape disagrees with the Grid's gets letterboxed instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Framing {
    aspect_ratio: f32,
    gutter: f32,
}

impl Framing {
    /// Frames Cells at `aspect_ratio` (width over height) with `gutter` of
    /// empty space between them, as a fraction of a Cell's width.
    ///
    /// Both values are clamped to something drawable rather than refused: a
    /// nonsensical framing is a configuration mistake, and the wall staying up
    /// with square Cells is better than the wall going dark.
    pub fn new(aspect_ratio: f32, gutter: f32) -> Self {
        Self {
            aspect_ratio: if aspect_ratio.is_finite() && aspect_ratio > 0.0 {
                aspect_ratio
            } else {
                1.0
            },
            gutter: if gutter.is_finite() && gutter >= 0.0 {
                gutter
            } else {
                0.0
            },
        }
    }

    /// Width over height of one Cell.
    pub fn aspect_ratio(&self) -> f32 {
        self.aspect_ratio
    }

    /// Space between Cells, as a fraction of a Cell's width.
    pub fn gutter(&self) -> f32 {
        self.gutter
    }
}

impl Default for Framing {
    /// The 4:5 portrait `booth.toml` frames display crops at, with a gutter
    /// wide enough to read as separate portraits rather than one sheet.
    fn default() -> Self {
        Self::new(0.8, 0.08)
    }
}

/// The Grid laid out on a surface of a given size.
///
/// Rebuilt on every resize. Cell geometry and texture residency are
/// deliberately separate: nothing here knows an image exists.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Layout {
    spec: GridSpec,
    cell_width: f32,
    cell_height: f32,
    gutter: f32,
    origin_x: f32,
    origin_y: f32,
}

impl Layout {
    /// Lays `spec` out on a surface `width` by `height` pixels, centred.
    ///
    /// A zero-sized surface — which is what a minimised window reports — yields
    /// zero-sized Cells rather than an error; there is nothing to draw, and the
    /// next resize lays the Grid out again.
    pub fn new(spec: GridSpec, width: u32, height: u32, framing: Framing) -> Self {
        let (cols, rows) = (spec.cols() as f32, spec.rows() as f32);
        let gutter = framing.gutter();

        // One unknown, two constraints: the Cell width at which the Grid plus
        // its gutters exactly fills the surface horizontally, and the one at
        // which it fills it vertically. The smaller fits both.
        let by_width = width as f32 / (cols + gutter * (cols + 1.0));
        let by_height = height as f32 / (rows / framing.aspect_ratio() + gutter * (rows + 1.0));
        let cell_width = by_width.min(by_height).max(0.0);
        let cell_height = cell_width / framing.aspect_ratio();

        let gutter = gutter * cell_width;
        let content_width = cols * cell_width + (cols + 1.0) * gutter;
        let content_height = rows * cell_height + (rows + 1.0) * gutter;

        Self {
            spec,
            cell_width,
            cell_height,
            gutter,
            origin_x: (width as f32 - content_width) / 2.0,
            origin_y: (height as f32 - content_height) / 2.0,
        }
    }

    /// The Grid this layout places.
    pub fn spec(&self) -> GridSpec {
        self.spec
    }

    /// Where `cell` sits, or `None` if it is outside this Grid.
    pub fn rect_of(&self, cell: CellIndex) -> Option<CellRect> {
        let (col, row) = self.spec.position_of(cell)?;

        Some(CellRect {
            x: self.origin_x + self.gutter + col as f32 * (self.cell_width + self.gutter),
            y: self.origin_y + self.gutter + row as f32 * (self.cell_height + self.gutter),
            width: self.cell_width,
            height: self.cell_height,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> GridSpec {
        GridSpec::new(4, 4).expect("valid grid")
    }

    fn layout(width: u32, height: u32) -> Layout {
        Layout::new(spec(), width, height, Framing::new(1.0, 0.0))
    }

    fn rect(layout: &Layout, cell: u32) -> CellRect {
        layout
            .rect_of(CellIndex(cell))
            .expect("cell is in the grid")
    }

    #[test]
    fn should_give_every_cell_a_rectangle_when_the_grid_is_laid_out() {
        let layout = layout(400, 400);

        let placed = (0..16)
            .filter(|cell| layout.rect_of(CellIndex(*cell)).is_some())
            .count();

        assert_eq!(placed, 16);
    }

    #[test]
    fn should_fill_the_surface_when_it_matches_the_grid_shape() {
        let layout = layout(400, 400);

        assert_eq!(
            rect(&layout, 0),
            CellRect {
                x: 0.0,
                y: 0.0,
                width: 100.0,
                height: 100.0
            }
        );
        assert_eq!(
            rect(&layout, 15),
            CellRect {
                x: 300.0,
                y: 300.0,
                width: 100.0,
                height: 100.0
            }
        );
    }

    #[test]
    fn should_place_cells_in_row_major_order_when_the_grid_is_laid_out() {
        let layout = layout(400, 400);

        assert_eq!((rect(&layout, 1).x, rect(&layout, 1).y), (100.0, 0.0));
        assert_eq!((rect(&layout, 4).x, rect(&layout, 4).y), (0.0, 100.0));
    }

    #[test]
    fn should_centre_the_grid_when_the_surface_is_wider_than_it_needs() {
        let layout = layout(800, 400);

        // The Grid is still 400 wide, so 200 of slack sits on each side.
        assert_eq!(rect(&layout, 0).x, 200.0);
        assert_eq!(rect(&layout, 0).y, 0.0);
        assert_eq!(rect(&layout, 0).width, 100.0);
    }

    #[test]
    fn should_keep_the_cell_aspect_when_the_surface_disagrees_with_it() {
        let layout = Layout::new(spec(), 800, 800, Framing::new(0.5, 0.0));

        let cell = rect(&layout, 0);
        assert!(
            (cell.width / cell.height - 0.5).abs() < 1e-5,
            "cells are {}x{}",
            cell.width,
            cell.height
        );
        // Height binds first at this aspect: four rows of double-height Cells.
        assert_eq!(cell.height, 200.0);
    }

    #[test]
    fn should_separate_cells_when_the_framing_asks_for_a_gutter() {
        let layout = Layout::new(spec(), 500, 500, Framing::new(1.0, 0.25));

        // Four Cells and five gutters of a quarter-Cell each: 5.25 Cell widths.
        let cell = rect(&layout, 0);
        assert!((cell.width - 500.0 / 5.25).abs() < 1e-3, "{cell:?}");
        assert!((cell.x - cell.width * 0.25).abs() < 1e-3, "{cell:?}");
    }

    #[test]
    fn should_refuse_a_cell_outside_the_grid_when_asked_for_its_rectangle() {
        assert_eq!(layout(400, 400).rect_of(CellIndex(16)), None);
    }

    #[test]
    fn should_produce_empty_cells_when_the_window_is_minimised() {
        let layout = layout(0, 0);

        assert_eq!(rect(&layout, 0).width, 0.0);
        assert_eq!(rect(&layout, 0).height, 0.0);
    }

    #[test]
    fn should_fall_back_to_square_cells_when_the_framing_is_nonsense() {
        let framing = Framing::new(0.0, -1.0);

        assert_eq!(framing.aspect_ratio(), 1.0);
        assert_eq!(framing.gutter(), 0.0);
    }
}
