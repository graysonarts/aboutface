//! Points, boxes and the similarity transform alignment is built on.
//!
//! All coordinates are pixels in the source image, as floats: YuNet's outputs
//! are continuous, and rounding them before the warp throws away precision the
//! alignment wants.

/// Ways a geometric construction can be impossible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GeometryError {
    /// A box with a non-positive extent encloses nothing.
    #[error("a bounding box must have a positive width and height")]
    EmptyBox,

    /// Every source point sits on top of every other, so no scale or rotation
    /// can be recovered from them.
    #[error("cannot estimate a transform: the source points are all coincident")]
    CoincidentPoints,
}

/// A point in image space, in pixels, with y increasing downwards.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// Horizontal position.
    pub x: f32,
    /// Vertical position.
    pub y: f32,
}

impl Point {
    /// A point at `(x, y)`.
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// An axis-aligned rectangle in image space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

impl BoundingBox {
    /// A box at `(x, y)` with the given extent.
    ///
    /// # Errors
    ///
    /// Returns [`GeometryError::EmptyBox`] if either extent is not positive.
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Result<Self, GeometryError> {
        if width <= 0.0 || height <= 0.0 {
            return Err(GeometryError::EmptyBox);
        }

        Ok(Self {
            x,
            y,
            width,
            height,
        })
    }

    /// Left edge.
    pub fn x(&self) -> f32 {
        self.x
    }

    /// Top edge.
    pub fn y(&self) -> f32 {
        self.y
    }

    /// Width in pixels.
    pub fn width(&self) -> f32 {
        self.width
    }

    /// Height in pixels.
    pub fn height(&self) -> f32 {
        self.height
    }

    /// Right edge.
    pub fn right(&self) -> f32 {
        self.x + self.width
    }

    /// Bottom edge.
    pub fn bottom(&self) -> f32 {
        self.y + self.height
    }

    /// The box's centre.
    pub fn center(&self) -> Point {
        Point::new(self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    /// Area in square pixels.
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Overlap with `other`, as intersection area over union area.
    ///
    /// Zero when the boxes are disjoint, one when they coincide. This is the
    /// measure non-maximum suppression collapses duplicate detections with.
    pub fn intersection_over_union(&self, other: &Self) -> f32 {
        let width = (self.right().min(other.right()) - self.x.max(other.x)).max(0.0);
        let height = (self.bottom().min(other.bottom()) - self.y.max(other.y)).max(0.0);
        let intersection = width * height;
        let union = self.area() + other.area() - intersection;

        if union <= 0.0 {
            0.0
        } else {
            intersection / union
        }
    }
}

/// The five landmarks YuNet emits, in the order it emits them.
///
/// YuNet is chosen precisely because detection hands back these five points in
/// one step, which is exactly what the alignment warp needs (ADR-0001). "Left"
/// and "right" are the image's, not the Visitor's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Landmarks {
    /// Eye on the image's left.
    pub left_eye: Point,
    /// Eye on the image's right.
    pub right_eye: Point,
    /// Nose tip.
    pub nose: Point,
    /// Mouth corner on the image's left.
    pub mouth_left: Point,
    /// Mouth corner on the image's right.
    pub mouth_right: Point,
}

impl Landmarks {
    /// The five points in YuNet's order.
    pub fn as_array(&self) -> [Point; 5] {
        [
            self.left_eye,
            self.right_eye,
            self.nose,
            self.mouth_left,
            self.mouth_right,
        ]
    }

    /// Builds landmarks from five points in YuNet's order.
    pub fn from_array(points: [Point; 5]) -> Self {
        Self {
            left_eye: points[0],
            right_eye: points[1],
            nose: points[2],
            mouth_left: points[3],
            mouth_right: points[4],
        }
    }
}

/// A rotation, a uniform scale and a translation — and nothing else.
///
/// Deliberately not an affine transform: shear and non-uniform scale would
/// stretch a face to fit the template, and the aligned crop is meant to
/// normalise pose, not to reshape the person.
///
/// Written out, the map is
/// `(x, y) ↦ (a·x − b·y + tx, b·x + a·y + ty)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimilarityTransform {
    a: f32,
    b: f32,
    tx: f32,
    ty: f32,
}

impl SimilarityTransform {
    /// Fits the similarity that best maps `from` onto `to`, in the
    /// least-squares sense.
    ///
    /// With more than two correspondences no similarity fits exactly, so the
    /// fit is a compromise across all five landmarks rather than a match of
    /// any two of them. Treating the pairs as complex numbers turns that
    /// least-squares problem into a single division, which is why the
    /// implementation is four sums long.
    ///
    /// # Errors
    ///
    /// Returns [`GeometryError::CoincidentPoints`] if the source points carry
    /// no spread for a scale and rotation to be read from.
    pub fn estimate<const N: usize>(
        from: &[Point; N],
        to: &[Point; N],
    ) -> Result<Self, GeometryError> {
        let count = N as f32;
        let from_center = centroid(from, count);
        let to_center = centroid(to, count);

        // c = Σ w·conj(z) / Σ |z|², with z the centred source and w the
        // centred target as complex numbers.
        let (mut real, mut imaginary, mut norm) = (0.0, 0.0, 0.0);
        for (source, target) in from.iter().zip(to) {
            let (zx, zy) = (source.x - from_center.x, source.y - from_center.y);
            let (wx, wy) = (target.x - to_center.x, target.y - to_center.y);
            real += wx * zx + wy * zy;
            imaginary += wy * zx - wx * zy;
            norm += zx * zx + zy * zy;
        }

        if norm <= f32::EPSILON {
            return Err(GeometryError::CoincidentPoints);
        }

        let a = real / norm;
        let b = imaginary / norm;

        Ok(Self {
            a,
            b,
            tx: to_center.x - (a * from_center.x - b * from_center.y),
            ty: to_center.y - (b * from_center.x + a * from_center.y),
        })
    }

    /// Maps a point through the transform.
    pub fn apply(&self, point: Point) -> Point {
        Point::new(
            self.a * point.x - self.b * point.y + self.tx,
            self.b * point.x + self.a * point.y + self.ty,
        )
    }

    /// The uniform scale factor.
    pub fn scale(&self) -> f32 {
        (self.a * self.a + self.b * self.b).sqrt()
    }

    /// The transform that undoes this one, or `None` if it collapses
    /// everything to a point.
    ///
    /// The warp samples the source through this inverse: a destination pixel
    /// asks where it came from, so every destination pixel gets a value.
    pub fn inverse(&self) -> Option<Self> {
        let determinant = self.a * self.a + self.b * self.b;
        if determinant <= f32::EPSILON {
            return None;
        }

        let (a, b) = (self.a / determinant, -self.b / determinant);

        Some(Self {
            a,
            b,
            tx: -(a * self.tx - b * self.ty),
            ty: -(b * self.tx + a * self.ty),
        })
    }
}

fn centroid<const N: usize>(points: &[Point; N], count: f32) -> Point {
    let (x, y) = points
        .iter()
        .fold((0.0, 0.0), |(x, y), point| (x + point.x, y + point.y));

    Point::new(x / count, y / count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f32, y: f32) -> Point {
        Point { x, y }
    }

    #[test]
    fn should_recover_a_quarter_turn_and_a_doubling_when_landmarks_are_transformed() {
        // Source square, and the same square rotated 90° anticlockwise in image
        // coordinates (y down), scaled by two, and moved by (10, 20). The
        // target coordinates are written out by hand so the assertion cannot
        // agree with the estimator by construction.
        let from = [
            point(0.0, 0.0),
            point(1.0, 0.0),
            point(1.0, 1.0),
            point(0.0, 1.0),
        ];
        let to = [
            point(10.0, 20.0),
            point(10.0, 22.0),
            point(8.0, 22.0),
            point(8.0, 20.0),
        ];

        let transform = SimilarityTransform::estimate(&from, &to).expect("non-degenerate points");

        assert!((transform.scale() - 2.0).abs() < 1e-4, "{transform:?}");
        for (source, expected) in from.iter().zip(to) {
            let mapped = transform.apply(*source);
            assert!(
                (mapped.x - expected.x).abs() < 1e-3 && (mapped.y - expected.y).abs() < 1e-3,
                "{source:?} mapped to {mapped:?}, expected {expected:?}"
            );
        }
    }

    #[test]
    fn should_estimate_the_identity_when_the_points_are_unmoved() {
        let points = [
            point(3.0, 4.0),
            point(-2.0, 7.0),
            point(11.0, 0.5),
            point(0.0, -6.0),
        ];

        let transform = SimilarityTransform::estimate(&points, &points).expect("distinct points");

        assert!((transform.scale() - 1.0).abs() < 1e-5);
        let mapped = transform.apply(point(5.0, -1.0));
        assert!(
            (mapped.x - 5.0).abs() < 1e-4 && (mapped.y + 1.0).abs() < 1e-4,
            "{mapped:?}"
        );
    }

    #[test]
    fn should_fit_the_best_compromise_when_the_points_do_not_match_exactly() {
        // Three points on a line, the middle one nudged: no similarity maps
        // these exactly, so the fit must land between them rather than fail.
        let from = [point(0.0, 0.0), point(1.0, 0.0), point(2.0, 0.0)];
        let to = [point(0.0, 0.0), point(1.0, 1.0), point(2.0, 0.0)];

        let transform = SimilarityTransform::estimate(&from, &to).expect("non-degenerate points");

        let middle = transform.apply(point(1.0, 0.0));
        assert!(
            (middle.x - 1.0).abs() < 1e-3 && (middle.y - 0.3333).abs() < 1e-3,
            "least-squares fit put the middle point at {middle:?}"
        );
    }

    #[test]
    fn should_reject_estimation_when_every_source_point_is_the_same() {
        let from = [point(2.0, 2.0); 3];
        let to = [point(0.0, 0.0), point(1.0, 0.0), point(0.0, 1.0)];

        assert_eq!(
            SimilarityTransform::estimate(&from, &to),
            Err(GeometryError::CoincidentPoints)
        );
    }

    #[test]
    fn should_invert_a_transform_when_it_is_not_degenerate() {
        let from = [point(0.0, 0.0), point(4.0, 0.0), point(4.0, 4.0)];
        let to = [point(1.0, 1.0), point(1.0, 9.0), point(-7.0, 9.0)];
        let transform = SimilarityTransform::estimate(&from, &to).expect("non-degenerate points");

        let inverse = transform.inverse().expect("invertible");

        let round_trip = inverse.apply(transform.apply(point(3.0, -2.0)));
        assert!(
            (round_trip.x - 3.0).abs() < 1e-3 && (round_trip.y + 2.0).abs() < 1e-3,
            "{round_trip:?}"
        );
    }

    #[test]
    fn should_expose_the_five_landmarks_in_yunet_order() {
        let landmarks = Landmarks {
            left_eye: point(1.0, 2.0),
            right_eye: point(3.0, 2.0),
            nose: point(2.0, 3.0),
            mouth_left: point(1.5, 4.0),
            mouth_right: point(2.5, 4.0),
        };

        assert_eq!(
            landmarks.as_array(),
            [
                point(1.0, 2.0),
                point(3.0, 2.0),
                point(2.0, 3.0),
                point(1.5, 4.0),
                point(2.5, 4.0),
            ]
        );
    }

    #[test]
    fn should_report_the_centre_when_a_box_is_built_from_its_corner_and_size() {
        let bbox = BoundingBox::new(10.0, 20.0, 30.0, 40.0).expect("positive extent");

        assert_eq!(bbox.center(), point(25.0, 40.0));
        assert_eq!((bbox.right(), bbox.bottom()), (40.0, 60.0));
    }

    #[test]
    fn should_reject_a_box_when_an_extent_is_not_positive() {
        assert_eq!(
            BoundingBox::new(0.0, 0.0, 0.0, 5.0),
            Err(GeometryError::EmptyBox)
        );
    }

    #[test]
    fn should_report_full_overlap_when_a_box_is_compared_with_itself() {
        let bbox = BoundingBox::new(1.0, 2.0, 3.0, 4.0).expect("positive extent");

        assert!((bbox.intersection_over_union(&bbox) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn should_report_no_overlap_when_boxes_are_disjoint() {
        let left = BoundingBox::new(0.0, 0.0, 10.0, 10.0).expect("positive extent");
        let right = BoundingBox::new(20.0, 0.0, 10.0, 10.0).expect("positive extent");

        assert_eq!(left.intersection_over_union(&right), 0.0);
    }

    #[test]
    fn should_report_a_third_when_boxes_share_half_their_area() {
        // Two 10x10 boxes offset by 5 in x: intersection 50, union 150.
        let left = BoundingBox::new(0.0, 0.0, 10.0, 10.0).expect("positive extent");
        let right = BoundingBox::new(5.0, 0.0, 10.0, 10.0).expect("positive extent");

        assert!((left.intersection_over_union(&right) - 1.0 / 3.0).abs() < 1e-6);
    }
}
