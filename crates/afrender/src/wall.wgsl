// One instanced quad per occupied Cell.
//
// The quad's corners are generated from the vertex index rather than read from
// a buffer: every Cell is the same rectangle, and the only thing that varies is
// where it sits and which layer of the portrait array it samples.

struct Cell {
    @location(0) centre: vec2<f32>,
    @location(1) half_size: vec2<f32>,
    @location(2) slot: u32,
};

struct Fragment {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) slot: u32,
};

// Two triangles, counter-clockwise, in the quad's own -1..1 space.
const CORNERS = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, 1.0),
);

@group(0) @binding(0) var portraits: texture_2d_array<f32>;
@group(0) @binding(1) var portrait_sampler: sampler;

@vertex
fn vertex(@builtin(vertex_index) index: u32, cell: Cell) -> Fragment {
    let corner = CORNERS[index];

    var out: Fragment;
    out.position = vec4<f32>(cell.centre + corner * cell.half_size, 0.0, 1.0);
    // Texture space runs downwards; clip space runs upwards.
    out.uv = vec2<f32>(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
    out.slot = cell.slot;
    return out;
}

@fragment
fn fragment(in: Fragment) -> @location(0) vec4<f32> {
    return textureSample(portraits, portrait_sampler, in.uv, in.slot);
}
