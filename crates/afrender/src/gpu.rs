//! The device, the pipeline, and the portrait texture array.
//!
//! Everything here needs a GPU, which is why the Grid's arithmetic, the
//! Window's membership and the slot policy live in their own modules and are
//! tested without one.

use std::collections::HashMap;
use std::sync::Arc;

use afcore::{CellIndex, FaceId, GridSpec};
use wgpu::CurrentSurfaceTexture;
use winit::window::Window as OsWindow;

use crate::error::RenderError;
use crate::geometry::{Framing, Layout};
use crate::portrait::{Portrait, SLOT_HEIGHT, SLOT_WIDTH};
use crate::residency::Residency;
use crate::window::Window;

/// One Cell's quad, as the vertex shader wants it.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CellInstance {
    /// Centre of the quad in clip space.
    centre: [f32; 2],
    /// Half its width and height in clip space.
    half_size: [f32; 2],
    /// The texture array layer holding this Cell's portrait.
    slot: u32,
}

/// Texels a portrait occupies, used to size the staging write.
const BYTES_PER_TEXEL: u32 = 4;

/// The wall on a device: one texture array, one pipeline, one instance buffer.
pub(crate) struct WallGpu {
    // The surface borrows the window, so the window outlives it by being held
    // here too — dropping this struct releases both, in that order.
    surface: wgpu::Surface<'static>,
    os_window: Arc<OsWindow>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    configuration: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    portraits: wgpu::Texture,
    instances: wgpu::Buffer,
    drawn: u32,
    background: wgpu::Color,
    framing: Framing,
    layout: Layout,
    window: Window,
    residency: Residency,
}

impl WallGpu {
    /// Brings the wall up on `window`, showing `portraits` in `spec`'s Grid.
    ///
    /// # Errors
    ///
    /// Returns an error if no adapter or device is available, the surface
    /// cannot be created, or a display crop cannot be decoded.
    pub(crate) fn new(
        window: Arc<OsWindow>,
        spec: GridSpec,
        framing: Framing,
        background: wgpu::Color,
        portraits: &[Portrait],
    ) -> Result<Self, RenderError> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(Arc::clone(&window))?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))?;

        // Downlevel defaults keep the CPU-class GPUs ADR-0006 leaves in play
        // viable, but they cap a texture array at 256 layers and the Grid runs
        // to `afcore::MAX_CELLS`. One slot per Cell is the whole texture budget
        // (ADR-0004), so those layers are asked for by name.
        let cells = spec.cell_count();
        let supported = adapter.limits().max_texture_array_layers;
        if cells > supported {
            return Err(RenderError::GridTooLarge { cells, supported });
        }
        let mut required_limits =
            wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
        required_limits.max_texture_array_layers =
            cells.max(required_limits.max_texture_array_layers);

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("about:face wall"),
                required_limits,
                ..Default::default()
            }))?;

        let configuration = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or(RenderError::UnsupportedSurface)?;
        surface.configure(&device, &configuration);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("portraits"),
            size: wgpu::Extent3d {
                width: SLOT_WIDTH,
                height: SLOT_HEIGHT,
                // One slot per Cell and no more: the texture budget follows
                // Grid size, not Corpus size (ADR-0004).
                depth_or_array_layers: cells,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("portraits"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("portraits"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("portraits"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("wall.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("wall"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("wall"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<CellInstance>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x2,
                        1 => Float32x2,
                        2 => Uint32,
                    ],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: configuration.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cells"),
            size: (cells as usize * size_of::<CellInstance>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut wall = Self {
            surface,
            os_window: window,
            device,
            queue,
            configuration,
            pipeline,
            bind_group,
            portraits: texture,
            instances,
            drawn: 0,
            background,
            framing,
            layout: Layout::new(spec, size.width, size.height, framing),
            window: Window::onto(spec, std::iter::empty()),
            residency: Residency::with_capacity(cells),
        };

        wall.set_portraits(portraits)?;

        Ok(wall)
    }

    /// Shows a different set of portraits in the same Grid.
    ///
    /// A Face that was already on the wall keeps its slot and is not decoded
    /// again, so a Capture costs one upload rather than a Grid's worth — which
    /// is the same property Drift will rely on (ADR-0004).
    ///
    /// # Errors
    ///
    /// Returns an error if a display crop cannot be decoded.
    pub(crate) fn set_portraits(&mut self, portraits: &[Portrait]) -> Result<(), RenderError> {
        let sources: HashMap<FaceId, &Portrait> = portraits
            .iter()
            .map(|portrait| (portrait.face(), portrait))
            .collect();

        self.window = Window::onto(self.layout.spec(), portraits.iter().map(Portrait::face));

        let resident: Vec<FaceId> = self.window.resident().collect();
        for upload in self.residency.sync(resident) {
            // INVARIANT: the Window was built from these portraits, so every
            // resident Face has one.
            self.upload(upload.slot, sources[&upload.face])?;
        }
        self.rebuild_instances();

        Ok(())
    }

    /// Asks the windowing system for a frame.
    pub(crate) fn request_redraw(&self) {
        self.os_window.request_redraw();
    }

    /// Re-lays the Grid for a new surface size.
    ///
    /// Textures are untouched: a resize moves and rescales quads, it does not
    /// re-decode a single portrait.
    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.configuration.width = width;
        self.configuration.height = height;
        self.surface.configure(&self.device, &self.configuration);
        self.layout = Layout::new(self.layout.spec(), width, height, self.framing);
        self.rebuild_instances();
    }

    /// Draws one frame.
    pub(crate) fn render(&mut self) {
        // Whatever happens below, the next frame is asked for: a wall that
        // stops drawing because one frame was occluded stays stopped until some
        // unrelated event wakes it, and nobody is at the keyboard.
        self.request_redraw();

        let frame = match self.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(frame) | CurrentSurfaceTexture::Suboptimal(frame) => {
                frame
            }
            // A lost or outdated surface is what a display change looks like;
            // reconfiguring and skipping this frame is the whole recovery.
            CurrentSurfaceTexture::Lost | CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.configuration);
                return;
            }
            // Occluded, timed out, or refused: there is nothing to draw on this
            // frame and the next redraw request tries again.
            _ => return,
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wall"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("wall"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.background),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            if self.drawn > 0 {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_vertex_buffer(0, self.instances.slice(..));
                pass.draw(0..6, 0..self.drawn);
            }
        }

        self.queue.submit([encoder.finish()]);
        // Presentation paces the loop: the wall draws continuously at display
        // rate, which changes nothing on a still Stage 1 Grid but is what
        // everything from Stage 2 onward animates against.
        self.queue.present(frame);
    }

    /// Writes one portrait into its slot.
    fn upload(&self, slot: u32, portrait: &Portrait) -> Result<(), RenderError> {
        let texels = portrait.decode()?;

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.portraits,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: slot,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &texels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SLOT_WIDTH * BYTES_PER_TEXEL),
                rows_per_image: Some(SLOT_HEIGHT),
            },
            wgpu::Extent3d {
                width: SLOT_WIDTH,
                height: SLOT_HEIGHT,
                depth_or_array_layers: 1,
            },
        );

        Ok(())
    }

    /// Rewrites the instance buffer from the current Layout and Window.
    fn rebuild_instances(&mut self) {
        let (width, height) = (
            self.configuration.width as f32,
            self.configuration.height as f32,
        );
        let mut instances = Vec::with_capacity(self.window.occupied());

        for cell in 0..self.layout.spec().cell_count() {
            let cell = CellIndex(cell);
            let (Some(face), Some(rect)) = (self.window.face_at(cell), self.layout.rect_of(cell))
            else {
                continue;
            };
            let Some(slot) = self.residency.slot_of(face) else {
                continue;
            };

            // Surface pixels, top-left origin, into clip space, centre origin.
            instances.push(CellInstance {
                centre: [
                    (rect.x + rect.width / 2.0) / width * 2.0 - 1.0,
                    1.0 - (rect.y + rect.height / 2.0) / height * 2.0,
                ],
                half_size: [rect.width / width, rect.height / height],
                slot,
            });
        }

        self.drawn = instances.len() as u32;
        if self.drawn > 0 {
            self.queue
                .write_buffer(&self.instances, 0, bytemuck::cast_slice(&instances));
        }
    }
}

/// The GPU adapter the wall would draw on, and the backend it would use.
///
/// The startup self-check reports it, because ADR-0006 leaves the hardware
/// open and "which backend did I actually get" is a thing the operator must be
/// able to compare between candidate machines. `None` means this build found no
/// adapter at all and [`show`](crate::show) would fail.
/// Asked for the same way the wall asks, minus the surface — there is
/// no window yet at self-check time — so the answer is the adapter the wall
/// will get unless the surface itself is the thing that fails.
pub fn adapter_report() -> Option<String> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        compatible_surface: None,
        ..Default::default()
    }))
    .ok()?;

    let info = adapter.get_info();
    Some(format!(
        "{} ({:?}, {:?})",
        info.name, info.backend, info.device_type
    ))
}
