//! The window and its event loop.

use std::sync::Arc;

use afcore::GridSpec;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::WindowId;

use crate::error::RenderError;
use crate::geometry::Framing;
use crate::gpu::WallGpu;
use crate::portrait::Portrait;

/// How the wall is put on screen.
#[derive(Debug, Clone, PartialEq)]
pub struct WallSpec {
    grid: GridSpec,
    framing: Framing,
    background: wgpu::Color,
    title: String,
}

impl WallSpec {
    /// A wall showing `grid`, framed the house way.
    pub fn new(grid: GridSpec) -> Self {
        Self {
            grid,
            framing: Framing::default(),
            // Nearly black: a gallery wall, not a desktop.
            background: wgpu::Color {
                r: 0.05,
                g: 0.05,
                b: 0.06,
                a: 1.0,
            },
            title: String::from("About:Face"),
        }
    }

    /// Frames the Cells differently.
    ///
    /// How tightly a portrait is framed is meant to be settled by eye on the
    /// wall, so it is the one thing here a caller can change.
    pub fn with_framing(mut self, framing: Framing) -> Self {
        self.framing = framing;
        self
    }
}

/// Shows `portraits` on the wall and returns when the window closes.
///
/// Faces beyond the Grid's Cell count stay in the Corpus: the Grid is a Window
/// onto it (ADR-0004). Cells beyond the portrait count stay empty.
///
/// # Errors
///
/// Returns an error if the window, the GPU device or a display crop is
/// unavailable.
pub fn show(spec: WallSpec, portraits: Vec<Portrait>) -> Result<(), RenderError> {
    let event_loop = EventLoop::new()?;
    // Each presented frame asks for the next, so the loop sleeps between them
    // rather than spinning: the display's own rate paces the wall.
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut wall = WallApp {
        spec,
        portraits,
        gpu: None,
        failure: None,
    };
    event_loop.run_app(&mut wall)?;

    match wall.failure {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// The event loop's view of the wall.
struct WallApp {
    spec: WallSpec,
    portraits: Vec<Portrait>,
    gpu: Option<WallGpu>,
    /// A failure inside a callback, which cannot return one.
    failure: Option<RenderError>,
}

impl WallApp {
    fn start(&mut self, event_loop: &ActiveEventLoop) -> Result<(), RenderError> {
        let attributes = winit::window::Window::default_attributes()
            .with_title(self.spec.title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 800.0));
        let window = Arc::new(event_loop.create_window(attributes)?);

        self.gpu = Some(WallGpu::new(
            window,
            self.spec.grid,
            self.spec.framing,
            self.spec.background,
            &self.portraits,
        )?);

        Ok(())
    }
}

impl ApplicationHandler for WallApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }

        match self.start(event_loop) {
            // Nothing draws itself: the first frame is asked for here, and
            // every frame after it is asked for by the one before.
            Ok(()) => {
                if let Some(gpu) = self.gpu.as_ref() {
                    gpu.request_redraw();
                }
            }
            Err(error) => {
                self.failure = Some(error);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                gpu.resize(size.width, size.height);
                gpu.request_redraw();
            }
            WindowEvent::RedrawRequested => gpu.render(),
            _ => {}
        }
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        // The device, the surface, the portrait texture array and the window go
        // here, before the process does: an installation that quits and
        // restarts must not leak a display's worth of textures each time.
        self.gpu = None;
    }
}
