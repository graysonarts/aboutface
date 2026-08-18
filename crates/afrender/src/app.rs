//! The window and its event loop.

use std::sync::Arc;

use afcore::GridSpec;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
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
    // A wall with nothing behind it answers the Shutter with what it already
    // shows: pressing the key on a Corpus nobody is capturing into changes
    // nothing rather than emptying the Grid.
    let unchanged = portraits.clone();
    show_live(spec, portraits, move || unchanged.clone())
}

/// Shows `portraits`, and asks `on_shutter` for the wall again whenever the
/// Shutter is pressed.
///
/// The wall knows nothing about cameras, models or the Corpus: a press is a
/// question, and the answer is the Window the caller now wants shown. A caller
/// that took no Capture — no face in the frame — answers with the Faces it
/// already had, and nothing on screen moves.
///
/// The callback runs on the render thread, so the wall stops drawing for as
/// long as a Capture takes. That is visible and deliberate at Stage 1: the
/// per-Capture cost is what ADR-0006 is measuring.
///
/// # Errors
///
/// Returns an error if the window, the GPU device or a display crop is
/// unavailable.
pub fn show_live(
    spec: WallSpec,
    portraits: Vec<Portrait>,
    on_shutter: impl FnMut() -> Vec<Portrait>,
) -> Result<(), RenderError> {
    let event_loop = EventLoop::new()?;
    // Each presented frame asks for the next, so the loop sleeps between them
    // rather than spinning: the display's own rate paces the wall.
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut wall = WallApp {
        spec,
        portraits,
        on_shutter,
        gpu: None,
        failure: None,
    };
    event_loop.run_app(&mut wall)?;

    match wall.failure {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// What a key press asks the wall to do.
///
/// Internal: the wall acts on these itself, and a caller that wanted to bind
/// its own keys would need the event loop, which this crate owns.
///
/// The keyboard is a placeholder for the booth's physical control (ADR-0005
/// leaves the Shutter's form open); naming the intent rather than the key is
/// what lets that control replace it without the wall changing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WallInput {
    /// Take one Capture.
    Shutter,

    /// Close the wall.
    Quit,
}

/// What a physical key means to the wall, if anything.
fn input_for(key: PhysicalKey) -> Option<WallInput> {
    match key {
        PhysicalKey::Code(KeyCode::Space) => Some(WallInput::Shutter),
        PhysicalKey::Code(KeyCode::Escape) => Some(WallInput::Quit),
        _ => None,
    }
}

/// The event loop's view of the wall.
struct WallApp<F: FnMut() -> Vec<Portrait>> {
    spec: WallSpec,
    portraits: Vec<Portrait>,
    /// What the piece does with a Shutter press.
    on_shutter: F,
    gpu: Option<WallGpu>,
    /// A failure inside a callback, which cannot return one.
    failure: Option<RenderError>,
}

impl<F: FnMut() -> Vec<Portrait>> WallApp<F> {
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

impl<F: FnMut() -> Vec<Portrait>> ApplicationHandler for WallApp<F> {
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
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key,
                        state: ElementState::Pressed,
                        // A held key is one gesture, not a Capture per repeat.
                        repeat: false,
                        ..
                    },
                ..
            } => match input_for(physical_key) {
                Some(WallInput::Shutter) => {
                    self.portraits = (self.on_shutter)();
                    if let Err(error) = gpu.set_portraits(&self.portraits) {
                        self.failure = Some(error);
                        event_loop.exit();
                        return;
                    }
                    gpu.request_redraw();
                }
                Some(WallInput::Quit) => event_loop.exit(),
                None => {}
            },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_take_a_capture_when_the_shutter_key_is_pressed() {
        assert_eq!(
            input_for(PhysicalKey::Code(KeyCode::Space)),
            Some(WallInput::Shutter)
        );
    }

    #[test]
    fn should_ignore_enter_so_the_shutter_is_the_one_key_the_booth_names() {
        // The startup banner says SPACE. A second, undocumented Shutter key is
        // a Capture a Visitor did not mean to take.
        assert_eq!(input_for(PhysicalKey::Code(KeyCode::Enter)), None);
    }

    #[test]
    fn should_close_the_wall_when_escape_is_pressed() {
        assert_eq!(
            input_for(PhysicalKey::Code(KeyCode::Escape)),
            Some(WallInput::Quit)
        );
    }

    #[test]
    fn should_ignore_a_key_the_wall_has_no_meaning_for() {
        // A Visitor leaning on the keyboard must not be able to trigger a
        // Capture by accident.
        assert_eq!(input_for(PhysicalKey::Code(KeyCode::KeyQ)), None);
    }
}
