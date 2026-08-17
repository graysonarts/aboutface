//! Ways the wall fails to come up.

use std::path::{Path, PathBuf};

/// Ways the wall fails to come up, or fails to keep drawing.
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    /// The windowing system refused an event loop or a window.
    #[error("cannot open a window: {0}")]
    Window(#[from] winit::error::OsError),

    /// The event loop itself could not be created or run.
    #[error("event loop failed: {0}")]
    EventLoop(#[from] winit::error::EventLoopError),

    /// No GPU adapter this build can draw on.
    #[error("no usable GPU adapter: {0}")]
    Adapter(#[from] wgpu::RequestAdapterError),

    /// An adapter was found but would not hand over a device.
    #[error("cannot open the GPU device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),

    /// The window could not be turned into a drawing surface.
    #[error("cannot create a drawing surface: {0}")]
    Surface(#[from] wgpu::CreateSurfaceError),

    /// The adapter and the window disagree about what can be drawn on.
    #[error("the GPU adapter supports no surface format this window can present")]
    UnsupportedSurface,

    /// The Grid asks for more portraits than this GPU holds in one array.
    #[error("a grid of {cells} cells needs {cells} texture layers; this GPU allows {supported}")]
    GridTooLarge {
        /// Cells the Grid was asked for.
        cells: u32,
        /// Texture array layers the adapter supports.
        supported: u32,
    },

    /// A display crop could not be read.
    #[error("cannot read the display crop {path}: {source}")]
    Image {
        /// The image that could not be read.
        path: PathBuf,
        /// The underlying decode failure.
        source: image::ImageError,
    },
}

impl RenderError {
    pub(crate) fn image(path: impl AsRef<Path>, source: image::ImageError) -> Self {
        Self::Image {
            path: path.as_ref().to_path_buf(),
            source,
        }
    }
}
