//! Press a key, one frame lands on disk.
//!
//! The keypress is a placeholder for the physical control a Visitor will
//! actually press (Stage 4); the vocabulary and the code path are the real
//! ones. It lives in `examples/` rather than in `afbooth` because wiring the
//! booth binary is a separate ticket.
//!
//! ```text
//! cargo run -p afcapture --example shutter_keypress            # real camera
//! cargo run -p afcapture --example shutter_keypress -- --fake  # no hardware
//! cargo run -p afcapture --example shutter_keypress -- captures/tonight
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use afcapture::testing::{FakeCamera, sample_path};
use afcapture::{Camera, CameraSelector, NokhwaCamera, Shutter};
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, read};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};

fn main() -> ExitCode {
    let (directory, fake) = parse_arguments();

    let camera: Box<dyn Camera> = if fake {
        Box::new(FakeCamera::replaying(
            ["1.jpg", "2.jpg", "3.jpg", "4.jpg"].map(sample_path),
        ))
    } else {
        Box::new(NokhwaCamera::new(CameraSelector::Default))
    };

    match run(Shutter::new(camera, directory)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(report) => {
            eprintln!("{report}");
            ExitCode::FAILURE
        }
    }
}

fn parse_arguments() -> (PathBuf, bool) {
    let mut directory = PathBuf::from("captures");
    let mut fake = false;

    for argument in std::env::args().skip(1) {
        if argument == "--fake" {
            fake = true;
        } else {
            directory = PathBuf::from(argument);
        }
    }

    (directory, fake)
}

fn run(mut shutter: Shutter<Box<dyn Camera>>) -> Result<(), String> {
    // Absent, busy, disconnected and permission-denied all arrive here as
    // distinct errors. Report them and leave; never panic in front of a
    // Visitor.
    shutter
        .open()
        .map_err(|error| describe("could not open the camera", &error))?;

    println!("camera: {}", shutter.camera().describe());
    println!("captures: {}", shutter.directory().display());
    println!("space or enter — capture     q or esc — quit");

    let outcome = loop_until_quit(&mut shutter);
    let closed = shutter
        .close()
        .map_err(|error| describe("could not release the camera", &error));

    outcome.and(closed)
}

fn loop_until_quit(shutter: &mut Shutter<Box<dyn Camera>>) -> Result<(), String> {
    let _raw = RawMode::enter()?;

    loop {
        let event = read().map_err(|error| format!("could not read the keyboard: {error}"))?;

        let Event::Key(KeyEvent { code, kind, .. }) = event else {
            continue;
        };
        // Windows reports press *and* release; one press must not be two
        // Captures.
        if kind != KeyEventKind::Press {
            continue;
        }

        match code {
            KeyCode::Char(' ') | KeyCode::Enter => match shutter.press() {
                Ok(capture) => print!(
                    "captured {}x{} -> {}\r\n",
                    capture.width,
                    capture.height,
                    capture.path.display()
                ),
                Err(error) => {
                    return Err(describe("the Capture failed", &error));
                }
            },
            KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
            _ => {}
        }
    }
}

/// Renders an error and everything underneath it, so the cause is not lost.
fn describe(context: &str, error: &dyn std::error::Error) -> String {
    let mut report = format!("{context}: {error}");
    let mut source = error.source();
    while let Some(cause) = source {
        report.push_str(&format!("\n  caused by: {cause}"));
        source = cause.source();
    }
    report
}

/// Restores the terminal however the loop exits.
struct RawMode;

impl RawMode {
    fn enter() -> Result<Self, String> {
        enable_raw_mode().map_err(|error| format!("could not read single keypresses: {error}"))?;
        Ok(Self)
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
    }
}
