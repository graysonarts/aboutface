//! Point it at a photograph; it finds the face and writes both crops.
//!
//! No camera is involved: this runs entirely off a file on disk, which is how
//! the detection and alignment path is exercised before there is a booth to run
//! it in.
//!
//! ```text
//! cargo run -p afvision --example detect_face -- samples/1.jpg
//! cargo run -p afvision --example detect_face -- samples/2.jpg --out crops --margin 0.6 --aspect 1.0
//! ```
//!
//! Two files land in the output directory per photograph: `<name>-aligned.png`,
//! the 112×112 crop the embedder will see, and `<name>-display.png`, the
//! portrait the wall will show. They are not interchangeable — the aligned crop
//! is warped to a fixed landmark template and is not meant to be looked at.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use afvision::{
    ALIGNED_SIZE, DisplayCropSpec, FaceDetector, Faces, ModelRole, ModelSpec, align, display_crop,
    select_execution_provider,
};

fn main() -> ExitCode {
    let options = match Options::parse(std::env::args().skip(1)) {
        Ok(options) => options,
        Err(message) => {
            eprintln!("{message}\n\n{USAGE}");
            return ExitCode::FAILURE;
        }
    };

    match run(&options) {
        Ok(()) => ExitCode::SUCCESS,
        Err(report) => {
            eprintln!("{report}");
            ExitCode::FAILURE
        }
    }
}

const USAGE: &str = "usage: detect_face <photograph> [--out DIR] [--model FILE] \
                     [--margin F] [--aspect F] [--width N] [--bias F]";

struct Options {
    photograph: PathBuf,
    out: PathBuf,
    model: PathBuf,
    crop: DisplayCropSpec,
}

impl Options {
    fn parse(arguments: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut photograph = None;
        let mut out = PathBuf::from("crops");
        let mut model = PathBuf::from("models/face_detection_yunet_2023mar.onnx");
        // The house framing is `DisplayCropSpec`'s own default, so this tool
        // and `booth.toml` cannot drift apart; the flags only override it.
        let house = DisplayCropSpec::default();
        let (mut margin, mut aspect, mut width, mut bias) = (
            house.margin(),
            house.aspect_ratio(),
            house.width(),
            house.vertical_bias(),
        );

        let mut arguments = arguments;
        while let Some(argument) = arguments.next() {
            let mut value = |name: &str| {
                arguments
                    .next()
                    .ok_or_else(|| format!("{name} needs a value"))
            };
            match argument.as_str() {
                "--out" => out = PathBuf::from(value("--out")?),
                "--model" => model = PathBuf::from(value("--model")?),
                "--margin" => margin = number(&value("--margin")?, "--margin")?,
                "--aspect" => aspect = number(&value("--aspect")?, "--aspect")?,
                "--width" => width = number(&value("--width")?, "--width")? as u32,
                "--bias" => bias = number(&value("--bias")?, "--bias")?,
                other if other.starts_with("--") => {
                    return Err(format!("unknown option {other}"));
                }
                other => photograph = Some(PathBuf::from(other)),
            }
        }

        let crop = DisplayCropSpec::new(margin, aspect, width)
            .map_err(|error| error.to_string())?
            .with_vertical_bias(bias);

        Ok(Self {
            photograph: photograph.ok_or("no photograph given")?,
            out,
            model,
            crop,
        })
    }
}

fn number(text: &str, name: &str) -> Result<f32, String> {
    text.parse()
        .map_err(|_| format!("{name} expects a number, got {text:?}"))
}

fn run(options: &Options) -> Result<(), String> {
    let spec = ModelSpec::new(ModelRole::Detector, ".", &options.model, None)
        .map_err(|error| error.to_string())?;
    spec.ensure_present().map_err(|error| error.to_string())?;

    let provider = select_execution_provider();
    println!("execution provider: {provider}");

    let image = image::open(&options.photograph)
        .map_err(|error| format!("cannot read {}: {error}", options.photograph.display()))?
        .to_rgb8();
    println!(
        "{}: {}x{}",
        options.photograph.display(),
        image.width(),
        image.height()
    );

    let mut detector = FaceDetector::open(&spec, provider).map_err(|error| error.to_string())?;
    let faces = detector.detect(&image).map_err(|error| error.to_string())?;

    for (index, face) in faces.detections().iter().enumerate() {
        let bbox = face.bbox();
        println!(
            "face {index}: score {:.3} box ({:.1}, {:.1}) {:.1}x{:.1}",
            face.score(),
            bbox.x(),
            bbox.y(),
            bbox.width(),
            bbox.height()
        );
        for (name, point) in [
            ("left eye", face.landmarks().left_eye),
            ("right eye", face.landmarks().right_eye),
            ("nose", face.landmarks().nose),
            ("mouth left", face.landmarks().mouth_left),
            ("mouth right", face.landmarks().mouth_right),
        ] {
            println!("  {name:<11} ({:.1}, {:.1})", point.x, point.y);
        }
    }

    // No face and several faces are outcomes, not errors to paper over: the
    // multi-face policy at the Shutter is still an open question, so the tool
    // reports what it saw and stops rather than picking one.
    let face = match faces {
        Faces::One(face) => face,
        Faces::None => return Err("no face in the photograph".to_owned()),
        Faces::Many(faces) => {
            return Err(format!(
                "{} faces in the photograph; this tool crops one",
                faces.len()
            ));
        }
    };

    let aligned = align(&image, face.landmarks()).map_err(|error| error.to_string())?;
    let display = display_crop(&image, face.bbox(), &options.crop);

    std::fs::create_dir_all(&options.out)
        .map_err(|error| format!("cannot create {}: {error}", options.out.display()))?;
    let stem = options
        .photograph
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| "face".to_owned());

    write(&aligned, &options.out.join(format!("{stem}-aligned.png")))?;
    write(&display, &options.out.join(format!("{stem}-display.png")))?;

    println!("aligned crop: {ALIGNED_SIZE}x{ALIGNED_SIZE}");
    println!(
        "display crop: {}x{} (margin {:.2}, aspect {:.2}, bias {:.2})",
        options.crop.width(),
        options.crop.height(),
        options.crop.margin(),
        options.crop.aspect_ratio(),
        options.crop.vertical_bias()
    );

    Ok(())
}

fn write(image: &image::RgbImage, path: &Path) -> Result<(), String> {
    image
        .save(path)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    println!("wrote {}", path.display());

    Ok(())
}
