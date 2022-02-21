//! This crate ties in all the parts of the projects and provides a clean command line interface
//! to make encoding/decoding images feasible.

use coders::{
    dummy_coders::DummyCoder,
    factorized_prior_coders::{FactorizedPriorDecoder, FactorizedPriorEncoder},
    statistics::Statistics,
    Decoder, Encoder,
};
use env_logger::Builder;
use image::io::Reader as ImageReader;
use image::RgbImage;
use ml::{
    models::{CodingModel, JohnstonDecoder, MinnenEncoder},
    weight_loader::NpzWeightLoader,
};
use ndarray::{Array, Array3};
use ndarray_npy::read_npy;
use nshare::ToNdarray3;
use quicli::prelude::*;
use std::{
    ffi::OsStr,
    fs,
    io::{Read, Write},
};
use std::{fs::File, path::PathBuf};
use structopt::StructOpt;

/// Compresses an image using a neural network
#[derive(Debug, StructOpt)]
struct CompressOpts {
    /// Path to image that should be compressed
    #[structopt(parse(from_os_str))]
    image: PathBuf,
    /// Output path, writes to /path/to/image.bin if not available
    #[structopt(short = "o", long = "output", parse(from_os_str))]
    output: Option<PathBuf>,
    /// Sets the desired bitrate per pixel
    #[structopt(short = "b", long = "bitrate")]
    bitrate: Option<f64>,
    /// Reserved for debugging and testing. Encodes with a dummy encoder.
    #[structopt(short, long)]
    debug: bool,
    #[structopt(flatten)]
    verbosity: Verbosity,
}

/// Decompresses an image using a neural network
#[derive(Debug, StructOpt)]
struct DecompressOpts {
    /// Path to the Zipnet-compressed image
    #[structopt(parse(from_os_str))]
    compressed: PathBuf,
    /// Output path
    #[structopt(parse(from_os_str))]
    output: PathBuf,
    /// Reserved for debugging and testing. Encodes with a dummy encoder.
    #[structopt(short, long)]
    debug: bool,

    #[structopt(flatten)]
    verbosity: Verbosity,
}

/// Prints statistics about compression and decompression process
#[derive(Debug, StructOpt)]
struct AutoEncoderOpts {
    /// Path to the input image. Output image is saved under "<input_image>_recovered.png"
    #[structopt(parse(from_os_str))]
    image: PathBuf,
    #[structopt(flatten)]
    verbosity: Verbosity,
    #[structopt(long)]
    from_latents: bool,
}

/// Does a combined compression and decompression process with no
#[derive(Debug, StructOpt)]
struct StatsOpts {
    /// Path to the input image
    #[structopt(parse(from_os_str))]
    image: PathBuf,
    #[structopt(flatten)]
    verbosity: Verbosity,
}

/// Compress and decompress images easily using a neural network.
#[derive(Debug, StructOpt)]
#[structopt(name = "ZipNet")]
enum Zipnet {
    #[structopt(
        name = "compress",
        about = "Compresses an image using a neural network."
    )]
    Compress(CompressOpts),
    #[structopt(
        name = "decompress",
        about = "Decompress an image previously compressed by ZipNet."
    )]
    Decompress(DecompressOpts),
    #[structopt(
        name = "autoencoder",
        about = "Does a pure autoencoder run, no compression involved."
    )]
    AutoEncoder(AutoEncoderOpts),
    #[structopt(
        name = "statistics",
        about = "Prints out statistics about the decompression and compression process to StdOut. \
        Performs multiple compression and decompression passes and thus might take a while during execution."
    )]
    Statistics(StatsOpts),
}

/// Trait for the subcommands that zipnet uses
trait ZipnetOpts {
    /// Performs the subcommand
    fn run(&self);
    /// Returns the verbosity command
    fn get_verbosity(&self) -> &Verbosity;
    /// Sets up logging
    fn setup_env_logger(&self) -> CliResult {
        let mut builder = Builder::from_default_env();

        builder
            .filter(None, self.get_verbosity().log_level().to_level_filter())
            .init();

        Ok(())
    }
}

impl ZipnetOpts for DecompressOpts {
    // Performs Decompression
    fn run(&self) {
        let mut file = File::open(&self.compressed).unwrap();
        let mut decoder: Box<dyn Decoder<_>> = if self.debug {
            Box::new(DummyCoder::new())
        } else {
            Box::new(FactorizedPriorDecoder::new())
        };
        let metadata = fs::metadata(&self.compressed).unwrap();
        let mut encoded_bin = vec![0; metadata.len() as usize];
        file.read(&mut encoded_bin).unwrap();

        let encoded = bincode::deserialize(&encoded_bin).unwrap();

        let decoded = decoder.decode(encoded).unwrap();
        let image = array_to_image(decoded.map(|x| to_pixel(x, self.debug)));
        image.save(&self.output).unwrap();
    }
    fn get_verbosity(&self) -> &Verbosity {
        &self.verbosity
    }
}

/// Turns output from neural net into a pixel value, performs postprocessing
fn to_pixel(x: &f32, debug: bool) -> u8 {
    if debug {
        x.round().clamp(0.0, 255.0) as u8
    } else {
        (x.clamp(0.0, 1.0) * 255.0).round() as u8
    }
}

/// Turns ndarray to rgb image
///
/// Taken from <https://stackoverflow.com/questions/56762026/how-to-save-ndarray-in-rust-as-image>
fn array_to_image(arr: Array3<u8>) -> RgbImage {
    // we get the image in PT layout, which is (C,H,W), but need (H,W,C)
    let permuted_view = arr.view().permuted_axes([1, 2, 0]);
    // again hack to fix the memory layout
    let permuted_array: Array3<u8> = Array::from_shape_vec(
        permuted_view.dim(),
        permuted_view.iter().map(|x| *x).collect(),
    )
    .unwrap();

    assert!(permuted_array.is_standard_layout());

    let (height, width, _) = permuted_array.dim();
    let raw = permuted_array.into_raw_vec();

    RgbImage::from_raw(width as u32, height as u32, raw)
        .expect("container should have the right size for the image dimensions")
}

impl ZipnetOpts for CompressOpts {
    // Performs Compression
    fn run(&self) {
        // preprocessing and getting the image
        let img_data = get_image(&self.image);
        let mut encoder: Box<dyn Encoder<_>> = if self.debug {
            Box::new(DummyCoder::new())
        } else {
            Box::new(FactorizedPriorEncoder::new())
        };

        let encoded = encoder.encode(&img_data);
        let encoded_bin = bincode::serialize(&encoded).unwrap();

        let mut alternate_output_name = self.image.clone();
        alternate_output_name.set_extension("bin");

        let filepath = match &self.output {
            Some(p) => p,
            None => &alternate_output_name,
        };
        let mut file = File::create(filepath).unwrap();
        file.write(&encoded_bin).unwrap();
    }

    fn get_verbosity(&self) -> &Verbosity {
        &self.verbosity
    }
}

/// Returns preprocessed image from path buffer
fn get_image(im_path: &PathBuf) -> Array3<f32> {
    match im_path.extension().and_then(OsStr::to_str).unwrap() {
        "npy" => read_npy(im_path).unwrap(),
        "png" | "jpg" => get_image_raw(im_path).map(|x| *x as f32 / 255.0),
        _ => panic!("Image had unrecognized type. Only .jpg, .png and .npy are supported."),
    }
}

/// Returns image without preprocessing. Only useable for actual images, not npy arrays.
fn get_image_raw(im_path: &PathBuf) -> Array3<u8> {
    ImageReader::open(im_path)
        .unwrap()
        .decode()
        .unwrap()
        .to_rgb8()
        .into_ndarray3()
}

impl ZipnetOpts for StatsOpts {
    // Prints out statistics to StdOut
    fn run(&self) {
        let img_data = get_image_raw(&self.image);
        let stats = Statistics::new(&img_data);
        println!("{}", stats);
    }
    fn get_verbosity(&self) -> &Verbosity {
        &self.verbosity
    }
}

impl ZipnetOpts for AutoEncoderOpts {
    fn run(&self) {
        let mut loader = NpzWeightLoader::full_loader();
        let analyzer = MinnenEncoder::new(&mut loader);
        let synthesizer = JohnstonDecoder::new(&mut loader);

        let latent = if self.from_latents {
            read_npy(&self.image).unwrap()
        } else {
            // preprocessing and getting the image
            let img_data = get_image(&self.image);
            analyzer.forward_pass(&img_data)
        };
        let reconstructed = synthesizer.forward_pass(&latent);

        let reconstructed_image = array_to_image(reconstructed.map(|x| to_pixel(x, false)));

        // getting new output path
        let stem = self.image.file_stem().unwrap();
        let new_filename = stem.to_str().unwrap().to_owned() + "-reconstructed-ae.png";
        let output_path = &self.image.parent().unwrap().join(new_filename);

        reconstructed_image.save(&output_path).unwrap();
    }
    fn get_verbosity(&self) -> &Verbosity {
        &self.verbosity
    }
}

impl ZipnetOpts for Zipnet {
    fn run(&self) {
        match self {
            Zipnet::Compress(c) => c.run(),
            Zipnet::Decompress(c) => c.run(),
            Zipnet::Statistics(c) => c.run(),
            Zipnet::AutoEncoder(c) => c.run(),
        }
    }

    fn get_verbosity(&self) -> &Verbosity {
        match self {
            Zipnet::Compress(c) => c.get_verbosity(),
            Zipnet::Decompress(c) => c.get_verbosity(),
            Zipnet::Statistics(c) => c.get_verbosity(),
            Zipnet::AutoEncoder(c) => c.get_verbosity(),
        }
    }
}

fn main() -> CliResult {
    let args = Zipnet::from_args();
    args.setup_env_logger()?;
    args.run();
    Ok(())
}
