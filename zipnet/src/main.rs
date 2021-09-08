use bincode::serialize;
use coders::{
    dummy_coders::DummyCoder,
    hierarchical_coders::{MeanScaleHierarchicalDecoder, MeanScaleHierarchicalEncoder},
    statistics::Statistics,
    Decoder, Encoder,
};
use image::RgbImage;
use image::{io::Reader as ImageReader, DynamicImage};
use ndarray::Array3;
use nshare::ToNdarray3;
use quicli::prelude::*;
use std::{
    array, fs,
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
        self.get_verbosity().setup_env_logger("zipnet")?;
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
            Box::new(MeanScaleHierarchicalDecoder::MinnenJohnstonDecoder())
        };
        let metadata = fs::metadata(&self.compressed).unwrap();
        let mut encoded_bin = vec![0; metadata.len() as usize];
        file.read(&mut encoded_bin).unwrap();

        let encoded = bincode::deserialize(&encoded_bin).unwrap();

        let decoded = decoder.decode(encoded).unwrap();
        let image = array_to_image(decoded.map(to_pixel));
        image.save(&self.output).unwrap();
    }
    fn get_verbosity(&self) -> &Verbosity {
        &self.verbosity
    }
}

// Turns output from neural net into a pixel value
// TODO: Translate correct python conversion code:
//
// def write_png(filename, image):
//     """Saves an image to a PNG file."""
//     image = quantize_image(image)
//     string = tf.image.encode_png(image)
//     return tf.write_file(filename, string)
//
// def quantize_image(image):
//     image = tf.round(image * 255)
//     image = tf.saturate_cast(image, tf.uint8)
//     return image
//
fn to_pixel(x: &f32) -> u8 {
    x.round().clamp(0.0, 255.0) as u8
}

// From: https://stackoverflow.com/questions/56762026/how-to-save-ndarray-in-rust-as-image
fn array_to_image(arr: Array3<u8>) -> RgbImage {
    assert!(arr.is_standard_layout());

    let (height, width, _) = arr.dim();
    let raw = arr.into_raw_vec();

    RgbImage::from_raw(width as u32, height as u32, raw)
        .expect("container should have the right size for the image dimensions")
}

impl ZipnetOpts for CompressOpts {
    // Performs Compression
    fn run(&self) {
        let img_data = get_image(&self.image).map(|x| *x as f32);
        let mut encoder: Box<dyn Encoder<_>> = if self.debug {
            Box::new(DummyCoder::new())
        } else {
            Box::new(MeanScaleHierarchicalEncoder::MinnenJohnstonEncoder())
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

fn get_image(im_path: &PathBuf) -> Array3<u8> {
    let img = ImageReader::open(im_path).unwrap().decode().unwrap();
    match img {
        DynamicImage::ImageRgb8(i) => i.into_ndarray3(),
        DynamicImage::ImageRgba8(i) => i.into_ndarray3(),
        // TODO: Handle this more gracefully
        _ => panic!("Wrong image type given."),
    }
}

impl ZipnetOpts for StatsOpts {
    // Prints out statistics to StdOut
    fn run(&self) {
        let img_data = get_image(&self.image);
        let stats = Statistics::new(&img_data);
        println!("{}", stats);
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
        }
    }

    fn get_verbosity(&self) -> &Verbosity {
        match self {
            Zipnet::Compress(c) => c.get_verbosity(),
            Zipnet::Decompress(c) => c.get_verbosity(),
            Zipnet::Statistics(c) => c.get_verbosity(),
        }
    }
}

fn main() -> CliResult {
    let args = Zipnet::from_args();
    args.setup_env_logger()?;
    args.run();
    Ok(())
}
