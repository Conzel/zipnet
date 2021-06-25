use quicli::prelude::*;
use std::path::PathBuf;
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
    #[structopt(flatten)]
    verbosity: Verbosity,
}

/// Decompresses an image using a neural network
#[derive(Debug, StructOpt)]
struct DecompressOpts {
    /// Path to the Zipnet-compressed image
    #[structopt(parse(from_os_str))]
    compressed: PathBuf,
    /// Output path, if none given, just writes to input path with file
    /// ending removed.
    #[structopt(short = "o", long = "output", parse(from_os_str))]
    output: Option<PathBuf>,
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

/// Compresses an image using the given Compression options.
fn compress(cfg: CompressOpts) {}

/// Decompresses an image using the given decompression options.
fn decompress(cfg: DecompressOpts) {}

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
    fn run(&self) {}
    fn get_verbosity(&self) -> &Verbosity {
        &self.verbosity
    }
}

impl ZipnetOpts for CompressOpts {
    // Performs Compression
    fn run(&self) {}
    fn get_verbosity(&self) -> &Verbosity {
        &self.verbosity
    }
}

impl ZipnetOpts for StatsOpts {
    // Prints out statistics to StdOut
    fn run(&self) {}
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
