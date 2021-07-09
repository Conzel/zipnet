import glob
import os
import sys

import tensorflow.compat.v1 as tf

from utils import read_png, read_npy_file_helper, get_runname


def parse_args(argv):
    """Parses command line arguments."""
    import argparse

    # from absl import app
    from absl.flags import argparse_flags

    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # High-level options.
    parser.add_argument(
        "--verbose",
        "-V",
        action="store_true",
        help="Report bitrate and distortion when training or compressing.",
    )
    parser.add_argument(
        "--num_filters", type=int, default=-1, help="Number of filters in the latents."
    )
    parser.add_argument(
        "--num_hfilters",
        type=int,
        default=-1,
        help="Number of filters in the hyper latents.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="./checkpoints",
        help="Directory where to save/load model checkpoints.",
    )
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="What to do: 'train' loads training data and trains (or continues "
        "to train) a new model. 'compress' reads an image file (lossless "
        "PNG format) and writes a compressed binary file. 'decompress' "
        "reads a binary file and reconstructs the image (in PNG format). "
        "input and output filenames need to be provided for the latter "
        "two options. Invoke '<command> -h' for more information.",
    )

    # 'train' subcommand.
    train_cmd = subparsers.add_parser(
        "train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Trains (or continues to train) a new model.",
    )
    train_cmd.add_argument(
        "--train_glob",
        default="images/*.png",
        help="Glob pattern identifying training data. This pattern must expand "
        "to a list of RGB images in PNG format.",
    )
    train_cmd.add_argument(
        "--batchsize", type=int, default=8, help="Batch size for training."
    )
    train_cmd.add_argument(
        "--patchsize", type=int, default=256, help="Size of image patches for training."
    )
    train_cmd.add_argument(
        "--lambda",
        type=float,
        default=0.01,
        dest="lmbda",
        help="Lambda for rate-distortion tradeoff.",
    )
    train_cmd.add_argument(
        "--last_step",
        type=int,
        default=1000000,
        help="Train up to this number of steps.",
    )
    train_cmd.add_argument(
        "--preprocess_threads",
        type=int,
        default=16,
        help="Number of CPU threads to use for parallel decoding of training "
        "images.",
    )
    train_cmd.add_argument(
        "--logdir",
        default="/tmp/tf_logs",  # '--log_dir' seems to conflict with absl.flags's existing
        help="Directory for storing Tensorboard logging files; set to empty string '' to disable Tensorboard logging.",
    )
    train_cmd.add_argument(
        "--save_checkpoint_secs",
        type=int,
        default=300,
        help="Seconds elapsed b/w saving models.",
    )
    train_cmd.add_argument(
        "--save_summary_secs",
        type=int,
        default=60,
        help="Seconds elapsed b/w saving tf summaries.",
    )

    # 'compress' subcommand.
    compress_cmd = subparsers.add_parser(
        "compress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a PNG file, compresses it, and writes a TFCI file.",
    )
    compress_cmd.add_argument(
        "--results_dir",
        default="./results",
        help="Directory for storing compression stats/results; set to empty string '' to disable.",
    )
    compress_cmd.add_argument(
        "--lambda",
        type=float,
        default=-1,
        dest="lmbda",
        help="Lambda for rate-distortion tradeoff.",
    )
    compress_cmd.add_argument(
        "--sga_its",
        type=int,
        default=2000,
        help="Number of SGA (Stochastic Gumbel Annealing) iterations .",
    )
    compress_cmd.add_argument(
        "--annealing_rate", type=float, default=1e-3, help="Annealing rate for SGA."
    )
    compress_cmd.add_argument(
        "--t0",
        type=int,
        default=700,
        help="Number of 'soft-quantization' optimization iterations before annealing in SGA.",
    )

    # 'decompress' subcommand.
    decompress_cmd = subparsers.add_parser(
        "decompress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a TFCI file, reconstructs the image, and writes back "
        "a PNG file.",
    )

    # Arguments for both 'compress' and 'decompress'.
    for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
        cmd.add_argument(
            "runname",
            help="Model name identifier constructed from run config, like 'bmshj2018-num_filters=...'",
        )
        cmd.add_argument("input_file", help="Input filename.")
        cmd.add_argument(
            "output_file",
            nargs="?",
            help="Output filename (optional). If not provided, appends '{}' to "
            "the input filename.".format(ext),
        )

    # Parse arguments.
    args = parser.parse_args(argv[1:])
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    return args
