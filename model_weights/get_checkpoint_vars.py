#!/usr/bin/env python3
# coding: utf-8
import tensorflow as tf
import pprint
import argparse
from tensorflow.python.training import py_checkpoint_reader


def write_vars(args):
    checkpoint_name = args.checkpoint
    shapes_out = args.outfile

    latest = tf.train.latest_checkpoint(checkpoint_name)
    reader = py_checkpoint_reader.NewCheckpointReader(latest)

    with open(shapes_out, "wt") as out:
        pprint.pprint(reader.get_variable_to_shape_map(), stream=out)


def main():
    parser = argparse.ArgumentParser(
        description="Output variables with shape from a tf checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("outfile", type=str, help="Path to write to")
    args = parser.parse_args()
    write_vars(args)

if __name__ == "__main__":
    main()
