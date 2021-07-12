#!/usr/bin/env python3
# Gets model weights from a tensorflow checkpoint
# coding: utf-8
import os
import tensorflow as tf
import json
import argparse
import ast
import h5py
from tqdm import tqdm
from tensorflow.python.training import py_checkpoint_reader


def load_valid_keys(path):
    """Tries to get a list of valid model keys from a file (file must contain either a python list or a python dict
    and nothing else. 

    Returns list of keys from the file."""
    with open(path, "r") as f:
        contents = f.read()
        obj = ast.literal_eval(contents)
    if type(obj) == dict:
        return obj.keys()
    elif type(obj) == list:
        return obj
    else:
        raise TypeError("File didn't contain valid keys: {path}")


def write_weights(args):
    checkpoint_name = args.checkpoint
    outfile = args.outfile

    valid_key_list = None
    if args.valid_keys is not None:
        valid_key_list = load_valid_keys(args.valid_keys)

    latest = tf.train.latest_checkpoint(checkpoint_name)
    reader = py_checkpoint_reader.NewCheckpointReader(latest)

    valid_keys = reader.get_variable_to_shape_map().keys()
    if valid_key_list is not None:
        valid_keys = list(filter(lambda k: k in valid_key_list, valid_keys))

    _, filetype = os.path.splitext(outfile)

    if filetype == ".json":
        d = {}
        for key in tqdm(valid_keys):
            arr_list = reader.get_tensor(key).flatten().tolist()
            d[key] = arr_list

        with open(outfile, "wt") as out:
            json.dump(d, out)

    elif filetype == ".h5":
        with h5py.File(outfile, 'w') as h5f:
            d = {}
            for key in tqdm(valid_keys):
                arr = reader.get_tensor(key).flatten()
                h5f.create_dataset(key, data=arr)

    else:
        raise TypeError(
            "Unknown file extension. Only .h5 and .json are allowed.")


def main():
    parser = argparse.ArgumentParser(
        description="Output model weights from a tensorflow checkpoint. Weights are output in H5 or JSON with keys as names and values as flattened (!) weight arrays.")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument(
        "outfile", type=str, help="Path to write to. This can be a hd5 (.h5) or json (.json) file.")
    parser.add_argument("--valid-keys", dest="valid_keys", type=str,
                        help="File to a path that contains all keys that we want to include from the weights. Must contain either a python dict (where we use the keys) or list (where we use the content as list).")
    args = parser.parse_args()
    write_weights(args)


if __name__ == "__main__":
    main()
