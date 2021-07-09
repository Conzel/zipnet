# Yibo Yang, 2020

import tensorflow.compat.v1 as tf


def read_png(filename):
    """Loads a image file as float32 HxWx3 array; tested to work on png and jpg images."""
    string = tf.read_file(filename)
    image = tf.image.decode_image(string, channels=3)
    image = tf.cast(image, tf.float32)
    image /= 255
    return image


def quantize_image(image):
    image = tf.round(image * 255)
    image = tf.saturate_cast(image, tf.uint8)
    return image


def write_png(filename, image):
    """Saves an image to a PNG file."""
    image = quantize_image(image)
    string = tf.image.encode_png(image)
    return tf.write_file(filename, string)


def convert_float_to_uint8(image):
    image = tf.round(image * 255)
    image = tf.saturate_cast(image, tf.uint8)
    return image


def convert_uint8_to_float(image):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image


import numpy as np


# for reading images in .npy format
def read_npy_file_helper(file_name_in_bytes):
    # data = np.load(file_name_in_bytes.decode('utf-8'))
    data = np.load(
        file_name_in_bytes
    )  # turns out this works too without decoding to str first
    # assert data.dtype is np.float32   # needs to match the type argument in the caller tf.data.Dataset.map
    return data


def get_runname(
    args_dict,
    record_keys=("num_filters", "num_hfilters", "lmbda", "last_step"),
    prefix="",
):
    """
    Given a dictionary of cmdline arguments, return a string that identifies the training run.
    :param args_dict:
    :return:
    """
    config_strs = []  # ['key1=val1', 'key2=val2', ...]

    # for key, val in args_dict.items():
    #     if isinstance(val, (list, tuple)):
    #         val_str = '_'.join(map(str, val))
    #         config_strs.append('%s=%s' % (key, val_str))

    for key in record_keys:
        if key == "num_hfilters" and int(args_dict[key]) <= 0:
            continue
        config_strs.append("%s=%s" % (key, args_dict[key]))

    return "-".join([prefix] + config_strs)
