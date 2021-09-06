#!/usr/bin/env python
import jinja2
import numpy as np
import itertools
import tensorflow as tf
import os
import torch


class RandomArrayTest:
    def __init__(self, test_name, layer_name, random_test_objects):
        """Struct that represents one Random Array Test.
        test_name: str, name of the test case
        layer_name: str, name of the layer to test
        random_test_objects: [TestObject]
        """
        assert layer_name == "ConvolutionLayer" or layer_name == "TransposedConvolutionLayer", "Layer name unknown"
        self.test_name = test_name
        self.layer_name = layer_name
        if self.layer_name == "ConvolutionLayer":
            self.function_name = "convolve"
        elif self.layer_name == "TransposedConvolutionLayer":
            self.function_name = "transposed_convolve"
        self.random_test_objects = random_test_objects


class RandomArrayTestObject:
    def __init__(self, input_arr, kernel, output_arr, padding, stride=1):
        """Struct that represents one test case for the random array tests.
        input_arr: ndarray, 3-Dimensional floating point numpy array
        output_arr: ndarray, 3-Dimensional floating point numpy array
        kernel: ndarray, 3-Dimensional floating point numpy array, weights of
        the convolutional layer
        stride: int
        padding: str, valid or same. Valid padding just applies the kernel directly,
        same padding ensures that inputsize = outputsize
        """
        if padding == "VALID":
            self.padding = "Padding::Valid"
        elif padding == "SAME":
            self.padding = "Padding::Same"
        else:
            raise ValueError(f"Illegal padding value {padding}")

        self.input_arr = numpy_array_to_rust(input_arr)
        self.output_arr = numpy_array_to_rust(output_arr)
        self.kernel = numpy_array_to_rust(kernel, shape_vec=True)
        self.stride = stride


def numpy_array_to_rust(x, shape_vec=False):
    """
        Outputs a numpy array as a Rust ndarray.
        If shape_vec is set to true, outputs 
        the array creationg through the shape_vec Rust function.
        The Rust array macro seems broken for 4-D arrays, so this is a 
        workaround.
    """
    # This removes the "dtype=..." info in the representation,
    # if needed
    if x.dtype == np.float64:
        ending_delimiter = -1
    elif x.dtype == np.float32:
        ending_delimiter = -16
    else:
        raise ValueError("array has an unsupported datatype: {x.dtype}")

    if shape_vec:
        x_shape = x.shape
        x = x.flatten()
    # removes leading array and closing paren tokens
    array_repr = f"{repr(x)}"[6:][:ending_delimiter].replace("\n", "\n\t\t")
    if shape_vec:
        return f"Array::from_shape_vec({x_shape}, vec!{array_repr}).unwrap()"
    else:
        return f"array!{array_repr}".rstrip().rstrip(",")


def torch_to_tf_img(x):
    return np.moveaxis(x, 0, 2)


def tf_to_torch_img(x):
    return np.moveaxis(x, 2, 0)


def tf_to_torch_ker(k):
    return np.moveaxis(k, [2, 3], [1, 0])


def conv2d_random_array_test(num_arrays_per_case=3, use_torch=False, seed=260896, padding="VALID"):
    """Returns a Test case that can be rendered with the 
    test_py_impl_random_arrays_template.rs into a Rust test
    that tests the conv2d Rust implementation against tf.nn.conv2d.

    num_arrays_per_case: int, number of different random arrays generated
    per (img_shape, kernel_shape) combination
    use_torch: bool, set to true if we should use the pytorch implementation to compare against.
    False for the tensorflow implementation"""
    np.random.seed(seed)
    img_shapes = [(5, 5, 1), (10, 15, 1), (15, 10, 1),
                  (6, 6, 3), (10, 15, 3), (15, 10, 3)]
    kernel_shapes = [(3, 3, 1, 3), (5, 5, 1, 2), (3, 3, 3, 2), (5, 5, 3, 2)]

    objects = []
    for im_shape, ker_shape in list(itertools.product(img_shapes, kernel_shapes)):
        if im_shape[2] != ker_shape[2]:
            continue  # shapes are not compatible, channel size missmatch

        for i in range(num_arrays_per_case):
            im = np.random.rand(*im_shape).astype(dtype=np.float32)
            ker = np.random.rand(*ker_shape).astype(dtype=np.float32)

            im_pt = torch.FloatTensor(
                np.expand_dims(tf_to_torch_img(im), axis=0))
            ker_pt = torch.FloatTensor(tf_to_torch_ker(ker))
            out_pt = torch.nn.functional.conv2d(im_pt, ker_pt)
            out_pt_numpy = torch_to_tf_img(np.squeeze(
                out_pt.numpy(), axis=0).astype(np.float32))

            # axis 0 is batch dimension, which we need to remove and add back in
            im_tf = tf.constant(
                np.expand_dims(im, axis=0), dtype=tf.float32)
            ker_tf = tf.constant(ker, dtype=tf.float32)
            out_tf = tf.nn.conv2d(im_tf, ker_tf, strides=[
                1, 1, 1, 1], padding=padding)
            out_tf_numpy = np.squeeze(
                out_tf.numpy(), axis=0).astype(np.float32)

            # to make sure tf and pt implementations agree
            assert np.allclose(
                out_tf_numpy, out_pt_numpy), f"Torch and Tensorflow implementations didn't match.\nTorch: {out_pt_numpy}\n Tensorflow:{out_tf_numpy}"

            if use_torch:
                out = out_pt_numpy
            else:
                out = out_tf_numpy


            # reordering the images and weights
            #
            # For weights:
            #   TF ordering:
            #     kheight x kwidth x in x out
            #   our ordering:
            #     out x in x kwidth x kheight
            #
            # For images:
            #   TF ordering:
            #     height x width x channels
            #   our ordering:
            #     channels x height x width
            im = np.moveaxis(im, [0, 1, 2], [1, 2, 0])
            ker = np.moveaxis(ker, [0, 1, 2, 3], [3, 2, 1, 0])
            out = np.moveaxis(out, [0, 1, 2], [1, 2, 0])

            test_obj = RandomArrayTestObject(im, ker, out, padding)
            objects.append(test_obj)

    if use_torch:
        test_name = "conv2d_torch"
    else:
        test_name = "conv2d"
    return RandomArrayTest(test_name, "ConvolutionLayer", objects)


def conv2d_transpose_random_array_test(num_arrays_per_case=3):
    """Returns a Test case that can be rendered with the 
    test_py_impl_random_arrays_template.rs into a Rust test
    that tests the conv2d_transpose Rust implementation against tf.nn.conv2d.

    num_arrays_per_case: int, number of different random arrays generated
    per (img_shape, kernel_shape) combination"""
    img_shapes = [(5, 5, 1), (10, 15, 1), (15, 10, 1),
                  (6, 6, 3), (10, 15, 3), (15, 10, 3)]
    kernel_shapes = [(3, 3, 3, 1), (5, 5, 2, 1), (3, 3, 2, 3), (5, 5, 2, 3)]
    padding = "SAME"

    objects = []
    for im_shape, ker_shape in list(itertools.product(img_shapes, kernel_shapes)):
        if im_shape[2] != ker_shape[3]:
            continue  # shapes are not compatible, channel size missmatch
        for i in range(num_arrays_per_case):
            im = np.random.rand(*im_shape).astype(np.float32)
            ker = np.random.rand(*ker_shape).astype(np.float32)
            # axis 0 is batch dimension, which we need to remove and add back in
            im_tf = tf.constant(np.expand_dims(im, axis=0), dtype=tf.float32)
            ker_tf = tf.constant(ker, dtype=tf.float32)
            output_shape = (1, im_shape[0], im_shape[1], ker_shape[2])
            # conv2d transpose expected filters as [height, width, out, in]
            # https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose
            out_tf = tf.nn.conv2d_transpose(
                im_tf, ker_tf, output_shape=output_shape, strides=[1, 1, 1, 1], padding=padding)
            out = np.squeeze(out_tf.numpy(), axis=0)

            # reordering the images and weights
            # ! This is different than conv2d !
            #
            # For weights:
            #   TF ordering:
            #     kheight x kwidth x out x in
            #   our ordering:
            #     out x in x kwidth x kheight
            #
            # For images:
            #   TF ordering:
            #     height x width x channels
            #   our ordering:
            #     channels x height x width
            im = np.moveaxis(im, [0, 1, 2], [1, 2, 0])
            ker = np.moveaxis(ker, [0, 1, 2, 3], [3, 2, 0, 1])
            out = np.moveaxis(out, [0, 1, 2], [1, 2, 0])

            test_obj = RandomArrayTestObject(im, ker, out, padding)
            objects.append(test_obj)

    return RandomArrayTest("conv2d_transpose", "TransposedConvolutionLayer", objects)


def write_test_to_file(ml_test_folder, test_content, test_name):
    test_filename = f"{test_name}_automated_test.rs"
    test_output_filepath = os.path.join(
        ml_test_folder, test_filename)
    with open(test_output_filepath, "w+") as conv2d_output_file:
        conv2d_output_file.write(test_content)
        print(f"Successfully wrote {test_name} test to {test_output_filepath}")
    os.system(f"rustfmt {test_output_filepath}")
    print(f"Formatted {test_name} test.")


def main():
    # Tensorflow conv2d inputs are given as
    # - batch_shape + [in_height, in_width, in_channels]
    # and weights as
    # - [filter_height * filter_width * in_channels, output_channels]
    # See also: https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
    # analog for conv2d_transpose:
    # https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose

    np.set_printoptions(suppress=True)
    # loading Jinja with the random array test template
    loader = jinja2.FileSystemLoader("./templates")
    env = jinja2.Environment(loader=loader)
    template = env.get_template("test_py_impl_random_arrays_template.rs")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ml_test_folder = os.path.join(project_root, "ml", "tests")

    # writing out the conv2d test cases
    conv2d_test_case = conv2d_random_array_test()
    conv2d_test_content = template.render(
        random_tests=[conv2d_test_case], file=__file__)
    write_test_to_file(ml_test_folder, conv2d_test_content, "conv2d")

    # writing out the conv2d test cases with torch
    conv2d_torch_test_case = conv2d_random_array_test(use_torch=True)
    conv2d_torch_test_content = template.render(
        random_tests=[conv2d_torch_test_case], file=__file__)
    write_test_to_file(
        ml_test_folder, conv2d_torch_test_content, "conv2d_torch")

    # writing out the conv2d_tranposed test cases
    conv2d_transpose_test_case = conv2d_transpose_random_array_test()
    conv2d_transpose_test_content = template.render(
        random_tests=[conv2d_transpose_test_case], file=__file__)
    write_test_to_file(ml_test_folder, conv2d_transpose_test_content,
                       "conv2d_transpose")


if __name__ == "__main__":
    main()
