#!/usr/bin/env python
import jinja2
import numpy as np
import itertools
import tensorflow as tf
import os

np.random.seed(260896)

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
            raise ValueError("Transposed conv not yet implemented")
        self.random_test_objects = random_test_objects

class RandomArrayTestObject:
    def __init__(self, input_arr, kernel, output_arr, stride=1, padding=0):
        """Struct that represents one test case for the random array tests.
        input_arr: ndarray, 3-Dimensional floating point numpy array
        output_arr: ndarray, 3-Dimensional floating point numpy array
        kernel: ndarray, 3-Dimensional floating point numpy array, weights of
        the convolutional layer
        stride: int
        padding: int
        """
        self.input_arr = numpy_array_to_rust(input_arr)
        self.output_arr = numpy_array_to_rust(output_arr)
        self.kernel = numpy_array_to_rust(kernel, shape_vec=True)
        self.stride = stride
        self.padding = padding

def numpy_array_to_rust(x, shape_vec=False):
    """
        Outputs a numpy array as a Rust ndarray.
        If shape_vec is set to true, outputs 
        the array creationg through the shape_vec Rust function.
        The Rust array macro seems broken for 4-D arrays, so this is a 
        workaround.
    """
    if shape_vec:
        x_shape = x.shape
        x = x.flatten()
    # removes leading array and closing paren tokens
    array_repr = f"{repr(x)}"[6:][:-1].replace("\n", "\n\t\t")
    if shape_vec:
        return f"Array::from_shape_vec({x_shape}, vec!{array_repr}).unwrap()"
    else:
        return f"array!{array_repr}"

def conv2d_random_array_test(num_arrays_per_case=3): 
    """Returns a Test case that can be rendered with the 
    test_py_impl_random_arrays_template.rs into a Rust test
    that tests the conv2d Rust implementation against tf.nn.conv2d.
    
    num_arrays_per_case: int, number of different random arrays generated
    per (img_shape, kernel_shape) combination"""
    img_shapes = [(5,5,1), (10,15,1), (15,10,1), (6,6,3), (10,15,3), (15,10,3)]
    kernel_shapes = [(3,3,1,3), (5,5,1,2), (3,3,3,2), (5,5,3,2)]
    
    # padding and strides are 1 for now

    objects = []
    for im_shape, ker_shape in list(itertools.product(img_shapes, kernel_shapes)):
        if im_shape[2] != ker_shape[2]:
            continue # shapes are not compatible, channel size missmatch
        for i in range(num_arrays_per_case):
            im = np.random.rand(*im_shape)
            ker = np.random.rand(*ker_shape)
            # axis 0 is batch dimension, which we need to remove and add back in
            im_tf = tf.constant(np.expand_dims(im, axis=0), dtype=tf.float64)
            ker_tf = tf.constant(ker, dtype=tf.float64)
            out_tf = tf.nn.conv2d(im_tf, ker_tf, strides=[1,1,1,1], padding="VALID")
            out = np.squeeze(out_tf.numpy(), axis=0)

            # reordering the images and weights
            #
            # For weights:
            #   TF ordering: 
            #     kheight x kwidth x in x out
            #   our ordering:
            #     in x out x kwidth x kheight
            #
            # For images:
            #   TF ordering: 
            #     height x width x channels
            #   our ordering:
            #     channels x height x width
            im = np.moveaxis(im, [0,1,2], [1,2,0])
            ker = np.moveaxis(ker, [0,1,2,3], [3,2,0,1])
            out = np.moveaxis(out, [0,1,2], [1,2,0])

            test_obj = RandomArrayTestObject(im, ker, out)
            objects.append(test_obj)
   
    return RandomArrayTest("conv2d", "ConvolutionLayer", objects)


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
    conv2d_test_content = template.render(random_tests=[conv2d_test_case], file=__file__)

    conv2d_output_filename = os.path.join(ml_test_folder, "convolutions_automated_test.rs")
    with open(conv2d_output_filename, "w+") as conv2d_output_file:
        conv2d_output_file.write(conv2d_test_content)
        print(f"Successfully wrote conv2d test to {conv2d_output_filename}")


    # TODO:
    # writing out the transposed conv2d test cases


if __name__ == "__main__":
    main()
