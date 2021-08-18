#!/usr/bin/env python
import jinja2
import numpy as np

class TestObject:
"""The structs that are used in the Jinja templates."""
    def __init__(self, layer_name, input_arr, output_arr, kernel, stride=1, padding=0):
        assert!(layer_name == "ConvolutionLayer" || layer_name == "TransposedConvolutionLayer", "Layer name unknown")
        self.input_arr = input_arr
        self.output_arr = output_arr
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

def numpy_array_to_rust(x):
    """
        Outputs a numpy array as a Rust ndarray.
    """
    array_repr = f"{x}".replace("\n", ",\n\t\t")
    return f"array!{array_repr}"

# Tensorflow conv2d inputs are given as
# - batch_shape + [in_height, in_width, in_channels]
# and weights as 
# - [filter_height * filter_width * in_channels, output_channels]
# See also: https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
# analog for conv2d_transpose:
# https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose

np.set_printoptions(suppress=True)

loader = jinja2.FileSystemLoader("./templates")
env = jinja2.Environment(loader=loader)

template = env.get_template("test_py_impl_random_arrays_template.rs")

num_arrs = 10

random_arrays = [numpy_array_to_rust(np.random.rand(5,5,5,5)) 
        for x in range(num_arrs)]


