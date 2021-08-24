# Scripts
This directory contains various scripts, currently all are there for automatic code generation. 

## Automated unit tests
We have implemented automated unit test generation for convolution and transposed convolutions. These have to conform tightly to the tensorflow implementation (https://www.tensorflow.org/api_docs/python/tf/nn/conv2d and https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose respectively), which is why automatic code generation provides useful. We create random arrays, pass them through the tensorflow outputs and check the output of the Rust implementation against them. 

For usage, refer to the Readme in the project root, under `Testing`.

## Model generation
The Machine Learning models require a lot of boilerplate code, while they can be described in a high level of abstraction. For this, we have a specifications file under `specifications/model_specifications.json`. These can be used to describe the needed models. We can then use `generate_tests.py` to automatically regenerate them under `ml/src/models.rs`.

The specification is built as follows:

- ```root: [<model_1>, <model_2>, ... <model_n>]```
- ```
  model: {
    name: str, name of the Rust struct created,
    weight_name: str, name of the layer in the weights file,
    layers: [<layer_1>, <layer_2>, ... <layer_n>]
  }
  ```
- ```
  layer: {
    type: str, convolution or convolution_transposed,
    filters: int, number of filters that the convolution uses,
    channels: int, number of channels of the convolution. only necessary on the first layer, as subsequent ones can calculate the number of channels from previous information,
    stride: int, same stride used in all dimensions
    padding: str, same (output dim identical to input dim) or valid (no padding, only to make kernel fit),
    bias: bool, currently unused,
    activation: str, gdn, igdn, relu or none,
    kernel_shape: str, must have form of "(kernel width, kernel height)"
  }
  ```