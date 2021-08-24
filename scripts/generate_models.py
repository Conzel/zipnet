#!/usr/bin/env python
import jinja2
import os
import json
import copy


class Activation:
    """
    Activation that can be parsed by the models_template.rs Jinja2 template
    """

    def __init__(self, name, corresponding_layer):
        assert name in ["gdn", "igdn", "relu"]
        if name == "gdn":
            self.preinit_weights_gdn_type(corresponding_layer)
            self.gdn_init()
        elif name == "igdn":
            self.preinit_weights_gdn_type(corresponding_layer)
            self.igdn_init()
        elif name == "relu":
            self.relu_init()
        else:
            raise ValueError(
                "Unknown activation passed {name}, only gdn, igdn, relu accepted")

    class Weight:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    def relu_init(self):
        self.layer_name = "ReluLayer"
        self.name = "relu"

    def preinit_weights_gdn_type(self, corresponding_layer):
        filters = corresponding_layer["filters"]
        self.weights = [self.Weight("gamma", f"{filters}"), self.Weight(
            "beta", f"({filters},{filters})")]

    def gdn_init(self):
        self.layer_name = "GdnLayer"
        self.name = "gdn"

    def igdn_init(self):
        self.layer_name = "IgdnLayer"
        self.name = "igdn"


def add_channels(layer_spec_list):
    """
    Adds the channels to a list of layer specifications, always using the channels
    of the previous layers. 
    """
    ret = copy.deepcopy(layer_spec_list)
    for i in range(1, len(layer_spec_list)):
        ret[i]["channels"] = layer_spec_list[i-1]["filters"]
    return ret


class Layer:
    """
    Layer that can be parsed by the models_template.rs Jinja file.
    """

    def __init__(self, specification):
        """
        specification: dict, containing all the layer keys
        """
        self.kernel_height = specification["kernel_shape"][0]
        self.kernel_width = specification["kernel_shape"][1]

        layer_type = specification["type"]
        if layer_type == "convolution":
            self.name = "ConvolutionLayer"
        elif layer_type == "convolution_transpose":
            self.name = "TransposedConvolutionLayer"
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        activation = specification["activation"]
        if activation != "none":
            self.activation = Activation(
                activation, specification)
        else:
            self.activation = None

        padding = specification["padding"]
        if padding == "same":
            self.padding = "Padding::Same"
        elif padding == "valid":
            self.padding = "Padding::Valid"
        else:
            raise ValueError(f"Unknown padding mode: {padding}")

        self.filters = specification["filters"]
        self.stride = specification["stride"]
        self.channels = specification["channels"]


class Model:
    """
    Model that can be parsed by the models_template.rs Jinja file.
    """

    def __init__(self, specification):
        """
        specification: dict, containing all the model keys
        """
        self.name = specification["name"]
        self.weight_name = specification["weight_name"]
        self.layers = list(map(Layer, add_channels(specification["layers"])))


# def main():
# loading Jinja with the random array test template
loader = jinja2.FileSystemLoader("./templates")
env = jinja2.Environment(loader=loader)
template = env.get_template("models_template.rs")

specification_file = open("specifications/model_specification.json", "r")
specifications = json.load(specification_file)
models = list(map(Model, specifications))
models_rs_content = template.render(models=models, file=__file__)

print(models_rs_content)

# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ml_test_folder = os.path.join(project_root, "ml", "tests")

# # writing out the conv2d test cases
# conv2d_test_case = conv2d_random_array_test()
# conv2d_test_content = template.render(
#     random_tests=[conv2d_test_case], file=__file__)

# conv2d_output_filename = os.path.join(
#     ml_test_folder, "convolutions_automated_test.rs")
# with open(conv2d_output_filename, "w+") as conv2d_output_file:
#     conv2d_output_file.write(conv2d_test_content)
#     print(f"Successfully wrote conv2d test to {conv2d_output_filename}")
# os.system(f"rustfmt {conv2d_output_filename}")
# print(f"Formatted conv2d test.")

# TODO:
# writing out the transposed conv2d test cases


# if __name__ == "__main__":
#     main()
