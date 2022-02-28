#!/usr/bin/env python3
import jinja2
import os
import json
import copy
import sys
from ast import literal_eval


class Activation:
    """
    Activation that can be parsed by the models_template.rs Jinja2 template
    """

    def __init__(self, name, corresponding_layer):
        assert name in ["gdn", "igdn", "relu"]
        if name == "gdn":
            raise ValueError("GDN currently not supported.")
        elif name == "igdn":
            raise ValueError("GDN currently not supported.")
        elif name == "relu":
            self.relu_init()
        else:
            raise ValueError(
                "Unknown activation passed {name}, only gdn, igdn, relu accepted")

    def relu_init(self):
        self.rust_name = "ReluLayer"
        self.python_name = "relu"


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
        kernel_shape = literal_eval(specification["kernel_shape"])
        self.kernel_height = kernel_shape[0]
        self.kernel_width = kernel_shape[1]

        layer_type = specification["type"]
        if layer_type == "conv":
            self.rust_name = "ConvolutionLayer"
            self.python_name = "conv"
            self.filters = specification["filters"]
            self.channels = specification["channels"]
        elif layer_type == "conv_transpose":
            self.rust_name = "TransposedConvolutionLayer"
            self.python_name = "conv_transpose"
            # we have to swap the displayed way for transposed convolution layers
            self.channels = specification["filters"]
            self.filters = specification["channels"]
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

        self.stride = specification["stride"]


class Model:
    """
    Model that can be parsed by the models_template.rs Jinja file.
    """

    def __init__(self, specification):
        """
        specification: dict, containing all the model keys
        """
        self.rust_name = specification["rust_module_name"]
        self.python_name = specification["python_module_name"]
        layers = list(map(Layer, add_channels(specification["layers"])))
        self.layers = layers


def main(debug):
    loader = jinja2.FileSystemLoader("./templates")
    env = jinja2.Environment(loader=loader)
    template = env.get_template("models_template.rs")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ml_src_folder = os.path.join(project_root, "ml", "src")

    specification_file = open("specifications/model_specification.json", "r")
    specifications = json.load(specification_file)
    models = list(map(Model, specifications))
    models_rs_content = template.render(
        models=models, file=__file__, debug=debug)

    # writing out the models.rs file
    models_rs_output_filename = os.path.join(
        ml_src_folder, "models.rs")
    with open(models_rs_output_filename, "w+") as models_rs_output_file:
        models_rs_output_file.write(models_rs_content)
        print(
            f"Successfully wrote model specifications to {models_rs_output_filename}")
    os.system(f"rustfmt {models_rs_output_filename}")
    print(f"Formatted models.rs file.")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--debug":
        main(debug=True)
    else:
        main(debug=False)
