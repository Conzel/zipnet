#!/usr/bin/env python3
import pathlib
import jinja2
import os
import json
import copy
import sys
import argparse
from ast import literal_eval


class Weight:
    def __init__(self, shape: tuple[int, ...], name: str):
        self.shape = shape
        self.name = name


class Layer:
    """
    Layer that can be parsed by the models_template.rs Jinja file.
    """

    def __init__(self, spec, rust_name, python_name, rust_declaration,
                 weights, other_constructor_parameters):
        """
        Initializes the layer with the given arguments
        """
        self.rust_name = rust_name
        if spec.get("python_name") is None:
            self.python_name = python_name
        else:
            self.python_name = spec["python_name"]
        self.number = spec["number"]
        # Full name of the layer (with eventual generic parameters) in Rust
        self.rust_declaration = rust_declaration
        self.weights = weights
        self.other_constructor_parameters = other_constructor_parameters


def parse_layer_from_spec(d: dict) -> Layer:
    layer_type = d["type"]
    if layer_type == "conv":
        return make_convolution_layer(d, transpose=False)
    elif layer_type == "conv_transpose":
        return make_convolution_layer(d, transpose=True)
    elif layer_type == "relu":
        return make_relu_layer(d)
    elif layer_type == "gdn":
        return make_gdn_layer(d, inverse=False)
    elif layer_type == "igdn":
        return make_gdn_layer(d, inverse=True)
    else:
        raise ValueError(f"Unknown layer type {layer_type}")


def make_relu_layer(spec: dict) -> Layer:
    return Layer(spec, "ReluLayer", "relu", "ReluLayer", [], [])


def make_gdn_layer(spec: dict, inverse: bool) -> Layer:
    gamma = Weight((spec["filters"], spec["filters"]), "gamma")
    beta = Weight((spec["filters"]), "beta")
    if inverse:
        rust_name = "IgdnLayer"
        python_name = "igdn"
    else:
        rust_name = "GdnLayer"
        python_name = "gdn"
    gdn_parameter = parse_gdn_parameter_from_string(
        spec.get("parameters", "simplified"))
    return Layer(spec, rust_name, python_name, rust_name, [beta, gamma], [gdn_parameter])


def parse_gdn_parameter_from_string(parameter: str) -> str:
    if parameter.lower() == "simplified":
        return "GdnParameters::Simplified"
    elif parameter.lower() == "normal":
        return "GdnParameters::Normal"
    else:
        raise ValueError(f"Unknown GDN parameter {parameter}")


def make_convolution_layer(spec: dict, transpose: bool) -> Layer:
    kernel_shape = literal_eval(spec["kernel_shape"])
    channels = spec["channels"]
    filters = spec["filters"]
    if transpose:
        rust_name = "TransposedConvolutionLayer"
        python_name = "conv_transpose"
        kernel = Weight(
            (int(channels), int(filters), kernel_shape[0], kernel_shape[1]), "weight")
        if spec["bias"]:
            bias = Weight((int(filters),), "bias")
        else:
            bias = None
    else:
        rust_name = "ConvolutionLayer"
        python_name = "conv"
        kernel = Weight(
            (int(filters), int(channels), kernel_shape[0], kernel_shape[1]), "weight")
        if spec["bias"]:
            bias = Weight((int(filters),), "bias")
        else:
            bias = None
    if spec.get("python_name") is not None:
        # case we need to override the default name
        python_name = spec["python_name"]
    other_constructor_parameters = [spec["stride"],
                                    parse_padding_from_string(spec["padding"])]
    return Layer(spec, rust_name, python_name, rust_name + "<WeightPrecision>", [kernel, bias], other_constructor_parameters)


def parse_padding_from_string(padding: str) -> str:
    assert padding is not None
    if padding.lower() == "same":
        return "Padding::Same"
    elif padding.lower() == "valid":
        return "Padding::Valid"
    else:
        raise ValueError(f"Unknown padding {padding}")


def add_channels(layer_spec_list: list[dict]) -> list[dict]:
    """
    Adds the channels to a list of layer specifications, always using the channels
    of the previous layers.
    """
    ret = copy.deepcopy(layer_spec_list)
    # first filling in layers that have no channels or filters.
    # these must be activation functions, so they don't change the shape.
    # we can set their filters to the same as the layer before
    for i in range(1, len(layer_spec_list)):
        if ret[i].get("filters") is None:
            assert ret[i]["type"] != "conv" and ret[i]["type"] != "conv_transpose"
            ret[i]["filters"] = int(ret[i-1]["filters"])
    # augmenting all layers with missing channels: they
    # must have the output shape of the layer before.
    for i in range(1, len(layer_spec_list)):
        ret[i]["channels"] = int(ret[i-1]["filters"])
    return ret


def add_numbers(layer_spec_list: list[dict]) -> list[dict]:
    """
    Adds the counts to the layers.
    The second instance of a conv layer would then f.e. have number 1 (ofc starting
    at 0).
    """
    names = {}
    ret = copy.deepcopy(layer_spec_list)
    for l in ret:
        name = l.get("python_name", l["type"])
        l["number"] = names.get(name, 0)
        names[name] = names.get(name, 0) + 1
    return ret


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
        augmented_layers = add_numbers(add_channels(specification["layers"]))
        self.layers = [parse_layer_from_spec(
            layer_spec) for layer_spec in augmented_layers]


def main(args):
    loader = jinja2.FileSystemLoader("./templates")
    env = jinja2.Environment(loader=loader)
    template = env.get_template("models_template.rs")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ml_src_folder = os.path.join(project_root, "ml", "src")

    specification_file = open(args.specification, "r")
    specifications = json.load(specification_file)
    models = list(map(Model, specifications))
    models_rs_content = template.render(
        models=models, file=__file__, debug=args.debug)

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
    parser = argparse.ArgumentParser(
        description='Render a model file from a specification.')
    parser.add_argument('specification', metavar='SPEC', type=pathlib.Path,
                        help='Specification we should use to create the model file.')
    parser.add_argument("--debug", action='store_true',
                        help='Debug mode. Will activate trace outputs in the model output file.')

    args = parser.parse_args()

    main(args)
