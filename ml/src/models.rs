//! This module provides the hierarchical models used in the encoding and decoding process.
// This file has been automatically generated by Jinja2 via the
// script ./generate_models.py.
// Please do not change this file by hand.
use crate::{
    activation_functions::{GdnLayer, IgdnLayer, ReluLayer},
    weight_loader::WeightLoader,
    WeightPrecision,
};
use convolutions_rs::{
    convolutions::ConvolutionLayer, transposed_convolutions::TransposedConvolutionLayer, Padding,
};
use ndarray::*;

pub type InternalDataRepresentation = Array3<WeightPrecision>;

// A note on the weights:
// Naming convention:
// [architecture]_[coder]_[layer type]_[layer]_[weight type]

/// General model trait for en- and decoding
pub trait CodingModel {
    fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation;
}

impl CodingModel for ConvolutionLayer<WeightPrecision> {
    fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation {
        self.convolve(input)
    }
}

impl CodingModel for TransposedConvolutionLayer<WeightPrecision> {
    fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation {
        self.transposed_convolve(input)
    }
}

impl CodingModel for GdnLayer {
    fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation {
        self.activate(input)
    }
}

impl CodingModel for IgdnLayer {
    fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation {
        self.activate(input)
    }
}

impl CodingModel for ReluLayer {
    fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation {
        self.activate(input)
    }
}

pub struct MinnenEncoder {
    layer_0: ConvolutionLayer<WeightPrecision>,

    activation_0: GdnLayer,

    layer_1: ConvolutionLayer<WeightPrecision>,

    activation_1: GdnLayer,

    layer_2: ConvolutionLayer<WeightPrecision>,

    activation_2: GdnLayer,

    layer_3: ConvolutionLayer<WeightPrecision>,
}

impl CodingModel for MinnenEncoder {
    #[allow(clippy::let_and_return)]
    fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation {
        let x = input.clone();

        let x = self.layer_0.forward_pass(&x);

        let x = self.activation_0.forward_pass(&x);

        let x = self.layer_1.forward_pass(&x);

        let x = self.activation_1.forward_pass(&x);

        let x = self.layer_2.forward_pass(&x);

        let x = self.activation_2.forward_pass(&x);

        let x = self.layer_3.forward_pass(&x);

        x
    }
}

impl MinnenEncoder {
    pub fn new(loader: &mut impl WeightLoader) -> Self {
        let layer_0_weights = loader
            .get_weight("encoder_layer_0/kernel.npy", (5, 5, 3, 160))
            .unwrap();
        let layer_0 = ConvolutionLayer::new_tf(layer_0_weights, 2, Padding::Same);

        let activation_0_weight_0 = loader
            .get_weight(
                "analysis_transform/encoder_layer_0/gnd_0/reparam_beta.npy",
                160,
            )
            .unwrap();

        let activation_0_weight_1 = loader
            .get_weight(
                "analysis_transform/encoder_layer_0/gnd_0/reparam_gamma.npy",
                (160, 160),
            )
            .unwrap();

        let activation_0 = GdnLayer::new(activation_0_weight_0, activation_0_weight_1);

        let layer_1_weights = loader
            .get_weight("encoder_layer_1/kernel.npy", (5, 5, 160, 160))
            .unwrap();
        let layer_1 = ConvolutionLayer::new_tf(layer_1_weights, 2, Padding::Same);

        let activation_1_weight_0 = loader
            .get_weight(
                "analysis_transform/encoder_layer_1/gnd_1/reparam_beta.npy",
                160,
            )
            .unwrap();

        let activation_1_weight_1 = loader
            .get_weight(
                "analysis_transform/encoder_layer_1/gnd_1/reparam_gamma.npy",
                (160, 160),
            )
            .unwrap();

        let activation_1 = GdnLayer::new(activation_1_weight_0, activation_1_weight_1);

        let layer_2_weights = loader
            .get_weight("encoder_layer_2/kernel.npy", (5, 5, 160, 160))
            .unwrap();
        let layer_2 = ConvolutionLayer::new_tf(layer_2_weights, 2, Padding::Same);

        let activation_2_weight_0 = loader
            .get_weight(
                "analysis_transform/encoder_layer_2/gnd_2/reparam_beta.npy",
                160,
            )
            .unwrap();

        let activation_2_weight_1 = loader
            .get_weight(
                "analysis_transform/encoder_layer_2/gnd_2/reparam_gamma.npy",
                (160, 160),
            )
            .unwrap();

        let activation_2 = GdnLayer::new(activation_2_weight_0, activation_2_weight_1);

        let layer_3_weights = loader
            .get_weight("encoder_layer_3/kernel.npy", (5, 5, 160, 160))
            .unwrap();
        let layer_3 = ConvolutionLayer::new_tf(layer_3_weights, 2, Padding::Same);

        Self {
            layer_0,

            activation_0,

            layer_1,

            activation_1,

            layer_2,

            activation_2,

            layer_3,
        }
    }
}

pub struct JohnstonDecoder {
    layer_0: TransposedConvolutionLayer<WeightPrecision>,

    activation_0: IgdnLayer,

    layer_1: TransposedConvolutionLayer<WeightPrecision>,

    activation_1: IgdnLayer,

    layer_2: TransposedConvolutionLayer<WeightPrecision>,

    activation_2: IgdnLayer,

    layer_3: TransposedConvolutionLayer<WeightPrecision>,

    activation_3: IgdnLayer,
}

impl CodingModel for JohnstonDecoder {
    #[allow(clippy::let_and_return)]
    fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation {
        let x = input.clone();

        let x = self.layer_0.forward_pass(&x);

        let x = self.activation_0.forward_pass(&x);

        let x = self.layer_1.forward_pass(&x);

        let x = self.activation_1.forward_pass(&x);

        let x = self.layer_2.forward_pass(&x);

        let x = self.activation_2.forward_pass(&x);

        let x = self.layer_3.forward_pass(&x);

        let x = self.activation_3.forward_pass(&x);

        x
    }
}

impl JohnstonDecoder {
    pub fn new(loader: &mut impl WeightLoader) -> Self {
        let layer_0_weights = loader
            .get_weight("decoder_layer_0/kernel.npy", (5, 5, 79, 160))
            .unwrap();
        let layer_0 = TransposedConvolutionLayer::new_tf(layer_0_weights, 2, Padding::Same);

        let activation_0_weight_0 = loader
            .get_weight(
                "synthesis_transform/decoder_layer_0/igdn_0/reparam_beta.npy",
                79,
            )
            .unwrap();

        let activation_0_weight_1 = loader
            .get_weight(
                "synthesis_transform/decoder_layer_0/igdn_0/reparam_gamma.npy",
                (79, 79),
            )
            .unwrap();

        let activation_0 = IgdnLayer::new(activation_0_weight_0, activation_0_weight_1);

        let layer_1_weights = loader
            .get_weight("decoder_layer_1/kernel.npy", (5, 5, 22, 79))
            .unwrap();
        let layer_1 = TransposedConvolutionLayer::new_tf(layer_1_weights, 2, Padding::Same);

        let activation_1_weight_0 = loader
            .get_weight(
                "synthesis_transform/decoder_layer_1/igdn_1/reparam_beta.npy",
                22,
            )
            .unwrap();

        let activation_1_weight_1 = loader
            .get_weight(
                "synthesis_transform/decoder_layer_1/igdn_1/reparam_gamma.npy",
                (22, 22),
            )
            .unwrap();

        let activation_1 = IgdnLayer::new(activation_1_weight_0, activation_1_weight_1);

        let layer_2_weights = loader
            .get_weight("decoder_layer_2/kernel.npy", (5, 5, 43, 22))
            .unwrap();
        let layer_2 = TransposedConvolutionLayer::new_tf(layer_2_weights, 2, Padding::Same);

        let activation_2_weight_0 = loader
            .get_weight(
                "synthesis_transform/decoder_layer_2/igdn_2/reparam_beta.npy",
                43,
            )
            .unwrap();

        let activation_2_weight_1 = loader
            .get_weight(
                "synthesis_transform/decoder_layer_2/igdn_2/reparam_gamma.npy",
                (43, 43),
            )
            .unwrap();

        let activation_2 = IgdnLayer::new(activation_2_weight_0, activation_2_weight_1);

        let layer_3_weights = loader
            .get_weight("decoder_layer_3/kernel.npy", (5, 5, 3, 43))
            .unwrap();
        let layer_3 = TransposedConvolutionLayer::new_tf(layer_3_weights, 2, Padding::Same);

        let activation_3_weight_0 = loader
            .get_weight(
                "synthesis_transform/decoder_layer_3/igdn_3/reparam_beta.npy",
                3,
            )
            .unwrap();

        let activation_3_weight_1 = loader
            .get_weight(
                "synthesis_transform/decoder_layer_3/igdn_3/reparam_gamma.npy",
                (3, 3),
            )
            .unwrap();

        let activation_3 = IgdnLayer::new(activation_3_weight_0, activation_3_weight_1);

        Self {
            layer_0,

            activation_0,

            layer_1,

            activation_1,

            layer_2,

            activation_2,

            layer_3,

            activation_3,
        }
    }
}

pub struct MinnenHyperEncoder {
    layer_0: ConvolutionLayer<WeightPrecision>,

    activation_0: ReluLayer,

    layer_1: ConvolutionLayer<WeightPrecision>,

    activation_1: ReluLayer,

    layer_2: ConvolutionLayer<WeightPrecision>,
}

impl CodingModel for MinnenHyperEncoder {
    #[allow(clippy::let_and_return)]
    fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation {
        let x = input.clone();

        let x = self.layer_0.forward_pass(&x);

        let x = self.activation_0.forward_pass(&x);

        let x = self.layer_1.forward_pass(&x);

        let x = self.activation_1.forward_pass(&x);

        let x = self.layer_2.forward_pass(&x);

        x
    }
}

impl MinnenHyperEncoder {
    pub fn new(loader: &mut impl WeightLoader) -> Self {
        let layer_0_weights = loader
            .get_weight("hyperencoder_layer_0/kernel.npy", (3, 3, 160, 160))
            .unwrap();
        let layer_0 = ConvolutionLayer::new_tf(layer_0_weights, 1, Padding::Same);

        let activation_0 = ReluLayer::new();

        let layer_1_weights = loader
            .get_weight("hyperencoder_layer_1/kernel.npy", (5, 5, 160, 160))
            .unwrap();
        let layer_1 = ConvolutionLayer::new_tf(layer_1_weights, 2, Padding::Same);

        let activation_1 = ReluLayer::new();

        let layer_2_weights = loader
            .get_weight("hyperencoder_layer_2/kernel.npy", (5, 5, 160, 160))
            .unwrap();
        let layer_2 = ConvolutionLayer::new_tf(layer_2_weights, 2, Padding::Same);

        Self {
            layer_0,

            activation_0,

            layer_1,

            activation_1,

            layer_2,
        }
    }
}

pub struct JohnstonHyperDecoder {
    layer_0: TransposedConvolutionLayer<WeightPrecision>,

    activation_0: ReluLayer,

    layer_1: TransposedConvolutionLayer<WeightPrecision>,

    activation_1: ReluLayer,

    layer_2: TransposedConvolutionLayer<WeightPrecision>,
}

impl CodingModel for JohnstonHyperDecoder {
    #[allow(clippy::let_and_return)]
    fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation {
        let x = input.clone();

        let x = self.layer_0.forward_pass(&x);

        let x = self.activation_0.forward_pass(&x);

        let x = self.layer_1.forward_pass(&x);

        let x = self.activation_1.forward_pass(&x);

        let x = self.layer_2.forward_pass(&x);

        x
    }
}

impl JohnstonHyperDecoder {
    pub fn new(loader: &mut impl WeightLoader) -> Self {
        let layer_0_weights = loader
            .get_weight("hyperdecoder_layer_0/kernel.npy", (5, 5, 76, 160))
            .unwrap();
        let layer_0 = TransposedConvolutionLayer::new_tf(layer_0_weights, 2, Padding::Same);

        let activation_0 = ReluLayer::new();

        let layer_1_weights = loader
            .get_weight("hyperdecoder_layer_1/kernel.npy", (5, 5, 107, 76))
            .unwrap();
        let layer_1 = TransposedConvolutionLayer::new_tf(layer_1_weights, 2, Padding::Same);

        let activation_1 = ReluLayer::new();

        let layer_2_weights = loader
            .get_weight("hyperdecoder_layer_2/kernel.npy", (3, 3, 320, 107))
            .unwrap();
        let layer_2 = TransposedConvolutionLayer::new_tf(layer_2_weights, 1, Padding::Same);

        Self {
            layer_0,

            activation_0,

            layer_1,

            activation_1,

            layer_2,
        }
    }
}

mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::weight_loader::NpzWeightLoader;

    #[test]
    fn smoke_test_minnenencoder() {
        let mut loader = NpzWeightLoader::full_loader();
        let _encoder = MinnenEncoder::new(&mut loader);
    }

    #[test]
    fn smoke_test_johnstondecoder() {
        let mut loader = NpzWeightLoader::full_loader();
        let _encoder = JohnstonDecoder::new(&mut loader);
    }

    #[test]
    fn smoke_test_minnenhyperencoder() {
        let mut loader = NpzWeightLoader::full_loader();
        let _encoder = MinnenHyperEncoder::new(&mut loader);
    }

    #[test]
    fn smoke_test_johnstonhyperdecoder() {
        let mut loader = NpzWeightLoader::full_loader();
        let _encoder = JohnstonHyperDecoder::new(&mut loader);
    }
}
