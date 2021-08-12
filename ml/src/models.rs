pub type InternalDataRepresentation = Array3<ImagePrecision>;

// Names of the model weights in the weight file
// Naming convention:
// [architecture]_[coder]_[layer type]_[layer]_[weight type]
const MINNEN_ENCODER_CONV_L0_KERNEL: &str = "analysis_transform/layer_0/kernel_rdft";
const MINNEN_ENCODER_CONV_L0_BIAS: &str = "analysis_transform/layer_0/bias";
const MINEN_ENCODER_CONV_L0_GDN_BETA: &str = "analysis_transform/layer_0/gdn_0/reparam_beta";
const MINEN_ENCODER_CONV_L0_GDN_GAMMA: &str = "analysis_transform/layer_0/gdn_0/reparam_gamma";
const MINEN_ENCODER_CONV_L1_KERNEL: &str = "analysis_transform/layer_1/kernel_rdft";
const MINEN_ENCODER_CONV_L1_BIAS: &str = "analysis_transform/layer_1/bias";
const MINEN_ENCODER_CONV_L1_GDN_BETA: &str = "analysis_transform/layer_1/gdn_1/reparam_beta";
const MINEN_ENCODER_CONV_L1_GDN_GAMMA: &str = "analysis_transform/layer_1/gdn_1/reparam_gamma";
const MINEN_ENCODER_CONV_L2_KERNEL: &str = "analysis_transform/layer_2/kernel_rdft";
const MINEN_ENCODER_CONV_L2_BIAS: &str = "analysis_transform/layer_2/bias";
const MINEN_ENCODER_CONV_L2_GDN_BETA: &str = "analysis_transform/layer_2/gdn_2/reparam_beta";
const MINEN_ENCODER_CONV_L2_GDN_GAMMA: &str = "analysis_transform/layer_2/gdn_2/reparam_gamma";
const MINEN_ENCODER_CONV_L3_KERNEL: &str = "analysis_transform/layer_3/kernel_rdft";
const MINEN_ENCODER_CONV_L3_BIAS: &str = "analysis_transform/layer_3/bias";

use crate::{
    activation_functions::{leaky_relu, GdnLayer, IgdnLayer},
    convolutions::ConvolutionLayer,
    weight_loader::WeightLoader,
    ImagePrecision,
};
use ndarray::*;

/// General model trait for en- and decoding
pub trait CodingModel {
    fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation;
}

impl CodingModel for ConvolutionLayer {
    fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation {
        self.convolve(input)
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
/// Encoder / Hyperencoder pair as described in Minnen et al 2018,
/// https://arxiv.org/pdf/1809.02736.pdf without the autoregressive part
/// All layers are 5x5conv2,192, with kernel 5x5, stride 2, output channels 192
pub struct MinnenEncoder {
    layer_0: ConvolutionLayer,
    gdn_0: GdnLayer,
    layer_1: ConvolutionLayer,
    gdn_1: GdnLayer,
    layer_2: ConvolutionLayer,
    gdn_2: GdnLayer,
    layer_3: ConvolutionLayer,
}

impl CodingModel for MinnenEncoder {
    fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation {
        let x = self.layer_0.forward_pass(input);
        let x = self.gdn_0.forward_pass(&x);
        let x = self.layer_1.forward_pass(&x);
        let x = self.gdn_1.forward_pass(&x);
        let x = self.layer_2.forward_pass(&x);
        let x = self.gdn_2.forward_pass(&x);
        let x = self.layer_3.forward_pass(&x);
        x
    }
}

impl<'a> MinnenEncoder {
    pub fn new(loader: &'a impl WeightLoader<'a>) -> MinnenEncoder {
        let l0_kernel_weights = loader.get_weight(MINNEN_ENCODER_CONV_L0_KERNEL, (5, 5));
        // idk if the padding is correct
        // why are the weights 4 dimensional?
        // let layer_0 = ConvolutionLayer::new(l0_kernel_weights.unwrap(), 2, 1);
        todo!()
    }
}

pub struct MinnenHyperencoder {
    /// 3x3conv,1,320
    layer_0: ConvolutionLayer,
    /// 5x5conv,2,320
    layer_1: ConvolutionLayer,
    /// 5x5conv,2,320
    layer_2: ConvolutionLayer,
}

impl CodingModel for MinnenHyperencoder {
    fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation {
        let x = self.layer_0.forward_pass(input);
        let x = leaky_relu(&x);
        let x = self.layer_1.forward_pass(&x);
        let x = leaky_relu(&x);
        let x = self.layer_2.forward_pass(&x);
        x
    }
}

impl MinnenHyperencoder {
    pub fn new() -> MinnenHyperencoder {
        todo!()
    }
}

/// Decoder / Hyperdecoder pair described in Johnston et al 2019, https://arxiv.org/abs/1912.08771
/// Optimized Architecture for Decoder with MorphNet filter search.
/// We are usuing Dthe Decoder 5 architecture
pub struct JohnstonDecoder {
    /// 5x5deconv,2,76
    layer_0: ConvolutionLayer,
    igdn_0: IgdnLayer,
    /// 5x5deconv,2,107
    layer_1: ConvolutionLayer,
    igdn_1: IgdnLayer,
    /// 3x3deconv,1,320
    layer_2: ConvolutionLayer,
    igdn_2: IgdnLayer,
    /// 5x5conv,2,3
    layer_3: ConvolutionLayer,
}

impl CodingModel for JohnstonDecoder {
    fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation {
        let x = self.layer_0.forward_pass(input);
        let x = self.igdn_0.forward_pass(&x);
        let x = self.layer_1.forward_pass(&x);
        let x = self.igdn_1.forward_pass(&x);
        let x = self.layer_2.forward_pass(&x);
        let x = self.igdn_2.forward_pass(&x);
        let x = self.layer_3.forward_pass(&x);
        x
    }
}

impl JohnstonDecoder {
    pub fn new() -> JohnstonDecoder {
        todo!()
    }
}

pub struct JohnstonHyperdecoder {
    /// 5x5deconv,2,76
    layer_0: ConvolutionLayer,
    /// 5x5deconv,2,107
    layer_1: ConvolutionLayer,
    /// 3x3deconv,1,320
    layer_2: ConvolutionLayer,
}

impl CodingModel for JohnstonHyperdecoder {
    fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation {
        let x = self.layer_0.forward_pass(input);
        let x = leaky_relu(&x);
        let x = self.layer_1.forward_pass(&x);
        let x = leaky_relu(&x);
        let x = self.layer_2.forward_pass(&x);
        x
    }
}

impl JohnstonHyperdecoder {
    pub fn new() -> JohnstonHyperdecoder {
        todo!()
    }
}
