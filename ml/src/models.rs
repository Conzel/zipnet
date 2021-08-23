pub type InternalDataRepresentation = Array3<ImagePrecision>;

// Names of the model weights in the weight file
// Naming convention:
// [architecture]_[coder]_[layer type]_[layer]_[weight type]
const MINNEN_ENCODER_CONV_L0_KERNEL: &str = "analysis_transform/layer_0/kernel_rdft";
const MINNEN_ENCODER_CONV_L0_BIAS: &str = "analysis_transform/layer_0/bias";
const MINNEN_ENCODER_CONV_L0_GDN_BETA: &str = "analysis_transform/layer_0/gdn_0/reparam_beta";
const MINNEN_ENCODER_CONV_L0_GDN_GAMMA: &str = "analysis_transform/layer_0/gdn_0/reparam_gamma";
const MINNEN_ENCODER_CONV_L1_KERNEL: &str = "analysis_transform/layer_1/kernel_rdft";
const MINNEN_ENCODER_CONV_L1_BIAS: &str = "analysis_transform/layer_1/bias";
const MINNEN_ENCODER_CONV_L1_GDN_BETA: &str = "analysis_transform/layer_1/gdn_1/reparam_beta";
const MINNEN_ENCODER_CONV_L1_GDN_GAMMA: &str = "analysis_transform/layer_1/gdn_1/reparam_gamma";
const MINNEN_ENCODER_CONV_L2_KERNEL: &str = "analysis_transform/layer_2/kernel_rdft";
const MINNEN_ENCODER_CONV_L2_BIAS: &str = "analysis_transform/layer_2/bias";
const MINNEN_ENCODER_CONV_L2_GDN_BETA: &str = "analysis_transform/layer_2/gdn_2/reparam_beta";
const MINNEN_ENCODER_CONV_L2_GDN_GAMMA: &str = "analysis_transform/layer_2/gdn_2/reparam_gamma";
const MINNEN_ENCODER_CONV_L3_KERNEL: &str = "analysis_transform/layer_3/kernel_rdft";
const MINNEN_ENCODER_CONV_L3_BIAS: &str = "analysis_transform/layer_3/bias";

const MINNEN_HYPERENCODER_CONV_L0_KERNEL: &str = "hyper_analysis_transform/layer_0/kernel_rdft";
const MINNEN_HYPERENCODER_CONV_L0_BIAS: &str = "hyper_analysis_transform/layer_0/bias";
const MINNEN_HYPERENCODER_CONV_L1_KERNEL: &str = "hyper_analysis_transform/layer_1/kernel_rdft";
const MINNEN_HYPERENCODER_CONV_L1_BIAS: &str = "hyper_analysis_transform/layer_1/bias";
const MINNEN_HYPERENCODER_CONV_L2_KERNEL: &str = "hyper_analysis_transform/layer_2/kernel_rdft";

const JOHNSTON_DECODER_CONV_L0_KERNEL: &str = "synthesis_transform/layer_0/kernel_rdft";
const JOHNSTON_DECODER_CONV_L0_BIAS: &str = "synthesis_transform/layer_0/bias";
const JOHNSTON_DECODER_CONV_L0_IGDN_BETA: &str = "synthesis_transform/layer_0/igdn_0/reparam_beta";
const JOHNSTON_DECODER_CONV_L0_IGDN_GAMMA: &str =
    "synthesis_transform/layer_0/igdn_0/reparam_gamma";
const JOHNSTON_DECODER_CONV_L1_KERNEL: &str = "synthesis_transform/layer_1/kernel_rdft";
const JOHNSTON_DECODER_CONV_L1_BIAS: &str = "synthesis_transform/layer_1/bias";
const JOHNSTON_DECODER_CONV_L1_IGDN_BETA: &str = "synthesis_transform/layer_1/igdn_1/reparam_beta";
const JOHNSTON_DECODER_CONV_L1_IGDN_GAMMA: &str =
    "synthesis_transform/layer_1/igdn_1/reparam_gamma";
const JOHNSTON_DECODER_CONV_L2_KERNEL: &str = "synthesis_transform/layer_2/kernel_rdft";
const JOHNSTON_DECODER_CONV_L2_BIAS: &str = "synthesis_transform/layer_2/bias";
const JOHNSTON_DECODER_CONV_L2_IGDN_BETA: &str = "synthesis_transform/layer_2/igdn_2/reparam_beta";
const JOHNSTON_DECODER_CONV_L2_IGDN_GAMMA: &str =
    "synthesis_transform/layer_2/igdn_2/reparam_gamma";
const JOHNSTON_DECODER_CONV_L3_KERNEL: &str = "synthesis_transform/layer_3/kernel_rdft";
const JOHNSTON_DECODER_CONV_L3_BIAS: &str = "synthesis_transform/layer_3/bias";

const JOHNSTON_HYPERDECODER_CONV_L0_KERNEL: &str =
    "mb_t2018hyper_synthesis_transform/layer_0/kernel_rdft";
const JOHNSTON_HYPERDECODER_CONV_L0_BIAS: &str = "mb_t2018hyper_synthesis_transform/layer_0/bias";
const JOHNSTON_HYPERDECODER_CONV_L1_KERNEL: &str =
    "mb_t2018hyper_synthesis_transform/layer_1/kernel_rdft";
const JOHNSTON_HYPERDECODER_CONV_L1_BIAS: &str = "mb_t2018hyper_synthesis_transform/layer_1/bias";
const JOHNSTON_HYPERDECODER_CONV_L2_KERNEL: &str =
    "mb_t2018hyper_synthesis_transform/layer_2/kernel_rdft";
const JOHNSTON_HYPERDECODER_CONV_L2_BIAS: &str = "mb_t2018hyper_synthesis_transform/layer_2/bias";

// hyperparameters
const NUM_ENCODER_FILTERS: usize = 192;

// Refer to Johnston et al. 2019, Decoder 5
const NUM_DECODER_FILTERS_0: usize = 79;
const NUM_DECODER_FILTERS_1: usize = 22;
const NUM_DECODER_FILTERS_2: usize = 43;
const NUM_DECODER_FILTERS_3: usize = 3;

// Analog: Hyperdecoder 5
const NUM_HYPERDECODER_FILTERS_1: usize = 76;
const NUM_HYPERDECODER_FILTERS_0: usize = 107;
const NUM_HYPERDECODER_FILTERS_2: usize = 320;

use crate::{
    activation_functions::{leaky_relu, GdnLayer, IgdnLayer},
    convolutions::{ConvolutionLayer, Padding},
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
    pub fn new(loader: &mut impl WeightLoader) -> MinnenEncoder {
        // hyperparameter
        // Also called AnalysisTransform
        // all four layers have a stride of 2 & input kernel of 5 x 5
        let l0_kernel_weights = loader.get_weight(
            MINNEN_ENCODER_CONV_L0_KERNEL,
            (NUM_ENCODER_FILTERS, 3, 5, 5),
        );
        let layer_0 = ConvolutionLayer::new(l0_kernel_weights.unwrap(), 2, Padding::Same);
        let gdn_0_beta = loader.get_weight(MINNEN_ENCODER_CONV_L0_GDN_BETA, NUM_ENCODER_FILTERS);
        let gdn_0_gamma = loader.get_weight(
            MINNEN_ENCODER_CONV_L0_GDN_GAMMA,
            (NUM_ENCODER_FILTERS, NUM_ENCODER_FILTERS),
        );
        let gdn_0 = GdnLayer::new(gdn_0_beta.unwrap(), gdn_0_gamma.unwrap());
        let l1_kernel_weights = loader.get_weight(
            MINNEN_ENCODER_CONV_L1_KERNEL,
            (NUM_ENCODER_FILTERS, NUM_ENCODER_FILTERS, 5, 5),
        );
        let layer_1 = ConvolutionLayer::new(l1_kernel_weights.unwrap(), 2, Padding::Same);
        let gdn_1_beta = loader.get_weight(MINNEN_ENCODER_CONV_L1_GDN_BETA, NUM_ENCODER_FILTERS);
        let gdn_1_gamma = loader.get_weight(
            MINNEN_ENCODER_CONV_L1_GDN_GAMMA,
            (NUM_ENCODER_FILTERS, NUM_ENCODER_FILTERS),
        );
        let gdn_1 = GdnLayer::new(gdn_1_beta.unwrap(), gdn_1_gamma.unwrap());
        let l2_kernel_weights = loader.get_weight(
            MINNEN_ENCODER_CONV_L2_KERNEL,
            (NUM_ENCODER_FILTERS, NUM_ENCODER_FILTERS, 5, 5),
        );
        let layer_2 = ConvolutionLayer::new(l2_kernel_weights.unwrap(), 2, Padding::Same);
        let gdn_2_beta = loader.get_weight(MINNEN_ENCODER_CONV_L2_GDN_BETA, NUM_ENCODER_FILTERS);
        let gdn_2_gamma = loader.get_weight(
            MINNEN_ENCODER_CONV_L2_GDN_GAMMA,
            (NUM_ENCODER_FILTERS, NUM_ENCODER_FILTERS),
        );
        let gdn_2 = GdnLayer::new(gdn_2_beta.unwrap(), gdn_2_gamma.unwrap());
        let l3_kernel_weights = loader.get_weight(
            MINNEN_ENCODER_CONV_L3_KERNEL,
            (NUM_ENCODER_FILTERS, NUM_ENCODER_FILTERS, 5, 5),
        );
        let layer_3 = ConvolutionLayer::new(l3_kernel_weights.unwrap(), 2, Padding::Same);
        MinnenEncoder {
            layer_0,
            gdn_0,
            layer_1,
            gdn_1,
            layer_2,
            gdn_2,
            layer_3,
        }
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
/// We are usuing the Decoder 5 architecture
pub struct JohnstonDecoder {
    /// 5x5deconv,2,79
    layer_0: ConvolutionLayer,
    igdn_0: IgdnLayer,
    /// 5x5deconv,2,22
    layer_1: ConvolutionLayer,
    igdn_1: IgdnLayer,
    /// 5x5deconv,2,43
    layer_2: ConvolutionLayer,
    igdn_2: IgdnLayer,
    /// 5x5deconv,2,3
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
    pub fn new(loader: &mut impl WeightLoader) -> JohnstonDecoder {
        // also known as SynthesisTransform
        let l0_kernel_weights = loader.get_weight(
            JOHNSTON_DECODER_CONV_L0_KERNEL,
            (NUM_DECODER_FILTERS_0, NUM_ENCODER_FILTERS, 5, 5),
        );
        let layer_0 = ConvolutionLayer::new(l0_kernel_weights.unwrap(), 2, Padding::Same);
        let igdn_0_gamma =
            loader.get_weight(JOHNSTON_DECODER_CONV_L0_IGDN_GAMMA, NUM_DECODER_FILTERS_0);
        let igdn_0_beta = loader.get_weight(
            JOHNSTON_DECODER_CONV_L0_IGDN_BETA,
            (NUM_DECODER_FILTERS_0, NUM_DECODER_FILTERS_0),
        );
        let igdn_0 = IgdnLayer::new(igdn_0_gamma.unwrap(), igdn_0_beta.unwrap());

        let l1_kernel_weights = loader.get_weight(
            JOHNSTON_DECODER_CONV_L1_KERNEL,
            (NUM_DECODER_FILTERS_1, NUM_DECODER_FILTERS_0, 5, 5),
        );
        let layer_1 = ConvolutionLayer::new(l1_kernel_weights.unwrap(), 2, Padding::Same);
        let igdn_1_gamma =
            loader.get_weight(JOHNSTON_DECODER_CONV_L1_IGDN_GAMMA, NUM_DECODER_FILTERS_1);
        let igdn_1_beta = loader.get_weight(
            JOHNSTON_DECODER_CONV_L1_IGDN_BETA,
            (NUM_DECODER_FILTERS_1, NUM_DECODER_FILTERS_1),
        );
        let igdn_1 = IgdnLayer::new(igdn_1_gamma.unwrap(), igdn_1_beta.unwrap());

        let l2_kernel_weights = loader.get_weight(
            JOHNSTON_DECODER_CONV_L2_KERNEL,
            (NUM_DECODER_FILTERS_2, NUM_DECODER_FILTERS_1, 5, 5),
        );
        let layer_2 = ConvolutionLayer::new(l2_kernel_weights.unwrap(), 2, Padding::Same);
        let igdn_2_gamma =
            loader.get_weight(JOHNSTON_DECODER_CONV_L2_IGDN_GAMMA, NUM_DECODER_FILTERS_2);
        let igdn_2_beta = loader.get_weight(
            JOHNSTON_DECODER_CONV_L2_IGDN_BETA,
            (NUM_DECODER_FILTERS_2, NUM_DECODER_FILTERS_2),
        );
        let igdn_2 = IgdnLayer::new(igdn_2_gamma.unwrap(), igdn_2_beta.unwrap());

        let l3_kernel_weights = loader.get_weight(
            JOHNSTON_DECODER_CONV_L3_KERNEL,
            (NUM_DECODER_FILTERS_3, NUM_DECODER_FILTERS_2, 5, 5),
        );
        let layer_3 = ConvolutionLayer::new(l3_kernel_weights.unwrap(), 2, Padding::Same);
        JohnstonDecoder {
            layer_0,
            igdn_0,
            layer_1,
            igdn_1,
            layer_2,
            igdn_2,
            layer_3,
        }
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
