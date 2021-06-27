use crate::{convolutions::ConvolutionLayer, ImagePrecision, WeightPrecision};
use ndarray::*;

/// General model trait for en- and decoding
pub trait CodingModel {
    fn forward_pass(&self, input: &Array2<ImagePrecision>) -> Array2<ImagePrecision>;
}

/// Encoder / Hyperencoder pair as described in Minnen et al 2018,
/// https://arxiv.org/pdf/1809.02736.pdf without the autoregressive part
/// All layers are 5x5conv2,192, with kernel 5x5, stride 2, output channels 192
struct MinnenEncoder {
    layer_0: ConvolutionLayer,
    layer_1: ConvolutionLayer,
    layer_2: ConvolutionLayer,
    layer_3: ConvolutionLayer,
}
struct MinnenHyperencoder {
    /// 3x3conv,1,320
    layer_0: ConvolutionLayer,
    /// 5x5conv,2,320
    layer_1: ConvolutionLayer,
    /// 5x5conv,2,320
    layer_2: ConvolutionLayer,
}

impl CodingModel for MinnenEncoder {
    fn forward_pass(&self, input: &Array2<ImagePrecision>) -> Array2<ImagePrecision> {
        todo!()
    }
}

/// Decoder / Hyperdecoder pair described in Johnston et al 2019, https://arxiv.org/abs/1912.08771
/// Optimized Architecture for Decoder with MorphNet filter search.
/// We are usuing Dthe Decoder 5 architecture
struct JohnstonDecoder {
    /// 5x5deconv,2,76
    layer_0: ConvolutionLayer,
    /// 5x5deconv,2,107
    layer_1: ConvolutionLayer,
    /// 3x3deconv,1,320
    layer_2: ConvolutionLayer,
    /// 5x5conv,2,3
    layer_3: ConvolutionLayer,
}
struct JohnstonHyperdecoder {
    /// 5x5deconv,2,76
    layer_0: ConvolutionLayer,
    /// 5x5deconv,2,107
    layer_1: ConvolutionLayer,
    /// 3x3deconv,1,320
    layer_2: ConvolutionLayer,
}
