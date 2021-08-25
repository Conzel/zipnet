/// Transposed convolutions (also wrongly called deconvolution layers)
/// are learnable upsampling maps.
/// More can be read here:
/// - https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers
/// - https://www.youtube.com/watch?v=ByjaPdWXKJ4&t=1019s
/// - https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
use crate::{
    convolutions::{ConvolutionLayer, Padding},
    models::{CodingModel, InternalDataRepresentation},
    ConvKernel,
};
use ndarray::*;

/// Analog to a Convolution Layer
pub struct TransposedConvolutionLayer {
    convolution_layer: ConvolutionLayer,
}

impl CodingModel for TransposedConvolutionLayer {
    fn forward_pass(&self, input: &InternalDataRepresentation) -> InternalDataRepresentation {
        todo!()
    }
}

impl TransposedConvolutionLayer {
    pub fn new(weights: ConvKernel, stride: usize, padding: Padding) -> TransposedConvolutionLayer {
        TransposedConvolutionLayer {
            convolution_layer: ConvolutionLayer::new(weights, stride, padding),
        }
    }

    /// Performs a transposed convolution on the input image. This upsamples the image.
    /// More explanation can be read here:
    /// - https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers
    pub fn transposed_convolve(
        &self,
        image: &InternalDataRepresentation,
    ) -> InternalDataRepresentation {
        // TODO: Implement Transposed Convolution.
        // Can be done like a backwards pass of normal convolution
        todo!()
    }
}

#[cfg(test)]
mod tests {}
