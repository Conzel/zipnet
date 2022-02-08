use ndarray::{s, Array1, Array2, Array3};

use crate::models::InternalDataRepresentation;

/// Gaussian conditional layer  as introduced by J. Ballé, D. Minnen, S. Singh,
/// S. J. Hwang, N. Johnston, in "Variational image compression with a scale
/// hyperprior" <https://arxiv.org/abs/1802.01436>.
///      
/// We follow the implementation of CompressAI, which is available at
/// <https://interdigitalinc.github.io/CompressAI/entropy_models.html>
/// or (source on github)
/// <https://github.com/InterDigitalInc/CompressAI/blob/be98bdeb65bbed05002617e51da4aa77edf44ddb/compressai/entropy_models/entropy_models.py#L328>

pub type QuantizationPrecision = u16;

trait EntropyModel {
    fn quantize(&self, x: &Array3<f32>) -> Array3<QuantizationPrecision>;
    fn compress(&self, x: &Array3<u32>) -> Array3<u32>;
    fn decompress(&self, x: &Array3<u32>) -> Array3<f32>;
}

struct EntropyBottleneck {
    /// The quantized cdf as in CompressAI.
    ///
    /// The Quantized CDF is a 2D-tensor, in which the rows are the channels
    /// of the latent representation, and the columns are the possible values
    /// of the quantized latent representation. The CDF has been scaled by
    /// 2^prec (so the values are in (0, 2^prec)).
    ///
    /// So the value QCDF[i, j] is the value CDF(j) of the i-th channel.
    ///
    /// To get the PMF, one would have to take the difference between each neighboring
    /// value and divide by 2^precision.
    ///
    /// Care has to be taken: Each CDF only has as many entries as indicated by the
    /// value in cdf_lengths. Non-existing values are set to 0 to keep the
    /// array in a valid shape.
    quantized_cdf: Array2<QuantizationPrecision>,
    /// Lengths of the CDF as described above. Length is #of latent channels.
    cdf_lengths: Array1<QuantizationPrecision>,
    /// Offset of the quantization. To get y from a symbol (^= index into CDF) back,
    /// add this to the symbol.
    offset: i32,
}

impl EntropyModel for EntropyBottleneck {
    fn quantize(&self, x: &Array3<f32>) -> Array3<QuantizationPrecision> {
        let mut quantized = Array3::zeros(x.dim());
        let num_channels = x.shape()[0];
        for i in 0..num_channels {
            let min: i32 = self.offset;
            let max: i32 = self.cdf_lengths[i] as i32 + self.offset - 1;
            let mut quantized_slice_i = x
                .slice(s![i, .., ..])
                .map(|a| ((a.round() as i32).clamp(min, max) - self.offset) as u16);
            let mut channel_i = quantized.slice_mut(s![i, .., ..]);
            channel_i += &quantized_slice_i.view_mut();
            // ensures that every quantized element is a valid index into the CDF
            debug_assert!(channel_i.iter().all(|x| *x < self.cdf_lengths[i]));
        }
        quantized
    }

    fn compress(&self, x: &Array3<u32>) -> Array3<u32> {
        todo!();
    }

    fn decompress(&self, x: &Array3<u32>) -> Array3<f32> {
        todo!();
    }
}

impl EntropyBottleneck {
    fn new(
        quantized_cdf: Array2<QuantizationPrecision>,
        cdf_lengths: Array1<QuantizationPrecision>,
        offset: i32,
    ) -> EntropyBottleneck {
        EntropyBottleneck {
            quantized_cdf,
            cdf_lengths,
            offset,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_quantization() {
        let cdf = Array2::zeros((2, 5));
        let cdf_lengths = array![3, 5];
        let offset = -1;
        let bottleneck = EntropyBottleneck::new(cdf, cdf_lengths, offset);
        let x = array![
            [[1.2, 3.4, 3.4], [0.7, -1.1, -0.2], [0.7, -1.1, -0.2]],
            [[1.3, 3.7, 1.7], [4.2, -7.3, 0.0], [4.2, -7.3, 0.0]]
        ];
        let x_hat = bottleneck.quantize(&x);
        let res = array![
            [[2, 2, 2], [2, 0, 1], [2, 0, 1]],
            [[2, 4, 3], [4, 0, 1], [4, 0, 1]]
        ];
        assert_eq!(x_hat, res);
    }
}
