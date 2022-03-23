//! Module that contains the entropy models. Entropy models are there
//! to do the job of compression and decompression. They are used in a similar
//! way than they are in CompressAI / TensorFlow Compression.
//! For more information (and how the functions here work in detail),
//! refer to the python implementation in zipnet-torch.
//! The code here is a straight port of the python implementation.
use constriction::{
    stream::{
        model::DefaultContiguousCategoricalEntropyModel,
        queue::{DefaultRangeDecoder, DefaultRangeEncoder},
        Decode, Encode,
    },
    symbol,
};
use ndarray::{
    s, Array1, Array2, Array3, ArrayView, AsArray, Axis, Dimension, Ix2, Ix3, ShapeBuilder,
    StrideShape,
};

use crate::{models::InternalDataRepresentation, weight_loader::WeightLoader};

/// Entropy Bottleneck layer  as introduced by J. Ballé, D. Minnen, S. Singh,
/// S. J. Hwang, N. Johnston, in "Variational image compression with a scale
/// hyperprior" <https://arxiv.org/abs/1802.01436>.
///      
/// We follow the implementation of CompressAI, which is available at
/// <https://interdigitalinc.github.io/CompressAI/entropy_models.html>
/// or (source on github)
/// <https://github.com/InterDigitalInc/CompressAI/blob/be98bdeb65bbed05002617e51da4aa77edf44ddb/compressai/entropy_models/entropy_models.py#L328>

pub type QuantizationPrecision = u32;

pub trait EntropyModel {
    fn compress(&self, y: &Array3<f32>) -> Vec<u32>;
    fn decompress<Sh>(&self, y: &Vec<u32>, shape: Sh) -> Array3<f32>
    where
        Sh: ShapeBuilder<Dim = Ix3>;
}

impl EntropyModel for EntropyBottleneck {
    fn compress(&self, y: &Array3<f32>) -> Vec<u32> {
        let symbols = self.make_symbols(y);
        self.compress_symbols(&symbols)
    }
    fn decompress<Sh>(&self, y: &Vec<u32>, shape: Sh) -> Array3<f32>
    where
        Sh: ShapeBuilder<Dim = Ix3>,
    {
        let symbols = self.decompress_symbols(y, shape);
        self.unmake_symbols(&symbols)
    }
}

pub struct EntropyBottleneck {
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
    offsets: Array1<i32>,
    /// The latent variable y has a certain mean. As we want to quantize uniformly around
    /// 0, we have to subtract the mean from the variable y beforehand.
    /// The mean is given for every channel separately
    means: Array1<f32>,
}

impl EntropyBottleneck {
    fn new(
        quantized_cdf: Array2<QuantizationPrecision>,
        cdf_lengths: Array1<QuantizationPrecision>,
        offsets: Array1<i32>,
        means: Array1<f32>,
    ) -> EntropyBottleneck {
        debug_assert!(quantized_cdf.shape()[0] == cdf_lengths.shape()[0]);
        debug_assert!(quantized_cdf.shape()[0] == offsets.shape()[0]);
        debug_assert!(quantized_cdf.shape()[0] == means.shape()[0]);
        EntropyBottleneck {
            quantized_cdf,
            cdf_lengths,
            offsets,
            means,
        }
    }

    pub fn from_state_dict(loader: &mut impl WeightLoader) -> EntropyBottleneck {
        // How to automatically determine correct shape?
        let quantized_cdf = loader
            .get_weight::<_, _, i32>("entropy_bottleneck._quantized_cdf.npy", (192, 197))
            .unwrap()
            .map(|a| *a as u32);
        let offsets = loader
            .get_weight::<_, _, i32>("entropy_bottleneck._offset.npy", 192)
            .unwrap();
        let means = loader
            .get_weight::<_, _, f32>("entropy_bottleneck._medians.npy", 192)
            .unwrap();
        let cdf_lengths = loader
            .get_weight::<_, _, i32>("entropy_bottleneck._cdf_length.npy", 192)
            .unwrap()
            .map(|a| *a as u32);

        EntropyBottleneck::new(quantized_cdf, cdf_lengths, offsets, means)
    }

    /// Quantizes an array of y values to integers in the given quantization range
    fn quantize<'a, V>(&self, y: V, quant_range: (i32, i32)) -> Array2<i32>
    where
        V: AsArray<'a, f32, Ix2>,
    {
        let (q_min, q_max) = quant_range;
        let y_view: ArrayView<'a, f32, Ix2> = y.into();
        y_view.mapv(|a| a.round().clamp(q_min as f32, q_max as f32) as i32)
    }

    fn make_symbols(&self, y: &Array3<f32>) -> Array3<QuantizationPrecision> {
        let mut symbols = Array3::zeros(y.raw_dim());
        for c in 0..self.num_channels() {
            let quant_range = (
                self.offsets[c] as i32,
                self.cdf_lengths[c] as i32 + self.offsets[c] - 2,
            );
            let y_channel_shifted = y.slice(s![c, .., ..]).map(|a| a - self.means[c]);
            let y_channel_quantized = self.quantize(&y_channel_shifted, quant_range);
            let symbols_channel =
                y_channel_quantized.map(|a| (a - self.offsets[c] as i32) as QuantizationPrecision);
            let mut symbols_array_slice = symbols.slice_mut(s![c, .., ..]);
            symbols_array_slice += &symbols_channel;
        }
        return symbols;
    }

    fn unmake_symbols(&self, symbols: &Array3<QuantizationPrecision>) -> Array3<f32> {
        symbols.mapv(|a: QuantizationPrecision| a as f32)
            + self
                .offsets
                .mapv(|a| a as f32)
                .insert_axis(Axis(1))
                .insert_axis(Axis(2))
            + self.means.clone().insert_axis(Axis(1)).insert_axis(Axis(2))
    }

    fn compress_symbols(&self, symbols: &Array3<QuantizationPrecision>) -> Vec<u32> {
        let mut coder = DefaultRangeEncoder::new();
        for c in 0..self.num_channels() {
            let symbols_channel = symbols.slice(s![c, .., ..]);
            let model = self.get_model_for_channel(c);
            coder
                .encode_iid_symbols(symbols_channel.iter().map(|a| *a as usize), &model)
                .unwrap();
        }
        coder.into_compressed().unwrap()
    }

    fn get_model_for_channel(&self, channel: usize) -> DefaultContiguousCategoricalEntropyModel {
        let cdf_channel = self
            .quantized_cdf
            .slice(s![channel, 0..(self.cdf_lengths[channel] as usize)]);
        let pmf = cdf_to_pmf(cdf_channel.iter().cloned(), 16);
        DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities(&pmf).unwrap()
    }

    fn decompress_symbols<Sh>(
        &self,
        content: &Vec<u32>,
        target_shape: Sh,
    ) -> Array3<QuantizationPrecision>
    where
        Sh: ShapeBuilder<Dim = Ix3>,
    {
        let mut y_hat: Array3<QuantizationPrecision> = Array3::zeros(target_shape);
        let mut decoder = DefaultRangeDecoder::from_compressed(content).unwrap();
        let y_width = y_hat.shape()[1];
        let y_height = y_hat.shape()[2];
        let symbols_per_channel = y_width * y_height;
        for c in 0..self.num_channels() {
            let model = self.get_model_for_channel(c);
            let decoded_symbols_flat = decoder
                .decode_iid_symbols(symbols_per_channel, &model)
                .map(|r| r.ok().unwrap() as QuantizationPrecision)
                .collect();
            let decoded_symbols_shaped: Array2<QuantizationPrecision> =
                Array2::from_shape_vec((y_width, y_height), decoded_symbols_flat).unwrap();
            y_hat
                .slice_mut(s![c, .., ..])
                .assign(&decoded_symbols_shaped);
        }
        y_hat
    }

    fn num_channels(&self) -> usize {
        self.quantized_cdf.shape()[0]
    }
}

fn cdf_to_pmf<V>(quantized_cdf: V, precision: u16) -> Vec<f32>
where
    V: IntoIterator<Item = QuantizationPrecision>,
{
    let mut pmf: Vec<f32> = quantized_cdf
        .into_iter()
        .scan(0i32, |state, x| {
            let ret = (x as i32 - *state) as f32 / (2u32.pow(precision as u32) as f32);
            *state = x as i32;
            Some(ret)
        })
        .collect();
    pmf.drain(0..1); // the first element is 0
    pmf
}

#[cfg(test)]
mod tests {
    use crate::weight_loader::NpzWeightLoader;

    use super::*;
    use ndarray::{array, Array};

    fn get_example_bottleneck() -> EntropyBottleneck {
        let offsets = array![-6, -7, -7];
        let cdf_lenghts = array![13, 14, 15];
        let means = array![0.2, -0.5, 0.1];
        let quantized_cdf = array![
            [
                0, 12, 1232, 4273, 8000, 9000, 12000, 14000, 15000, 18292, 25020, 42034, 65535, 0,
                0
            ],
            [
                0, 12, 1232, 4273, 8000, 9000, 12000, 14000, 15000, 18292, 30238, 42034, 42038,
                65535, 0
            ],
            [
                0, 12, 1232, 4273, 8000, 8476, 12000, 13234, 15000, 18292, 25020, 42034, 52098,
                62000, 65535
            ],
        ];
        EntropyBottleneck::new(quantized_cdf, cdf_lenghts, offsets, means)
    }

    #[test]
    fn test_cdf_to_pdf() {
        let cdf = array![0, 8192, 16384, 24576, 32768, 40960, 49152, 57344, 65536];
        let pmf = cdf_to_pmf(cdf, 16);
        let unif = vec![0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125];
        assert_eq!(pmf, unif);
    }

    #[test]
    fn test_make_unmake_symbols_close() {
        let entropy_bottleneck = get_example_bottleneck();
        let y = Array::from_shape_vec(
            (3, 2, 2),
            vec![
                -0.3, 0.8, 1.2, 2.7, 3.5, -1.0, -2.0, 4.8, 5.2, -1.1, 0.2, 0.3,
            ],
        )
        .unwrap();
        let y_symbols = entropy_bottleneck.make_symbols(&y);
        let y_unsymboled = entropy_bottleneck.unmake_symbols(&y_symbols);

        assert!(y
            .iter()
            .zip(y_unsymboled.iter())
            .all(|(a, b)| (*a - b).abs() <= 0.5));
        assert!(y
            .iter()
            .zip(y_unsymboled.iter())
            .any(|(a, b)| (*a - b).abs() >= 0.1));
    }

    #[test]
    fn test_compress_decompress() {
        let entropy_bottleneck = get_example_bottleneck();
        let y = Array::from_shape_vec(
            (3, 2, 2),
            vec![
                -0.3, 0.8, 1.2, 2.7, 3.5, -1.0, -2.0, 4.8, 5.2, -1.1, 0.2, 0.3,
            ],
        )
        .unwrap();
        let symbols = entropy_bottleneck.make_symbols(&y);
        let compressed_symbols = entropy_bottleneck.compress_symbols(&symbols);
        let decompressed_symbols =
            entropy_bottleneck.decompress_symbols(&compressed_symbols, symbols.raw_dim());

        assert_eq!(symbols, decompressed_symbols);
    }

    #[test]
    fn smoke_test_entropy_bottleneck() {
        let mut loader = NpzWeightLoader::full_loader();
        let _entropy_bottleneck = EntropyBottleneck::from_state_dict(&mut loader);
    }
}
