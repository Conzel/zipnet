//! This module provides the hierarchical coders implementing the Encoder and Decoder
//! traits. We also provide functions to directly initialize a Hierarchical Encoder/Decoder
//! pair, using the trained weights from a python implementation.
//!
//! For the encoder, we have stuck to the paper by Minnen et al.
//! <https://arxiv.org/pdf/1809.02736.pdf>,
//! We made some adaption for the decoder as described by Johnston et al.
//! <https://arxiv.org/abs/1912.08771>.
use constriction::stream::model::DefaultContiguousCategoricalEntropyModel;
use constriction::stream::stack::AnsCoder;
use constriction::stream::Decode;
use constriction::stream::{model::DefaultLeakyQuantizer, stack::DefaultAnsCoder};
use ml::models::{CodingModel, JohnstonHyperDecoder, MinnenHyperEncoder};
use ml::models::{JohnstonDecoder, MinnenEncoder};
use ml::weight_loader::NpzWeightLoader;
use ml::ImagePrecision;
use ndarray::Array;
use ndarray::*;
use probability::distribution::Gaussian;

use crate::table_hyperpriors::{MINNEN_JOHNSTON_NUM_CHANNELS, MINNEN_JOHNSTON_SUPPORT_RANGE};
use crate::{table_hyperpriors, CodingResult, Decoder, EncodedData, Encoder};

// For quantization of the leaky gaussian. We should probably calculate this dynamically,
// but fine for now
const GAUSSIAN_SUPPORT_LOWER: i32 = -100;
const GAUSSIAN_SUPPORT_UPPER: i32 = 100;

/// A hierarchical encoder/decoder, as described in <https://arxiv.org/pdf/1809.02736.pdf>
/// (without the autoregressive part, see Fig. 19). The network architecture can be freely chosen
/// by picking appropriate coding models.
pub struct MeanScaleHierarchicalEncoder {
    latent_encoder: Box<dyn CodingModel>,
    /// The Hyperlatent encoder shall produce a vector of the form (mu_1, mu2, ..., mu_n, sigma_1, ... sigma_n)
    /// Where n is the number of latents
    hyperlatent_encoder: Box<dyn CodingModel>,
    hyperlatent_decoder: Box<dyn CodingModel>,
    /// This must be the same prior as used in the corresponding decoder
    hyperlatent_prior: TablePrior,
}

pub struct MeanScaleHierarchicalDecoder {
    latent_decoder: Box<dyn CodingModel>,
    hyperlatent_decoder: Box<dyn CodingModel>,
    /// This must be the same prior as used in the corresponding encoder
    hyperlatent_prior: TablePrior,
}

/// A probability distribution that just consists of a lookup-table.
/// Values outside of the support are pruned.
struct TablePrior {
    support: (i32, i32),
    lookup_table: &'static [f32],
}

impl TablePrior {
    /// Creates a new table prior. The support is the lowest and highest number the
    /// table has entries for. The table can be relative probabilities
    fn new(support: (i32, i32), lookup_table: &'static [f32]) -> TablePrior {
        TablePrior {
            support,
            lookup_table,
        }
    }

    /// Directly returns the underlying entropy model
    fn get_entropy_model(&self, channel: usize) -> DefaultContiguousCategoricalEntropyModel {
        DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities(
            &self.lookup_table[channel * MINNEN_JOHNSTON_SUPPORT_RANGE
                ..(channel + 1) * MINNEN_JOHNSTON_SUPPORT_RANGE],
        )
        .unwrap()
    }

    /// Takes a vector of integers and makes them conform to the entropy model of the table
    /// (symbols must normally start at 0 and may not be out of the support)
    fn to_symbols(&self, v: Vec<i32>) -> Vec<usize> {
        v.iter()
            .map(|x| {
                (if x < &self.support.0 {
                    self.support.0
                } else if x > &self.support.1 {
                    self.support.1
                } else {
                    *x
                } - self.support.0) as usize
            })
            .collect()
    }

    /// Reverse to the "to_symbols" function
    fn from_symbols(&self, symbols: Vec<usize>) -> Vec<i32> {
        symbols
            .iter()
            .map(|a| (*a as i32 + self.support.0))
            .collect()
    }

    /// Returns the hyperlatent prior that was trained on a Minnen Encoder / Johnston Decoder
    /// combo. We have pre-extracted the values from the trained python model
    /// and provide them under table_hyperprior.rs
    fn create_minnen_johnston_hyperlatent_prior() -> TablePrior {
        TablePrior::new(
            table_hyperpriors::MINNEN_JOHNSTON_SUPPORT,
            &table_hyperpriors::MINNEN_JOHNSTON_HYPERPRIOR,
        )
    }
}

fn encode_gaussians(
    coder: &mut AnsCoder<u32, u64>,
    symbols: Array1<i32>,
    means: &Array1<f64>,
    stds: &Array1<f64>,
) {
    let quantizer = DefaultLeakyQuantizer::new(GAUSSIAN_SUPPORT_LOWER..=GAUSSIAN_SUPPORT_UPPER);

    coder
        .encode_symbols_reverse(
            symbols
                .iter()
                .zip(means)
                .zip(stds)
                .map(|((&sym, &mean), &std)| (sym, quantizer.quantize(Gaussian::new(mean, std)))),
        )
        .unwrap();
}

fn decode_gaussians(
    coder: &mut AnsCoder<u32, u64>,
    means: &Array1<f64>,
    stds: &Array1<f64>,
) -> Option<Vec<i32>> {
    let quantizer = DefaultLeakyQuantizer::new(GAUSSIAN_SUPPORT_LOWER..=GAUSSIAN_SUPPORT_UPPER);

    coder
        .decode_symbols(
            means
                .iter()
                .zip(stds)
                .map(|(&mean, &std)| quantizer.quantize(Gaussian::new(mean as f64, std as f64))),
        )
        .collect::<Result<Vec<i32>, _>>()
        .ok()
}

fn make_even(i: usize) -> usize {
    if i % 2 == 0 {
        i
    } else {
        i + 1
    }
}

impl Encoder<Array3<ImagePrecision>> for MeanScaleHierarchicalEncoder {
    /// Encodes the given data, usually we assume an image.
    /// We return the encoded data, where main info consists of the actual data,
    /// the second info consists of hyperlatent_length, and latent length.
    fn encode(&mut self, data: &Array3<ImagePrecision>) -> EncodedData {
        let latents = self.latent_encoder.forward_pass(data);
        let hyperlatents = self.hyperlatent_encoder.forward_pass(&latents);
        let latent_parameters = self.hyperlatent_decoder.forward_pass(&hyperlatents);

        // If we give in an uneven latent shape, then the latent parameters will
        // have even shape and the naive comparison doesn't work
        debug_assert_eq!(
            make_even(latents.shape()[1]) * make_even(latents.shape()[2]),
            make_even(latent_parameters.shape()[1]) * make_even(latent_parameters.shape()[2]),
        );
        // ensures that we use the correct number of channels for the model
        debug_assert_eq!(hyperlatents.shape()[0], MINNEN_JOHNSTON_NUM_CHANNELS);
        debug_assert_eq!(
            latent_parameters.shape()[0],
            2 * MINNEN_JOHNSTON_NUM_CHANNELS
        );

        let flat_latents = Array::from_iter(latents.iter());

        // TODO: what to do if we have more latent parameters than latents? throw away the last
        // latents?
        let means = latent_parameters.slice(s![0..MINNEN_JOHNSTON_NUM_CHANNELS, .., ..]);
        let stds = latent_parameters
            .slice(s![
                MINNEN_JOHNSTON_NUM_CHANNELS..(2 * MINNEN_JOHNSTON_NUM_CHANNELS),
                ..,
                ..
            ])
            .map(|x| x.exp());

        let flat_means = Array::from_iter(means.iter());
        let flat_stds = Array::from_iter(stds.iter());

        let mut coder = DefaultAnsCoder::new();

        // Question: Why don't we use bits back?
        // Answer given by Bamler: It was found empiricially that bits back doesn't work :(

        // Encoding the latents y with p(y | z)
        encode_gaussians(
            &mut coder,
            flat_latents.mapv(|a| a.round() as i32),
            &flat_means.mapv(|a| *a as f64),
            &flat_stds.mapv(|a| *a as f64),
        );

        let quantized_hyperlatents = hyperlatents.map(|x| x.round() as i32);

        for i in 0..MINNEN_JOHNSTON_NUM_CHANNELS {
            // Encoding the hyperlatents z with p(z)
            coder
                .encode_iid_symbols_reverse(
                    self.hyperlatent_prior.to_symbols(
                        quantized_hyperlatents
                            .slice(s![i, .., ..])
                            .iter()
                            .map(|x| *x)
                            .collect(),
                    ),
                    &self.hyperlatent_prior.get_entropy_model(i),
                )
                .unwrap();
        }
        let data = coder.into_compressed().unwrap();

        // encoding side info
        let hl_shape = hyperlatents.shape();
        let l_shape = latents.shape();
        let side_info = vec![
            hl_shape[0] as u32,
            hl_shape[1] as u32,
            hl_shape[2] as u32,
            l_shape[0] as u32,
            l_shape[1] as u32,
            l_shape[2] as u32,
        ];
        EncodedData::new(data, side_info)
    }
}

impl Decoder<Array3<ImagePrecision>> for MeanScaleHierarchicalDecoder {
    /// The reverse of the encoding process.
    fn decode(&mut self, encoded_data: EncodedData) -> CodingResult<Array3<ImagePrecision>> {
        let side_info = encoded_data.side_info;
        let hyperlatents_shape = (
            side_info[0] as usize,
            side_info[1] as usize,
            side_info[2] as usize,
        );
        let hyperlatents_im_len = hyperlatents_shape.1 * hyperlatents_shape.2;
        // TODO: Still debate what we do about the latent even/uneven problem.
        // The make_even stuff is just a hack basically.
        let latents_shape = (
            side_info[3] as usize,
            make_even(side_info[4] as usize) as usize,
            make_even(side_info[5] as usize) as usize,
        );
        let mut coder = DefaultAnsCoder::from_compressed(encoded_data.main_info).unwrap();

        let mut hyperlatents = Array::zeros(hyperlatents_shape);
        // Decode the data:
        for i in 0..MINNEN_JOHNSTON_NUM_CHANNELS {
            let decoded_symbols: Result<Vec<usize>, _> = coder
                .decode_iid_symbols(
                    hyperlatents_im_len as usize,
                    &self.hyperlatent_prior.get_entropy_model(i),
                )
                .collect();

            let flat_hyperlatents_vec = self
                .hyperlatent_prior
                .from_symbols(decoded_symbols.unwrap());

            let hyperlatents_i = Array::from_shape_vec(
                (hyperlatents_shape.1, hyperlatents_shape.2),
                flat_hyperlatents_vec,
            )
            .unwrap();

            let mut slice = hyperlatents.slice_mut(s![i, .., ..]);
            slice.assign(&hyperlatents_i);
        }

        let latent_parameters = self
            .hyperlatent_decoder
            .forward_pass(&hyperlatents.map(|a| *a as ImagePrecision));

        // TODO: Extract to function with encoder functionality
        let means = latent_parameters.slice(s![0..MINNEN_JOHNSTON_NUM_CHANNELS, .., ..]);
        let stds = latent_parameters
            .slice(s![
                MINNEN_JOHNSTON_NUM_CHANNELS..(2 * MINNEN_JOHNSTON_NUM_CHANNELS),
                ..,
                ..
            ])
            .map(|x| x.exp());

        let flat_means = Array::from_iter(means.iter());
        let flat_stds = Array::from_iter(stds.iter());

        let flat_latents_vec = decode_gaussians(
            &mut coder,
            &flat_means.map(|a| **a as f64),
            &flat_stds.map(|a| **a as f64),
        )
        .unwrap();

        let latents = Array::from_shape_vec(latents_shape, flat_latents_vec).unwrap();

        Ok(self
            .latent_decoder
            .forward_pass(&latents.map(|a| *a as ImagePrecision)))
    }
}

impl MeanScaleHierarchicalEncoder {
    /// Returns an encoder as described in the Minnen paper.
    #[allow(non_snake_case)]
    pub fn MinnenJohnstonEncoder() -> MeanScaleHierarchicalEncoder {
        let mut loader = NpzWeightLoader::full_loader();
        let latent_encoder = Box::new(MinnenEncoder::new(&mut loader));
        let hyperlatent_encoder = Box::new(MinnenHyperEncoder::new(&mut loader));
        let hyperlatent_decoder = Box::new(JohnstonHyperDecoder::new(&mut loader));

        MeanScaleHierarchicalEncoder {
            latent_encoder,
            hyperlatent_encoder,
            hyperlatent_decoder,
            hyperlatent_prior: TablePrior::create_minnen_johnston_hyperlatent_prior(),
        }
    }
}

impl MeanScaleHierarchicalDecoder {
    #[allow(non_snake_case)]
    /// Returns decoder as described in the Johnston paper, architecture no. 5
    pub fn MinnenJohnstonDecoder() -> MeanScaleHierarchicalDecoder {
        let mut loader = NpzWeightLoader::full_loader();
        let latent_decoder = Box::new(JohnstonDecoder::new(&mut loader));
        let hyperlatent_decoder = Box::new(JohnstonHyperDecoder::new(&mut loader));

        MeanScaleHierarchicalDecoder {
            latent_decoder,
            hyperlatent_decoder,
            hyperlatent_prior: TablePrior::create_minnen_johnston_hyperlatent_prior(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_table_prior_symbols() {
        let prior = TablePrior::create_minnen_johnston_hyperlatent_prior();
        let ints = vec![0, 123847, -234762734, 1, 2];
        let symbols = prior.to_symbols(ints);
        let ints_rec = prior.from_symbols(symbols);
        assert_eq!(ints_rec, vec![0, prior.support.1, prior.support.0, 1, 2]);
    }

    #[test]
    pub fn smoke_test_decoder() {
        MeanScaleHierarchicalEncoder::MinnenJohnstonEncoder();
    }

    #[test]
    pub fn smoke_test_encoder() {
        MeanScaleHierarchicalDecoder::MinnenJohnstonDecoder();
    }
}
