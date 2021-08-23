use constriction::stream::model::{
    DefaultContiguousCategoricalEntropyModel, NonContiguousCategoricalDecoderModel,
};
use constriction::stream::stack::AnsCoder;
use constriction::stream::{model::DefaultLeakyQuantizer, stack::DefaultAnsCoder};
use constriction::stream::{Decode, Encode};
use ml::models::CodingModel;
use ml::models::{JohnstonDecoder, JohnstonHyperdecoder, MinnenEncoder, MinnenHyperencoder};
use ml::weight_loader::{NpzWeightLoader, WeightLoader};
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

/// A hierarchical encoder/decoder, as described in https://arxiv.org/pdf/1809.02736.pdf
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

pub struct GaussianPrior {
    means: Array1<f64>,
    std: Array1<f64>,
}

impl GaussianPrior {
    pub fn new(means: Array1<f64>, std: Array1<f64>) -> GaussianPrior {
        GaussianPrior { means, std }
    }
}

/// A probability distribution that just consists of a lookup-table.
/// Values outside of the support are pruned.
pub struct TablePrior {
    support: (i32, i32),
    lookup_table: &'static [f32], // entropy_model: DefaultContiguousCategoricalEntropyModel,
}

impl TablePrior {
    /// Creates a new table prior. The support is the lowest and highest number the
    /// table has entries for. The table can be relative probabilities
    pub fn new(support: (i32, i32), lookup_table: &'static [f32]) -> TablePrior {
        TablePrior {
            support,
            lookup_table,
        }
    }

    pub fn get_entropy_model(&self, channel: usize) -> DefaultContiguousCategoricalEntropyModel {
        DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities(
            &self.lookup_table[channel * MINNEN_JOHNSTON_SUPPORT_RANGE
                ..(channel + 1) * MINNEN_JOHNSTON_SUPPORT_RANGE],
        )
        .unwrap()
    }

    /// Takes a vector of integers and makes them conform to the entropy model of the table
    /// (symbols must normally start at 0 and may not be out of the support)
    pub fn to_symbols(&self, v: Vec<i32>) -> Vec<usize> {
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
    pub fn from_symbols(&self, symbols: Vec<usize>) -> Vec<i32> {
        symbols
            .iter()
            .map(|a| (a + self.support.0 as usize) as i32)
            .collect()
    }

    pub fn create_minnen_johnston_hyperlatent_prior() -> TablePrior {
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

impl Encoder<Array3<ImagePrecision>> for MeanScaleHierarchicalEncoder {
    /// The encoded Data contains the actual data in the first tuple entry.
    /// The second entry consists of
    /// [hyperlatent_length, ]
    fn encode(&mut self, data: &Array3<ImagePrecision>) -> EncodedData {
        let latents = self.latent_encoder.forward_pass(data);
        let hyperlatents = self.hyperlatent_encoder.forward_pass(&latents);
        let latent_parameters = self.hyperlatent_decoder.forward_pass(&hyperlatents);

        let latent_length = latents.len();

        debug_assert_eq!(latent_length, 2 * latent_parameters.len());
        // ensures that we use the correct number of channels for the model
        debug_assert_eq!(hyperlatents.shape()[0], MINNEN_JOHNSTON_NUM_CHANNELS);

        let flat_latents = Array::from_iter(latents.iter());
        // let flat_hyperlatents = Array::from_iter(hyperlatents.iter());
        let flat_latent_parameters = Array::from_iter(latent_parameters.iter());

        // This slice here is wrong, we need to slice along the last axis...
        let means = flat_latent_parameters.slice(s![0..latent_length]);
        let stds = flat_latent_parameters.slice(s![latent_length..flat_latent_parameters.len()]);

        let mut coder = DefaultAnsCoder::new();

        // Question: Why don't we use bits back?
        // Answer given by Bamler: It was found empiricially that bits back doesn't work :(

        // Encoding the latents y with p(y | z)
        encode_gaussians(
            &mut coder,
            flat_latents.mapv(|a| a.round() as i32),
            &means.mapv(|a| *a as f64),
            &stds.mapv(|a| *a as f64),
        );

        let quantized_hyperlatents = hyperlatents.map(|x| x.round() as i32);

        for i in 0..MINNEN_JOHNSTON_NUM_CHANNELS {
            // Encoding the hyperlatents z with p(z)
            coder.encode_iid_symbols_reverse(
                self.hyperlatent_prior.to_symbols(
                    quantized_hyperlatents
                        .slice(s![i, .., ..])
                        .iter()
                        .map(|x| *x)
                        .collect(),
                ),
                &self.hyperlatent_prior.get_entropy_model(i),
            );
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
        (data, side_info)
    }
}

impl Decoder<Array3<ImagePrecision>> for MeanScaleHierarchicalDecoder {
    fn decode(&mut self, encoded_data: EncodedData) -> CodingResult<Array3<ImagePrecision>> {
        let side_info = encoded_data.1;
        let hyperlatents_shape = (
            side_info[0] as usize,
            side_info[1] as usize,
            side_info[2] as usize,
        );
        let hyperlatents_im_len = hyperlatents_shape.1 * hyperlatents_shape.2;
        let hyperlatents_len = hyperlatents_shape.0 * hyperlatents_im_len;
        let latents_shape = (
            side_info[3] as usize,
            side_info[4] as usize,
            side_info[5] as usize,
        );
        let latents_len = latents_shape.0 * latents_shape.1 * latents_shape.2;
        let mut coder = DefaultAnsCoder::from_compressed(encoded_data.0).unwrap();

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
        let flat_latent_parameters = Array::from_iter(latent_parameters);

        let means = flat_latent_parameters.slice(s![0..latents_len]);
        let stds = flat_latent_parameters.slice(s![latents_len..flat_latent_parameters.len()]);

        let flat_latents_vec = decode_gaussians(
            &mut coder,
            &means.map(|a| *a as f64),
            &stds.map(|a| *a as f64),
        )
        .unwrap();

        let latents = Array::from_shape_vec(latents_shape, flat_latents_vec).unwrap();

        Ok(self
            .latent_decoder
            .forward_pass(&latents.map(|a| *a as ImagePrecision)))
    }
}

impl MeanScaleHierarchicalEncoder {
    pub fn MinnenEncoder() -> MeanScaleHierarchicalEncoder {
        let mut loader = NpzWeightLoader::full_loader();
        let latent_encoder = Box::new(MinnenEncoder::new(&mut loader));
        let hyperlatent_encoder = Box::new(MinnenHyperencoder::new());
        let hyperlatent_decoder = Box::new(MinnenHyperencoder::new());

        MeanScaleHierarchicalEncoder {
            latent_encoder,
            hyperlatent_encoder,
            hyperlatent_decoder,
            hyperlatent_prior: TablePrior::create_minnen_johnston_hyperlatent_prior(),
        }
    }
}

impl MeanScaleHierarchicalEncoder {
    pub fn JohnstonDecoder() -> MeanScaleHierarchicalDecoder {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
