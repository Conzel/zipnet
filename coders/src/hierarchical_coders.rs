use constriction::stream::stack::AnsCoder;
use constriction::stream::{model::DefaultLeakyQuantizer, stack::DefaultAnsCoder};
use constriction::stream::{Decode, Encode};
use ml::models::CodingModel;
use ml::models::{JohnstonDecoder, JohnstonHyperdecoder, MinnenEncoder, MinnenHyperencoder};
use ml::ImagePrecision;
use ndarray::Array;
use ndarray::*;
use probability::distribution::Gaussian;

use crate::{CodingResult, Decoder, EncodedData, Encoder};

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
    hyperlatent_prior: GaussianPrior,
    latent_size: usize,
    latent_shape: (usize, usize, usize),
}

pub struct MeanScaleHierarchicalDecoder {
    latent_decoder: Box<dyn CodingModel>,
    hyperlatent_decoder: Box<dyn CodingModel>,
    /// This must be the same prior as used in the corresponding encoder
    hyperlatent_prior: GaussianPrior,
    latent_size: usize,
    latent_shape: (usize, usize, usize),
    hyperlatent_shape: (usize, usize, usize),
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
    lookup_table: Vec<f32>,
}

impl TablePrior {
    /// Creates a new table prior. The support is the lowest and highest number the
    /// table has entries for. The table can be relative probabilities and will be normalized
    /// in the construction
    pub fn new(support: (i32, i32), lookup_table: Vec<f32>) -> TablePrior {
        let c: f32 = lookup_table.iter().sum();
        let normalized_table = lookup_table.iter().map(|a| a / c).collect();
        TablePrior {
            support,
            lookup_table: normalized_table,
        }
    }

    /// Returns the probability at the given value.
    pub fn get(&self, x: i32) -> f32 {
        let x_idx = if x < self.support.0 {
            self.support.0
        } else if x > self.support.1 {
            self.support.1
        } else {
            x
        };

        self.lookup_table[(x_idx - self.support.0) as usize]
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

fn deocde_gaussians(
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
    fn encode(&mut self, data: &Array3<ImagePrecision>) -> EncodedData {
        let latents = self.latent_encoder.forward_pass(data);
        let hyperlatents = self.hyperlatent_encoder.forward_pass(&latents);
        let latent_parameters = self.hyperlatent_decoder.forward_pass(&hyperlatents);

        debug_assert_eq!(latents.len(), 2 * hyperlatents.len());

        let flat_latents = Array::from_iter(latents.iter());
        let flat_hyperlatents = Array::from_iter(hyperlatents.iter());
        let flat_latent_parameters = Array::from_iter(latent_parameters.iter());

        debug_assert_eq!(latents.len(), self.latent_size);

        // This slice here is wrong, we need to slice along the last axis...
        let means = flat_latent_parameters.slice(s![0..self.latent_size]);
        let stds = flat_latent_parameters.slice(s![self.latent_size..flat_latent_parameters.len()]);

        let mut coder = DefaultAnsCoder::new();

        // Question: Why don't we use bits back?
        // It would work as follows
        // We encode the array using bits back coding. We denote the latents by y,
        // the hyperlatents by z. This means:
        // decode z, using p(z | y)
        // encode y, using p(y | z)
        // encode z, using p(z)
        // Where the probability distributions are accessed via the following:
        // p(z | y): Synthesis transform of the Hyperlatent Coder (^= Hyperlatent Decoder)
        // p(y | z): Analysis transform of the Hyperlatent Coder (^= Hyperlatent Encoder)
        // p(z): Prior of the Hyperlatent Coder

        // Answer given by Bamler: It was found empiricially that bits back doesn't work :(

        // Encoding the latents y with p(y | z)
        encode_gaussians(
            &mut coder,
            flat_latents.mapv(|a| a.round() as i32),
            &means.mapv(|a| *a as f64),
            &stds.mapv(|a| *a as f64),
        );

        // Encoding the hyperlatents z with p(z)
        encode_gaussians(
            &mut coder,
            flat_hyperlatents.mapv(|a| a.round() as i32),
            &self.hyperlatent_prior.means,
            &self.hyperlatent_prior.std,
        );
        coder.into_compressed().unwrap()
    }
}

impl Decoder<Array3<ImagePrecision>> for MeanScaleHierarchicalDecoder {
    fn decode(&mut self, encoded_data: EncodedData) -> CodingResult<Array3<ImagePrecision>> {
        let mut coder = DefaultAnsCoder::from_compressed(encoded_data).unwrap();

        // Decode the data:
        let flat_hyperlatents_vec = deocde_gaussians(
            &mut coder,
            &self.hyperlatent_prior.means,
            &self.hyperlatent_prior.std,
        )
        .unwrap();

        let hyperlatents =
            Array::from_shape_vec(self.hyperlatent_shape, flat_hyperlatents_vec).unwrap();
        let latent_parameters = self
            .hyperlatent_decoder
            .forward_pass(&hyperlatents.map(|a| *a as ImagePrecision));
        let flat_latent_parameters = Array::from_iter(latent_parameters);

        let means = flat_latent_parameters.slice(s![0..self.latent_size]);
        let stds = flat_latent_parameters.slice(s![self.latent_size..flat_latent_parameters.len()]);

        let flat_latents_vec = deocde_gaussians(
            &mut coder,
            &means.map(|a| *a as f64),
            &stds.map(|a| *a as f64),
        )
        .unwrap();

        let latents = Array::from_shape_vec(self.hyperlatent_shape, flat_latents_vec).unwrap();

        Ok(self
            .latent_decoder
            .forward_pass(&latents.map(|a| *a as ImagePrecision)))
    }
}

// Implementation of the coders
// For encoding and decoding, we assume a mean of 0 and a scale of 1.
// Not exactly elegant, but does the job :)
const PRIOR_MEAN: f64 = 0.0;
const PRIOR_SCALE: f64 = 1.0;

impl MeanScaleHierarchicalEncoder {
    pub fn MinnenEncoder() -> MeanScaleHierarchicalEncoder {
        let latent_encoder = Box::new(MinnenEncoder::new());
        let hyperlatent_encoder = Box::new(MinnenHyperencoder::new());
        let hyperlatent_decoder = Box::new(MinnenHyperencoder::new());

        let latent_shape = (8, 8, 8);
        let latent_size = latent_shape.0 * latent_shape.1 * latent_shape.2;
        let hyperlatent_mean = Array::ones(latent_size) * PRIOR_MEAN;
        let hyperlatent_std = Array::ones(latent_size) * PRIOR_SCALE;
        let hyperlatent_prior = GaussianPrior::new(hyperlatent_mean, hyperlatent_std);

        MeanScaleHierarchicalEncoder {
            latent_encoder,
            hyperlatent_encoder,
            hyperlatent_decoder,
            hyperlatent_prior,
            latent_size,
            latent_shape,
        }
    }
}

impl MeanScaleHierarchicalEncoder {
    pub fn JohnstonDecoder() -> MeanScaleHierarchicalDecoder {
        let latent_encoder = Box::new(MinnenEncoder::new());
        let latent_decoder = Box::new(JohnstonDecoder::new());
        let hyperlatent_decoder = Box::new(MinnenHyperencoder::new());

        // TODO: Replace with correct latent shape (have to give through argument?)
        let latent_shape = (8, 8, 8);
        let latent_size = latent_shape.0 * latent_shape.1 * latent_shape.2;
        let hyperlatent_shape = (8, 8, 8);
        let hyperlatent_mean = Array::ones(latent_size) * PRIOR_MEAN;
        let hyperlatent_std = Array::ones(latent_size) * PRIOR_SCALE;
        let hyperlatent_prior = GaussianPrior::new(hyperlatent_mean, hyperlatent_std);

        MeanScaleHierarchicalDecoder {
            latent_decoder: latent_decoder,
            hyperlatent_decoder,
            hyperlatent_prior,
            latent_size,
            latent_shape,
            hyperlatent_shape,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_prior_test() {
        let table = vec![0.5, 0.8, 0.7];
        let prior = TablePrior::new((-1, 1), table);
        assert_eq!(prior.get(-1), 0.25);
        assert_eq!(prior.get(-2), 0.25);
        assert_eq!(prior.get(-237466234), 0.25);
        assert_eq!(prior.get(1), 0.35);
        assert_eq!(prior.get(2), 0.35);
        assert_eq!(prior.get(237466234), 0.35);
        assert_eq!(prior.get(0), 0.4);
    }
}
