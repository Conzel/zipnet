use constriction::stream::stack::AnsCoder;
use constriction::stream::{model::DefaultLeakyQuantizer, stack::DefaultAnsCoder};
use ml::models::CodingModel;
use ml::ImagePrecision;
use ndarray::*;
use probability::distribution::Gaussian;

use crate::{CodingResult, Decoder, EncodedData, Encoder};

// For encoding and decoding, we assume a mean of 0 and a scale of 1.
// Not exactly elegant, but does the job :)
const PRIOR_MEAN: f64 = 0.0;
const PRIOR_SCALE: f64 = 1.0;
// For quantization of the leaky gaussian. We should probably calculate this dynamically,
// but fine for now
const GAUSSIAN_SUPPORT_LOWER: i32 = -100;
const GAUSSIAN_SUPPORT_UPPER: i32 = 100;

/// A hierarchical encoder, as described in https://arxiv.org/pdf/1809.02736.pdf
/// (without the autoregressive part, see Fig. 19). The network architecture can be freely chosen
/// by picking appropriate coding models.
struct MeanScaleHierarchicalEncoder {
    latent_encoder: Box<dyn CodingModel>,
    /// The Hyperlatent encoder shall produce a vector of the form (mu_1, mu2, ..., mu_n, sigma_1, ... sigma_n)
    /// Where n is the number of latents
    hyperlatent_encoder: Box<dyn CodingModel>,
}

/// The decoding counterpart to the MeanScaleHierarchicalEncoder
struct MeanScaleHierarchicalDecoder {
    latent_decoder: Box<dyn CodingModel>,
    hyperlatent_decoder: Box<dyn CodingModel>,
}

fn encode_gaussians(
    coder: &mut AnsCoder<u32, u64>,
    symbols: Array1<i32>,
    means: Array1<f64>,
    stds: Array1<f64>,
) {
    let quantizer = DefaultLeakyQuantizer::new(-100..=100);

    coder
        .encode_symbols_reverse(
            symbols
                .iter()
                .zip(&means)
                .zip(&stds)
                .map(|((&sym, &mean), &std)| (sym, quantizer.quantize(Gaussian::new(mean, std)))),
        )
        .unwrap();
}

impl Encoder<Array3<ImagePrecision>> for MeanScaleHierarchicalEncoder {
    fn encode(&mut self, data: &Array3<ImagePrecision>) -> EncodedData {
        let latents = self.latent_encoder.forward_pass(data);
        let hyperlatents = self.hyperlatent_encoder.forward_pass(&latents);

        debug_assert_eq!(latents.len(), 2 * hyperlatents.len());

        let flat_latents = Array::from_iter(latents.iter());
        let flat_hyperlatents = Array::from_iter(hyperlatents.iter());

        let means = flat_hyperlatents.slice(s![0..latents.len()]);
        let stds = flat_hyperlatents.slice(s![latents.len()..flat_hyperlatents.len()]);

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

        // Encoding the latents y with p(y | z)
        encode_gaussians(
            &mut coder,
            flat_latents.mapv(|a| a.round() as i32),
            means.mapv(|a| *a as f64),
            stds.mapv(|a| *a as f64),
        );

        // Encoding the hyperlatents z with p(z)
        let prior_means: Array1<f64> = Array::ones(flat_hyperlatents.len()) * PRIOR_MEAN;
        let prior_stds: Array1<f64> = Array::ones(flat_hyperlatents.len()) * PRIOR_SCALE;

        encode_gaussians(
            &mut coder,
            flat_hyperlatents.mapv(|a| a.round() as i32),
            prior_means,
            prior_stds,
        );
        coder.into_compressed().unwrap()
    }
}

impl Decoder<Array3<ImagePrecision>> for MeanScaleHierarchicalEncoder {
    fn decode(&mut self, encoded_data: &EncodedData) -> CodingResult<Array3<ImagePrecision>> {
        // Create an ANS Coder with default word and state size from the compressed data:
        // (ANS uses the same type for encoding and decoding, which makes the method very flexible
        // and allows interleaving small encoding and decoding chunks, e.g., for bits-back coding.)
        let mut coder = DefaultAnsCoder::from_compressed(encoded_data).unwrap();

        // Same entropy models and quantizer we used for encoding:
        let means = [35.2, -1.7, 30.1, 71.2, -75.1];
        let stds = [10.1, 25.3, 23.8, 35.4, 3.9];
        let quantizer = DefaultLeakyQuantizer::new(-100..=100);

        // Decode the data:
        coder
            .decode_symbols(
                means
                    .iter()
                    .zip(&stds)
                    .map(|(&mean, &std)| quantizer.quantize(Gaussian::new(mean, std))),
            )
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    }
}

// struct MinnenJohnstonCoder
