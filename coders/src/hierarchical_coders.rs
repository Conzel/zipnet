use constriction::stream::stack::AnsCoder;
use constriction::stream::{model::DefaultLeakyQuantizer, stack::DefaultAnsCoder};
use ml::models::CodingModel;
use ml::ImagePrecision;
use ndarray::*;
use probability::distribution::Gaussian;

use crate::{EncodedData, Encoder};

// For encoding and decoding, we assume a mean of 0 and a scale of 1.
// Not exactly elegant, but does the job :)
const PRIOR_MEAN: f64 = 0.0;
const PRIOR_SCALE: f64 = 1.0;

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

        // TODO: We could use bits back here I am pretty sure... :)
        // Encoding the latents
        encode_gaussians(
            &mut coder,
            flat_latents.mapv(|a| a.round() as i32),
            means.mapv(|a| *a as f64),
            stds.mapv(|a| *a as f64),
        );

        // Encoding the hyperlatents
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

// struct MinnenJohnstonCoder
