use constriction::stream::chain::DecoderError;
use ml::{
    entropy_models::{EntropyBottleneck, EntropyModel},
    models::{CodingModel, JohnstonDecoder, MinnenEncoder},
    weight_loader::NpzWeightLoader,
};
use ndarray::Array3;

use crate::{coding_errors::CodingError, CodingResult, Decoder, EncodedData, Encoder};

pub struct FactorizedPriorEncoder {
    analysis_transform: MinnenEncoder,
    entropy_bottleneck: EntropyBottleneck,
}

pub struct FactorizedPriorDecoder {
    synthesis_transform: JohnstonDecoder,
    entropy_bottleneck: EntropyBottleneck,
}

impl Encoder<Array3<f32>> for FactorizedPriorEncoder {
    fn encode(&mut self, input: &Array3<f32>) -> EncodedData {
        let latents = self.analysis_transform.forward_pass(input);
        let encoded = self.entropy_bottleneck.compress(&latents);
        let s = latents.shape();
        EncodedData::new(encoded, vec![s[0] as u32, s[1] as u32, s[2] as u32])
    }
}

impl Decoder<Array3<f32>> for FactorizedPriorDecoder {
    fn decode(&mut self, encoded: EncodedData) -> CodingResult<Array3<f32>> {
        let latents_encoded = encoded.main_info;
        let s = encoded.side_info;
        if s.len() != 3 {
            return Err(CodingError::DecodingError);
        }
        assert_eq!(s.len(), 3);
        let y_hat = self.entropy_bottleneck.decompress(
            &latents_encoded,
            (s[0] as usize, s[1] as usize, s[2] as usize),
        );
        Ok(self.synthesis_transform.forward_pass(&y_hat))
    }
}

impl FactorizedPriorEncoder {
    pub fn new() -> FactorizedPriorEncoder {
        let mut loader_encoder = NpzWeightLoader::full_loader();
        let mut loader_entropy = NpzWeightLoader::full_loader();
        FactorizedPriorEncoder {
            analysis_transform: MinnenEncoder::new(&mut loader_encoder),
            entropy_bottleneck: EntropyBottleneck::from_state_dict(&mut loader_entropy),
        }
    }
}

impl FactorizedPriorDecoder {
    pub fn new() -> FactorizedPriorDecoder {
        let mut loader_decoder = NpzWeightLoader::full_loader();
        let mut loader_entropy = NpzWeightLoader::full_loader();
        FactorizedPriorDecoder {
            synthesis_transform: JohnstonDecoder::new(&mut loader_decoder),
            entropy_bottleneck: EntropyBottleneck::from_state_dict(&mut loader_entropy),
        }
    }
}
