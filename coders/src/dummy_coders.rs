use crate::{CodingResult, Decoder, EncodedData, Encoder};
#[allow(dead_code)]

/// A dummy coder that can be used for dummy encoding and decoding.
/// It saves the data it receives at the encoding step and reproduces it later
/// at the decoding step.
pub struct DummyCoder<T: std::fmt::Debug + Clone> {
    data: T,
}

impl<T: std::fmt::Debug + Clone> Encoder<T> for DummyCoder<T> {
    fn encode(&mut self, data: &T) -> EncodedData {
        self.data = data.clone();
        Vec::new()
    }
}

impl<T: std::fmt::Debug + Clone> Decoder<T> for DummyCoder<T> {
    fn decode(&mut self, _: &EncodedData) -> CodingResult<T> {
        Ok(self.data.clone())
    }
}
/// A dummy decoder that only returns nonsense data but may be used in prototyping systems.
pub struct ErrorDecoder {}

impl<T: std::fmt::Debug> Decoder<T> for ErrorDecoder {
    /// Dummy decode method, just returns a decode error
    fn decode(&mut self, _: &EncodedData) -> CodingResult<T> {
        return Err(crate::coding_errors::CodingError::DecodingError);
    }
}

/// A dummy encoder that only returns nonsense data but may be used in prototyping systems.
pub struct DummyEncoder {}

impl<T: std::fmt::Debug> Encoder<T> for DummyEncoder {
    /// Dummy encode method, prints a message of the encoded data and returns nonsense data.
    fn encode(&mut self, data: &T) -> EncodedData {
        println!("Stub encoder called with data {:?}", data);
        Vec::new()
    }
}
