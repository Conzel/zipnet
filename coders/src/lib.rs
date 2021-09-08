pub mod coding_errors;
pub mod dummy_coders;
pub mod hierarchical_coders;
pub mod statistics;
pub mod table_hyperpriors;

use serde_derive::{Deserialize, Serialize};

/// The first vector is the actual data, the second vector can be any side information that is still
/// needed to decode the data (size, shape, ...)
#[derive(Serialize, Deserialize)]
pub struct EncodedData {
    main_info: Vec<u32>,
    side_info: Vec<u32>,
}

impl EncodedData {
    pub fn new(main_info: Vec<u32>, side_info: Vec<u32>) -> Self {
        Self {
            main_info,
            side_info,
        }
    }
}

pub type CodingResult<T> = std::result::Result<T, coding_errors::CodingError>;

/// The trait of an encoder. An encoder can take data of type T and returns a vector
/// of 32-bit integers, that contain the compress data.
pub trait Encoder<T> {
    /// Encodes the given data into a 32-bit vector.
    fn encode(&mut self, data: &T) -> EncodedData;
}

/// The trait of a decoder. An decoder can take data of type T and returns a vector
/// of 32-bit integers, that contain the compress data.
pub trait Decoder<T> {
    /// Decodes the given data.
    fn decode(&mut self, encoded_data: EncodedData) -> CodingResult<T>;
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
