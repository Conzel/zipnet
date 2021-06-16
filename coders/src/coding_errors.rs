use thiserror::Error;

#[derive(Error, Debug)]
pub enum CodingError {
    #[error("Could not decode data.")]
    DecodingError,
}
