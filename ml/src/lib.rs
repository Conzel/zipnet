//! This crate provides the basic Machine Learning functionality to build convolutional neural
//! networks, along with the hierarchical models used for encoding/decoding purposes.
mod activation_functions;
pub mod models;
pub mod weight_loader;

/// The floating point precision we assume the weights to have.
pub type WeightPrecision = f32;
/// The floating point precision we assume the given data to have.
pub type ImagePrecision = f32;
/// Shape and precision of the convolutional kernels used.
pub type ConvKernel = ndarray::Array4<WeightPrecision>;
