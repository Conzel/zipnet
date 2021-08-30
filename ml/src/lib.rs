pub mod activation_functions;
pub mod convolutions;
pub mod models;
pub mod transposed_convolutions;
pub mod weight_loader;

pub type WeightPrecision = f32;
pub type ImagePrecision = f32;
pub type ConvKernel = ndarray::Array4<WeightPrecision>;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
