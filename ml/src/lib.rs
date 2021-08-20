pub mod activation_functions;
pub mod convolutions;
pub mod fully_connected;
pub mod models;
pub mod weight_loader;

pub type WeightPrecision = f32;
pub type ImagePrecision = f32;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
