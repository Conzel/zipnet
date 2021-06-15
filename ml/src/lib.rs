pub mod convolutions;
pub mod feed_forward;

type WeightPrecision = f32;
type ImagePrecision = f32;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
