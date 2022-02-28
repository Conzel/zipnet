use image::{DynamicImage, RgbImage};
use ndarray::{Array, Array3};
use nshare::ToNdarray3;

/// Turns ndarray to rgb image
///
/// Taken from <https://stackoverflow.com/questions/56762026/how-to-save-ndarray-in-rust-as-image>
pub fn array_to_image(arr: Array3<u8>) -> RgbImage {
    // we get the image in PT layout, which is (C,H,W), but need (H,W,C)
    let permuted_view = arr.view().permuted_axes([1, 2, 0]);
    // again hack to fix the memory layout
    let permuted_array: Array3<u8> = Array::from_shape_vec(
        permuted_view.dim(),
        permuted_view.iter().map(|x| *x).collect(),
    )
    .unwrap();

    assert!(permuted_array.is_standard_layout());

    let (height, width, _) = permuted_array.dim();
    let raw = permuted_array.into_raw_vec();

    RgbImage::from_raw(width as u32, height as u32, raw)
        .expect("container should have the right size for the image dimensions")
}

/// Returns the image as pre-scaled array, ready to be put into an encoder
pub fn image_to_ndarray(img: &DynamicImage) -> Array3<f32> {
    img.to_rgb8().into_ndarray3().map(|x| *x as f32 / 255.0)
}

/// Turns output from neural net into a pixel value, performs postprocessing
pub fn to_pixel(x: &f32, debug: bool) -> u8 {
    if debug {
        x.round().clamp(0.0, 255.0) as u8
    } else {
        (x.clamp(0.0, 1.0) * 255.0).round() as u8
    }
}
