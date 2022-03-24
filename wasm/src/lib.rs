extern crate js_sys;
extern crate web_sys;
use coders::{
    factorized_prior_coders::{FactorizedPriorDecoder, FactorizedPriorEncoder},
    Decoder, Encoder,
};
use image::{load_from_memory_with_format, png::PngEncoder, ImageFormat};

mod utils;

use wasm_bindgen::prelude::*;
use zipnet::{array_to_image, image_to_ndarray, to_pixel};

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

/// Takes in the data of a png and returns.
#[wasm_bindgen]
pub fn encode_image(buffer: Vec<u8>) -> Vec<u8> {
    wasm_logger::init(wasm_logger::Config::default());
    console_error_panic_hook::set_once();

    let image = load_from_memory_with_format(&buffer, ImageFormat::Png).unwrap();
    log::info!("Successfully loaded image. Starting encoding...");
    let mut encoder: Box<dyn Encoder<_>> = Box::new(FactorizedPriorEncoder::new());

    let encoded = encoder.encode(&image_to_ndarray(&image));
    let encoded_bin = bincode::serialize(&encoded).unwrap();
    log::info!("Finished encoding.");

    encoded_bin
}

/// Takes in the binary data of an encoded image data and returns the binary representation of
/// a png.
#[wasm_bindgen]
pub fn decode_image(encoded_bin: Vec<u8>) -> Vec<u8> {
    wasm_logger::init(wasm_logger::Config::default());
    console_error_panic_hook::set_once();
    let mut decoder: Box<dyn Decoder<_>> = Box::new(FactorizedPriorDecoder::new());

    let encoded = bincode::deserialize(&encoded_bin).unwrap();

    log::info!("Successfully loaded encoded representation. Starting decoding...");
    let decoded = decoder.decode(encoded).unwrap();
    log::info!("Finished decoding. Image has size {:?}", decoded.shape());
    let img = array_to_image(decoded.map(|x| to_pixel(x, false)));
    let mut buffer = Vec::new();
    // according to:
    // https://www.reddit.com/r/learnrust/comments/jv2ker/extracting_vecu8_from_a_png_in_memory/
    let png_encoder = PngEncoder::new(&mut buffer);
    png_encoder
        .encode(
            &img,
            decoded.shape()[1] as u32,
            decoded.shape()[2] as u32,
            image::ColorType::Rgb8,
        )
        .unwrap();
    buffer
}
