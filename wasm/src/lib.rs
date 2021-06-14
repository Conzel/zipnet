extern crate js_sys;
extern crate web_sys;
use image::{load_from_memory_with_format, ImageFormat};

mod utils;

use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

// TODO: Change to a real implementation
#[wasm_bindgen]
pub fn encode_image(buffer: Vec<u8>) -> Vec<u8> {
    console_error_panic_hook::set_once();

    // Example on how to get an image to use, let in as sanity check on the buffer:
    let _result = load_from_memory_with_format(&buffer, ImageFormat::Png).unwrap();
    buffer
}

// TODO: Change to a real implementation
#[wasm_bindgen]
pub fn decode_image(buffer: Vec<u8>) -> Vec<u8> {
    buffer
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
