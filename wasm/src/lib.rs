extern crate js_sys;
extern crate web_sys;
use image::{load_from_memory_with_format, GenericImageView, ImageFormat};

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

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, wasm-template-rust!");
}

#[wasm_bindgen]
pub fn process_png_image(buffer: Vec<u8>) -> u32 {
    // Enables us to get better error messages in the browser
    console_error_panic_hook::set_once();

    let result = load_from_memory_with_format(&buffer, ImageFormat::Png).unwrap();
    result.width()
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
