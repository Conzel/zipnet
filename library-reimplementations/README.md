# Library reimplementations
In this folder we have reimplementations for some libraries. It turns out that `constriction` depends on `probability`, which in turn depends on `special`, which depends on libc. WASM doesn't support libc, so we have used the `mathru` crate (which implements the same functions basically, but in pure Rust) to replace the dependencies in probability on libc (4 functions total, error functions and gamma functions). The resulting crates are free of libc dependencies. Some tests fail because of it, but it is only 1 test in the constriction library, which concerns a function we don't use anyways. This is probably because of different underlying implementation details.

## Why can't we use libc?
Libc is not available on wasm-unknown-unknown (and only in a limited form on wasm-wasi, which doesn't contain the necessary functions for the special crate). To remedy this, we would have to compile to wasm-unknown-emscripten, which comes with it's own baggage (bigger wasm file, no support for wasm-bindgen, and it's kind of ugly to get to play nicely with Rust).

