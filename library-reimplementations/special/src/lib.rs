//! [Special functions][1].
//!
//! [1]: https://en.wikipedia.org/wiki/Special_functions

#![allow(warnings)]

#[cfg(test)]
extern crate assert;

mod beta;
mod consts;
mod error;
mod gamma;
mod math;

pub use beta::Beta;
pub use error::Error;
pub use gamma::Gamma;
