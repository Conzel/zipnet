//! A probability-theory toolbox.
//!
//! ## Example
//!
//! ```
//! use probability::prelude::*;
//!
//! let mut source = source::default();
//! let distribution = Uniform::new(0.0, 1.0);
//! let sampler = Independent(&distribution, &mut source);
//! let samples = sampler.take(10).collect::<Vec<_>>();
//! ```
#![allow(warnings)]

#[cfg(test)]
extern crate assert;

extern crate random;
extern crate special;

macro_rules! nonnan(
    ($argument:ident) => (if $argument.is_nan() { return ::std::f64::NAN; });
);

macro_rules! should(
    ($requirement:expr) => (debug_assert!($requirement));
    ($requirement:expr, $code:expr) => (debug_assert!($code, stringify!($requirement)));
);

pub mod distribution;
pub mod prelude;
pub mod sampler;
pub mod source;
