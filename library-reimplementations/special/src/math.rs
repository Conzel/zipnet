pub fn erf(x: f64) -> f64 {
    mathru::special::error::erf(x)
}

pub fn erfc(x: f64) -> f64 {
    mathru::special::error::erfc(x)
}

pub fn tgamma(x: f64) -> f64 {
    mathru::special::gamma::gamma(x)
}

// I don't know if this sign is even needed
pub fn lgamma(x: f64, sign: &mut i32) -> f64 {
    let res = mathru::special::gamma::ln_gamma(x);
    *sign = if res > 1.0 {
        1
    } else if res < 1.0 {
        -1
    } else {
        0
    };
    res
}
