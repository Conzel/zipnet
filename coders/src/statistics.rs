use image::Pixel;
use image::RgbImage;
use ndarray::*;
use std::fmt::Display;

type RgbImageRaw = Array3<u8>;

fn mse(orig: &RgbImageRaw, rec: &RgbImageRaw) -> f64 {
    let orig_f = orig.mapv(|a| a as f64);
    let rec_f = rec.mapv(|a| a as f64);
    (orig_f - rec_f).mapv(|a| a.powi(2)).sum() / (orig.len() as f64)
}

fn calc_psnr(orig: &Array3<u8>, rec: &Array3<u8>) -> f64 {
    debug_assert_eq!(orig.len(), rec.len());
    let max_sq = (255f64).powi(2);
    10.0 * (max_sq / mse(orig, rec)).log10()
}

pub struct Statistics {
    pub psnr: f64,
    pub encoding_time: f64,
    pub decoding_time: f64,
}

impl Statistics {
    pub fn new(im_raw: &RgbImageRaw) -> Statistics {
        // TODO: Use real decoded
        let decoded = im_raw;
        Statistics {
            psnr: calc_psnr(im_raw, decoded),
            encoding_time: 0.,
            decoding_time: 0.,
        }
    }
}

impl Display for Statistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PSNR: {}\nTime to encode: {} s\nTime to decode: {} s",
            self.psnr, self.encoding_time, self.decoding_time
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn mse_test() {
        let orig = array![[[0, 1]], [[0, 1]]];
        let rec = array![[[0, 1]], [[0, 0]]];
        assert_eq!(mse(&orig, &rec), 0.25);
        assert_eq!(mse(&rec, &orig), 0.25);

        let orig2 = array![[[0, 1]], [[0, 1]]];
        let rec2 = array![[[0, 1]], [[2, 1]]];
        assert_eq!(mse(&orig2, &rec2), 1.0);
        assert_eq!(mse(&rec2, &orig2), 1.0);
    }

    #[test]
    fn snr_test() {
        let orig = array![[[0, 1]], [[2, 0]]];
        let rec = array![[[0, 1]], [[0, 0]]];
        assert_eq!(calc_psnr(&rec, &orig), 10. * 255f64.powi(2).log10());
        assert_eq!(calc_psnr(&orig, &rec), 10. * 255f64.powi(2).log10());
    }
}
