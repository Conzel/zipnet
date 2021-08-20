use crate::{models::InternalDataRepresentation, ImagePrecision, WeightPrecision};
use ndarray::*;

/// Rust implementation of a convolutional layer.
/// The weight matrix shall have dimension (in that order)
/// input channels x output channels x kernel width x kernel height
/// (to comply with the order in which pytorch weights are saved).
pub struct ConvolutionLayer {
    /// Weight matrix of the kernel
    kernel: Array4<WeightPrecision>,
    kernel_width: usize,
    kernel_height: usize,
    stride: usize,
    padding: usize,
    num_input_channels: u16,
    num_output_channels: u16,
}

impl ConvolutionLayer {
    pub fn new(
        weights: Array4<WeightPrecision>,
        stride: usize,
        padding: usize,
    ) -> ConvolutionLayer {
        let num_input_channels = weights.len_of(Axis(0)) as u16; // Filters
        let num_output_channels = weights.len_of(Axis(1)) as u16; // Channels
        let kernel_width = weights.len_of(Axis(2)); // Width
        let kernel_height = weights.len_of(Axis(3)); // Height

        debug_assert!(stride > 0, "Stride of 0 passed");

        ConvolutionLayer {
            kernel: weights,
            kernel_width,
            kernel_height,
            stride,
            num_input_channels,
            num_output_channels,
            padding,
        }
    }

    /// Performs a convolution on the given image data using this layers parameters.
    /// We always convolve on flattened images and expect the input array in im2col
    /// style format (read more here).
    /// https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster
    pub fn convolve(&self, image: &InternalDataRepresentation) -> InternalDataRepresentation {
        todo!();
    }

    /// Naive implementation of 2d convolution for reference implementations
    fn conv_2d_naive<'a, T, V>(&self, kernel_weights: T, im2d: V) -> Array2<ImagePrecision>
    where
        // This trait bound ensures that kernel and im2d can be passed as owned array or view.
        // AsArray just ensures that im2d can be converted to an array view via ".into()".
        // Read more here: https://docs.rs/ndarray/0.12.1/ndarray/trait.AsArray.html
        V: AsArray<'a, ImagePrecision, Ix2>,
        T: AsArray<'a, ImagePrecision, Ix2>,
    {
        if self.padding > 0 {
            unimplemented!("Padding bigger than 0 is not supported yet.")
        }
        let im2d_arr: ArrayView2<f32> = im2d.into();
        let kernel_weights_arr: ArrayView2<f32> = kernel_weights.into();

        let im_width = im2d_arr.len_of(Axis(0));
        let im_height = im2d_arr.len_of(Axis(1));

        let new_im_width = (im_width - self.kernel_width) / self.stride + 1;
        let new_im_height = (im_height - self.kernel_height) / self.stride + 1;

        let mut ret = Array::zeros((new_im_width, new_im_height));

        for i in 0..new_im_width {
            let i_with_stride = i * self.stride;
            for j in 0..new_im_height {
                let j_with_stride = j * self.stride;
                let imslice = im2d_arr.slice(s![
                    i_with_stride..(i_with_stride + self.kernel_width),
                    j_with_stride..(j_with_stride + self.kernel_height)
                ]);

                let conv_entry = (&imslice * &kernel_weights_arr).sum();
                ret[[i, j]] = conv_entry;
            }
        }
        ret
    }

    fn im2col_ref<'a, T>(
        &self,
        im_arr: T,
        ker_height: usize,
        ker_width: usize,
        im_height_in: usize,
        im_width_in: usize,
        im_channel: usize,
    ) -> Array2<ImagePrecision>
    where
        T: AsArray<'a, ImagePrecision, Ix3>,
    {
        let im2d_arr: ArrayView3<f32> = im_arr.into();
        let new_h = (im_height_in - ker_height) / self.stride + 1;
        let new_w = (im_width_in - ker_width) / self.stride + 1;
        let mut img_matrix: Array2<ImagePrecision> =
            Array::zeros((new_h * new_w, im_channel * ker_height * ker_width)); // shape: (X, Y)
        let mut cont = 0 as usize;
        for i in 1..new_h + 1 {
            for j in 1..new_w + 1 {
                let patch = im2d_arr.slice(s![
                    ..,
                    i - 1 * self.stride..(i - 1 * self.stride + ker_height),
                    j - 1 * self.stride..(j - 1 * self.stride + ker_width),
                ]);
                let patchrow_unwrap: Array1<f32> = Array::from_iter(patch.map(|a| *a));

                // append it to matrix
                img_matrix.row_mut(cont).assign(&patchrow_unwrap);
                cont += 1;
            }
        }
        img_matrix
    }

    fn col2im_ref<'a, T>(
        &self,
        mat: T,
        height_prime: usize,
        width_prime: usize,
        C: usize,
    ) -> Array3<ImagePrecision>
    where
        T: AsArray<'a, ImagePrecision, Ix2>,
    {
        let img_vec: ArrayView2<f32> = mat.into();
        let filter_axis = img_vec.len_of(Axis(1));
        // let mut img_mat: Array3<ImagePrecision> =
        // Array::zeros((filter_axis, height_prime, width_prime)); ALTERNATE
        let mut img_mat: Array3<ImagePrecision> = Array::zeros((0, height_prime, width_prime));
        if C == 1 {
            for i in 0..filter_axis {
                let col = img_vec.slice(s![.., i]);
                let col_reshape = col.into_shape((height_prime, width_prime)).unwrap();
                // img_mat.assign(&col_reshape);  ALTERNATE
                img_mat.push(Axis(0), col_reshape).unwrap();
            }
        }
        img_mat
    }

    fn conv_2d<'a, T, V>(&self, kernel_weights: T, im2d: V) -> Array3<ImagePrecision>
    where
        // This trait bound ensures that kernel and im2d can be passed as owned array or view.
        // AsArray just ensures that im2d can be converted to an array view via ".into()".
        // Read more here: https://docs.rs/ndarray/0.12.1/ndarray/trait.AsArray.html

        // Weights.shape = [F, C, WW, HH]
        V: AsArray<'a, ImagePrecision, Ix3>,
        T: AsArray<'a, ImagePrecision, Ix4>,
    {
        let im2d_arr: ArrayView3<f32> = im2d.into();
        let kernel_weights_arr: ArrayView4<f32> = kernel_weights.into();

        // C X H X W
        let im_width = im2d_arr.len_of(Axis(2));
        let im_height = im2d_arr.len_of(Axis(1));
        let im_channel = im2d_arr.len_of(Axis(0));

        // HH = self.kernel_height, WW = self.kernel_width
        // calculate output sizes
        // new_h = (H+2*P-HH) / S+1
        let new_im_width = (im_width + 2 * self.padding - self.kernel_width) / self.stride + 1;
        let new_im_height = (im_height + 2 * self.padding - self.kernel_height) / self.stride + 1;

        // Alocate memory for output (?)
        let filter = self.num_input_channels as usize;

        // filter weights
        let c_out = self.num_output_channels as usize;
        let filter_col = kernel_weights_arr
            .into_shape((filter, self.kernel_height * self.kernel_width * c_out))
            .unwrap(); // weights.reshape(F, HH*WW*C)

        // prepare bias: TO DO

        // convolve
        let im_col = ConvolutionLayer::im2col_ref(
            self,
            im2d_arr,
            self.kernel_height,
            self.kernel_width,
            im_height,
            im_width,
            im_channel,
        );
        let filter_transpose = filter_col.t(); // SHAPE IS (1, N)
        let mul = im_col.dot(&filter_transpose); // + bias_m
        let activations = ConvolutionLayer::col2im_ref(self, &mul, new_im_height, new_im_width, 1); // filter is set to 1 (?)
        activations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_naive_2d_conv() {
        let test_img = array![[0., 1., 0.], [0., 0., 0.], [-1., 0., 0.]];
        let kernel = Array::from_shape_vec((1, 1, 2, 2), vec![0., 1., -1., 0.]).unwrap();
        let conv_layer = ConvolutionLayer::new(kernel, 1, 0);

        let convolved_image = conv_layer.conv_2d_naive(
            &(conv_layer.kernel.slice(s![0, 0, .., ..])),
            &test_img.view(),
        );

        assert_eq!(convolved_image, array![[1., 0.], [1., 0.]]);
    }

    #[test]
    fn test_naive_2d_conv_with_stride() {
        let test_img: Array2<ImagePrecision> = array![[0., 1., 0.], [0., 0., 0.], [-1., 0., 0.]];
        let kernel = Array::from_shape_vec((1, 1, 1, 1), vec![1.]).unwrap();
        let conv_layer = ConvolutionLayer::new(kernel, 2, 0);

        let convolved_image =
            conv_layer.conv_2d_naive(&(conv_layer.kernel.slice(s![0, 0, .., ..])), &test_img);

        assert_eq!(convolved_image, array![[0., 0.], [-1., 0.]]);
    }

    #[test]
    fn test_2d_conv() {
        let test_img = array![
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 5.0, 6.0, 7.0],
                [7.0, 8.0, 9.0, 9.0],
                [7.0, 8.0, 9.0, 9.0]
            ],
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 5.0, 6.0, 7.0],
                [7.0, 8.0, 9.0, 9.0],
                [7.0, 8.0, 9.0, 9.0]
            ],
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 5.0, 6.0, 7.0],
                [7.0, 8.0, 9.0, 9.0],
                [7.0, 8.0, 9.0, 9.0]
            ]
        ];
        let kernel = Array::from_shape_vec(
            (1, 3, 2, 2),
            vec![1., 2., 1., 2., 1., 2., 1., 2., 1., 2., 1., 2.],
        );
        let testker = kernel.unwrap();
        let conv_layer = ConvolutionLayer::new(testker, 1, 0);
        let output = arr3(&[[
            [57.0, 75.0, 93.0],
            [111.0, 129.0, 141.0],
            [138.0, 156.0, 162.0],
        ]]);
        let convolved_image = conv_layer.conv_2d(&(conv_layer.kernel), &test_img.view());

        assert_eq!(convolved_image, output);
    }
}
