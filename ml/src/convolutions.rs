use crate::{ImagePrecision, WeightPrecision};
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
    pub fn convolve(&self, image: &Array2<ImagePrecision>) -> Array2<ImagePrecision> {
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

    fn im2col_ref<'a, T>(&self, im_arr:T, ker_height:usize, ker_width:usize, im_height_in:usize, im_width_in:usize, im_height_out:usize, im_width_out:usize, im_channel:usize) -> Array2<ImagePrecision>
    where
        T: AsArray<'a, ImagePrecision, Ix3>,
    {
        let im2d_arr: ArrayView3<f32> = im_arr.into();
        let mut img_matrix = Array::zeros((im_channel*ker_height*ker_width, im_height_out*im_width_out)); // shape: (X, Y)
        let mut cont = 0 as usize;
        for i in 1..im_height_in {
            for j in 1..im_width_out {
                if (((j+ker_width)-1) <= im_width_in) && (((i+ker_height)-1) <= im_height_in) {
                    let patch = im2d_arr.slice(s![
                        i..(i+ker_height)-1,
                        j..(j+ker_width)-1,
                        ..
                    ]);
                    
                    let patch_h = patch.len_of(Axis(0));
                    let patch_w = patch.len_of(Axis(1));
                    let patch_c = patch.len_of(Axis(2));
                    let patchRow = patch.into_shape(patch_h*patch_w*patch_c).unwrap(); // shape: (x, 1)
                    


                    // append it to matrix
                    img_matrix.row_mut(cont).assign(&patchRow);
                    cont+=1;
                }

            }
        }
        img_matrix
    }

    fn conv_2d<'a, T, V>(&self, kernel_weights: T, im2d: V) -> Array2<ImagePrecision>
    where
        // This trait bound ensures that kernel and im2d can be passed as owned array or view.
        // AsArray just ensures that im2d can be converted to an array view via ".into()".
        // Read more here: https://docs.rs/ndarray/0.12.1/ndarray/trait.AsArray.html

        // Weights.shape = [F, C, WW, HH]
        V: AsArray<'a, ImagePrecision, Ix3>,
        T: AsArray<'a, ImagePrecision, Ix2>,
    {
        let im2d_arr: ArrayView3<f32> = im2d.into();
        let kernel_weights_arr: ArrayView2<f32> = kernel_weights.into();

        // W X H X C
        let im_width = im2d_arr.len_of(Axis(0));
        let im_height = im2d_arr.len_of(Axis(1));
        let im_channel = im2d_arr.len_of(Axis(2));

        // HH = self.kernel_height, WW = self.kernel_width
        // calculate output sizes
        // new_h = (H+2*P-HH) / S+1
        let new_im_width = (im_width+2*self.padding - self.kernel_width) / self.stride + 1;
        let new_im_height = (im_height+2*self.padding - self.kernel_height) / self.stride + 1;

        // Alocate memory for output (?)
        let filter = self.num_input_channels as usize;
        // let mut activations = Array::zeros((new_im_height, new_im_width, filter)); // NOTE: N=1

        // filter weights
        let c_out = self.num_output_channels as usize;
        let filter_col = kernel_weights_arr.into_shape((filter, self.kernel_height*self.kernel_width*c_out)).unwrap();// weights.reshape(F, HH*WW*C)

        // prepare bias: TO DO

        // convolve
        let im_col = ConvolutionLayer::im2col_ref(self, im2d_arr, self.kernel_height, self.kernel_width, im_height, im_width, new_im_height, new_im_width, im_channel);
        let mul = &filter_col * &im_col; // + bias_m
        println!("{:?}", mul);
        // let new_im_channel = mul.len_of(Axis(1)) as u16;
        // let activations = col2im_ref(mul, new_im_height, new_im_width, new_im_channel);
        let activations = array![[1., 0.], [-1., 0.]];
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
        let test_img = array![[[0., 1., 0.], [0., 0., 0.], [-1., 0., 0.]], [[0., 1., 0.], [0., 0., 0.], [-1., 0., 0.]], [[0., 1., 0.], [0., 0., 0.], [-1., 0., 0.]]];
        let kernel = Array::from_shape_vec((1, 1, 2, 2), vec![0., 1., -1., 0.]).unwrap();
        let conv_layer = ConvolutionLayer::new(kernel, 1, 0);

        let convolved_image = conv_layer.conv_2d(
            &(conv_layer.kernel.slice(s![0, 0, .., ..])),
            &test_img.view(),
        );

        assert_eq!(convolved_image, array![[1., 0.], [1., 0.]]);
    }

    #[test]
    fn test_2d_conv_with_stride() {
        let test_img: Array3<ImagePrecision> = array![[[0., 1., 0.], [0., 0., 0.], [-1., 0., 0.]], [[0., 1., 0.], [0., 0., 0.], [-1., 0., 0.]], [[0., 1., 0.], [0., 0., 0.], [-1., 0., 0.]]];
        let kernel = Array::from_shape_vec((1, 1, 1, 1), vec![1.]).unwrap();
        let conv_layer = ConvolutionLayer::new(kernel, 2, 0);

        let convolved_image =
            conv_layer.conv_2d(&(conv_layer.kernel.slice(s![0, 0, .., ..])), &test_img);

        assert_eq!(convolved_image, array![[0., 0.], [-1., 0.]]);
    }
}
