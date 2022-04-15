//! Module that contains transposed convolutions (also called deconvolution layers).
//!
//! More can be read here:
//! - <https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers>
//! - <https://github.com/akutzer/numpy_cnn/blob/master/CNN/Layer/TransposedConv.py>
//! - <https://ieee.nitk.ac.in/blog/deconv/>
use crate::{
    convolutions::{add_bias, ConvolutionLayer},
    ConvKernel, DataRepresentation, Padding,
};
use ndarray::*;
use num_traits::Float;
use std::ops::AddAssign;

/// Analog to a Convolution Layer
pub struct TransposedConvolutionLayer<F: Float> {
    convolution_layer: ConvolutionLayer<F>,
}

impl<F: 'static + Float + std::ops::AddAssign> TransposedConvolutionLayer<F> {
    /// Creates new transposed_convolutionLayer. The weights are given in
    /// Pytorch layout.
    /// (in channels, out channels, kernel_height, kernel_width)
    pub fn new(
        weights: ConvKernel<F>,
        bias: Option<Array1<F>>,
        stride: usize,
        padding: Padding,
    ) -> TransposedConvolutionLayer<F> {
        TransposedConvolutionLayer {
            convolution_layer: ConvolutionLayer::new(weights, bias, stride, padding),
        }
    }

    /// Creates new transposed_convolutionLayer. The weights are given in
    /// Tensorflow layout.
    /// (kernel height, kernel width, out channels, in channels)
    pub fn new_tf(
        weights: ConvKernel<F>,
        bias: Option<Array1<F>>,
        stride: usize,
        padding: Padding,
    ) -> TransposedConvolutionLayer<F> {
        TransposedConvolutionLayer {
            convolution_layer: ConvolutionLayer::new_tf(weights, bias, stride, padding),
        }
    }

    /// Analog to conv_transpose2d.
    pub fn transposed_convolve(&self, image: &DataRepresentation<F>) -> DataRepresentation<F> {
        let output = conv_transpose2d(
            &self.convolution_layer.kernel,
            self.convolution_layer.bias.as_ref(),
            &image.view(),
            self.convolution_layer.padding,
            self.convolution_layer.stride,
        );
        output
    }
}

/// Implementation of col2im as seen in the Pytorch cpp implementation.
/// This implementation doesn't just assign the matrix patch-wise to a new
/// matrix, but also sums the patches such that it unrolls the result
/// of conv-transpose.
/// mat is a matrix of size (image_width * image_height, c_out * k_h * k_w),
/// in which each entry is the value of a kernel pixel multiplied with an image pixels
/// with the channels summed out
/// The output naming is a bit confusing, as this function is originally used in the backward
/// pass. What this function returns is simply (channels, height, width).
fn col2im_pt<'a, F: 'a + Float + AddAssign>(
    data_col: &[F],
    channels: usize,
    height: usize,
    width: usize,
    output_height: usize,
    output_width: usize,
    kernel_h: usize,
    kernel_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride: usize,
) -> DataRepresentation<F> {
    let mut res: Array1<F> = Array::zeros(channels * height * width);

    let height_col = output_height;
    let width_col = output_width;
    let channels_col = channels * kernel_h * kernel_w;

    for c_col in 0..channels_col {
        let w_offset = c_col % kernel_w;
        let h_offset = (c_col / kernel_w) % kernel_h;
        let c_im = c_col / kernel_h / kernel_w;

        for h_col in 0..height_col {
            let h_im = (h_col * stride) as i64 - pad_h as i64 + h_offset as i64;

            for w_col in 0..width_col {
                let w_im = (w_col * stride) as i64 - pad_w as i64 + w_offset as i64;

                if h_im < height as i64 && h_im >= 0 && w_im < width as i64 && w_im >= 0 {
                    // TODO: Use unsafe get
                    let el = res
                        .get_mut((c_im * height + h_im as usize) * width + w_im as usize)
                        .unwrap();

                    // TODO: Use unsafe get
                    el.add_assign(data_col[(c_col * height_col + h_col) * width_col + w_col]);
                }
            }
        }
    }
    Array::from_shape_vec((channels, height, width), res.into_iter().collect()).unwrap()
}

/// Performs a transposed convolution on the input image. This upsamples the image.
/// This implementation is identical to the supplied by Pytorch and works by performing
/// a "backwards pass" on the input.
///
/// More explanation can be read here:
/// - <https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers>
///
/// NOTE: THERE IS A CHANGE IN KERNEL DIMENSIONS FOR CONV TRANSPOSED
/// Input:
/// -----------------------------------------------
/// - im2d: Input data of shape (C, H, W)
/// - kernel_weights: Filter weights of shape (C, F, HH, WW) // DIFFERENT from CONV2D
/// -----------------------------------------------
/// - 'stride': The number of pixels between adjacent receptive fields in the
///     horizontal and vertical directions, must be int
/// - 'pad': "Same" or "Valid"

/// Returns:
/// -----------------------------------------------
/// - out: Output data, of shape (F, H', W')
pub fn conv_transpose2d<'a, T, V, F: 'static + Float + AddAssign>(
    kernel_weights: T,
    bias: Option<&Array1<F>>,
    im2d: V,
    padding: Padding,
    stride: usize,
) -> DataRepresentation<F>
where
    V: AsArray<'a, F, Ix3>,
    T: AsArray<'a, F, Ix4>,
{
    let kernel_mat: ArrayView4<F> = kernel_weights.into();
    let im_mat: ArrayView3<F> = im2d.into();
    // kernel has shape (C, F, HH, WW)
    let k_c = kernel_mat.shape()[0];
    let k_f = kernel_mat.shape()[1];
    let k_h = kernel_mat.shape()[2];
    let k_w = kernel_mat.shape()[3];
    let ker_reshape = kernel_mat.into_shape((k_c, k_f * k_h * k_w)).unwrap();
    let im_c = im_mat.shape()[0];
    let im_h = im_mat.shape()[1];
    let im_w = im_mat.shape()[2];
    let im_reshape = im_mat.into_shape((im_c, im_h * im_w)).unwrap();
    let data_matrix: Array2<F> = im_reshape.t().dot(&ker_reshape);
    // output padding might be necessary to achieve the desired shape
    let (pad_h, pad_w, output_pad_h, output_pad_w) = match padding {
        Padding::Valid => (0, 0, 0, 0),
        Padding::Same => (k_h / 2, k_w / 2, stride - 1, stride - 1),
    };
    println!("{:?}, {:?}", pad_h, pad_w);
    let output_height = (im_h - 1) * stride + k_h + output_pad_h - 2 * pad_h;
    let output_width = (im_w - 1) * stride + k_w + output_pad_w - 2 * pad_w;
    // let output_height = (im_h - 1) * stride + k_h;
    // let output_width = (im_w - 1) * stride + k_w;
    let data_matrix_contiguous = Array::from_iter(data_matrix.view().t().iter().map(|a| *a));
    let t_conv_output = col2im_pt(
        data_matrix_contiguous.as_slice().unwrap(),
        k_f,
        output_height,
        output_width,
        im_h,
        im_w,
        k_h,
        k_w,
        pad_h,
        pad_w,
        stride,
    );
    return add_bias(&t_conv_output, bias);
}
