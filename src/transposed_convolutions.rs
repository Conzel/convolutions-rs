//! Module that contains transposed convolutions (also called deconvolution layers).
//!
//! More can be read here:
//! - <https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers>
//! - <https://github.com/akutzer/numpy_cnn/blob/master/CNN/Layer/TransposedConv.py>
//! - <https://ieee.nitk.ac.in/blog/deconv/>
use crate::{
    convolutions::{add_bias, get_padding_size, im2col_ref, ConvolutionLayer},
    ConvKernel, DataRepresentation, Padding,
};
use ndarray::*;
use num_traits::Float;

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

/// Performs a transposed convolution on the input image. This upsamples the image.
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
pub fn conv_transpose2d<'a, T, V, F: 'static + Float + std::ops::AddAssign>(
    kernel_weights: T,
    bias: Option<&Array1<F>>,
    im2d: V,
    padding: Padding,
    stride: usize,
) -> Array3<F>
where
    // This trait bound ensures that kernel and im2d can be passed as owned array or view.
    // AsArray just ensures that im2d can be converted to an array view via ".into()".
    // Read more here: https://docs.rs/ndarray/0.12.1/ndarray/trait.AsArray.html
    V: AsArray<'a, F, Ix3>,
    T: AsArray<'a, F, Ix4>,
{
    // Initialisations
    let im2d_arr: ArrayView3<F> = im2d.into();
    let kernel_weights_arr: ArrayView4<F> = kernel_weights.into();
    let im2d_stride: Array3<F>;
    let new_im_height: usize;
    let new_im_width: usize;
    let im_channel_stride: usize;
    let im_height_stride: usize;
    let im_width_stride: usize;

    let weight_shape = kernel_weights_arr.shape();

    let num_channels_out = weight_shape[0] as usize;
    let num_filters = weight_shape[1] as usize;
    let kernel_height = weight_shape[2] as usize;
    let kernel_width = weight_shape[3] as usize;

    // Dimensions: C, H, W
    let im_channel = im2d_arr.len_of(Axis(0));
    let im_height = im2d_arr.len_of(Axis(1));
    let im_width = im2d_arr.len_of(Axis(2));

    // Calculate output shapes H', W'
    // https://towardsdatascience.com/understand-transposed-convolutions-and-build-your-own-transposed-convolution-layer-from-scratch-4f5d97b2967
    // https://theano-pymc.readthedocs.io/en/latest/tutorial/conv_arithmetic.html
    // H' =  (H - 1) * stride  + HH
    // W' =  (W - 1) * stride  + WW
    new_im_height = (im_height - 1) * stride + kernel_height;
    new_im_width = (im_width - 1) * stride + kernel_width;

    // weights.reshape(F, HH*WW*C)
    // loop over F and C ; flip the kernel and append it back
    let mut filter_col: Array3<F> =
        Array::zeros((num_channels_out, num_filters, kernel_height * kernel_width));
    for i in 0..num_channels_out {
        for j in 0..num_filters {
            let patch_kernel_weights = kernel_weights_arr.slice(s![i, j, .., ..]);
            let mut weights_flatten = patch_kernel_weights
                .into_shape(kernel_height * kernel_width)
                .unwrap()
                .to_vec();
            //FLIP
            weights_flatten.reverse();
            let weights_reverse =
                Array::from_shape_vec(kernel_height * kernel_width, weights_flatten).unwrap();
            filter_col
                .slice_mut(s![i, j, 0..kernel_height * kernel_width])
                .assign(&weights_reverse);
        }
    }
    filter_col.swap_axes(0, 1);
    let target_shape = (num_filters, num_channels_out * kernel_height * kernel_width);
    let filter_col_flatten =
        Array::from_shape_vec(target_shape, filter_col.iter().map(|a| *a).collect()).unwrap();

    // STRIDE > 1
    if stride != 1 {
        // https://github.com/akutzer/numpy_cnn/blob/master/CNN/Layer/TransposedConv.py
        // ASSUMPTION: stride[0] == stride[1]
        let stride_h = stride * im_height;
        let stride_w = stride * im_width;
        let mut im2d_stride_full: Array3<F> = Array::zeros((im_channel, stride_h, stride_w));
        im2d_stride_full
            .slice_mut(s![.., ..;stride, ..;stride])
            .assign(&im2d_arr);
        let stride_h_crop = (stride_h - stride) + 1;
        let stride_w_crop = (stride_w - stride) + 1;
        im2d_stride = im2d_stride_full
            .slice_mut(s![.., ..stride_h_crop, ..stride_w_crop])
            .into_owned();
    } else {
        im2d_stride = im2d_arr.into_owned();
    };

    im_channel_stride = im2d_stride.len_of(Axis(0));
    im_height_stride = im2d_stride.len_of(Axis(1));
    im_width_stride = im2d_stride.len_of(Axis(2));

    // PADDING
    // fn:im2col() for with padding always
    let pad_h = kernel_height - 1;
    let pad_w = kernel_width - 1;
    let mut im2d_arr_pad: Array3<F> = Array::zeros((
        im_channel_stride,
        im_height_stride + pad_h + pad_h,
        im_width_stride + pad_w + pad_w,
    ));
    let pad_int_h = im_height_stride + pad_h;
    let pad_int_w = im_width_stride + pad_w;
    // https://github.com/rust-ndarray/ndarray/issues/823
    im2d_arr_pad
        .slice_mut(s![.., pad_h..pad_int_h, pad_w..pad_int_w])
        .assign(&im2d_stride);

    let im_height_pad = im2d_arr_pad.len_of(Axis(1));
    let im_width_pad = im2d_arr_pad.len_of(Axis(2));

    let im_col = im2col_ref(
        im2d_arr_pad.view(),
        kernel_height,
        kernel_width,
        im_height_pad,
        im_width_pad,
        im_channel,
        1,
    );
    // NOTE: The kernel strides across the image at 1 regardless of the stride we provide

    let filter_transpose = filter_col_flatten.t();
    let mul = im_col.dot(&filter_transpose); // + bias_m

    let mut mul_reshape = mul
        .into_shape((new_im_height, new_im_width, num_filters))
        .unwrap()
        .into_owned();
    mul_reshape.swap_axes(0, 2);
    mul_reshape.swap_axes(1, 2);

    let output = if padding == Padding::Same {
        let (_, _, pad_top, pad_bottom, pad_left, pad_right) = get_padding_size(
            im_height * stride,
            im_width * stride,
            stride,
            kernel_height,
            kernel_width,
        );

        let pad_right_int = new_im_width - pad_right;
        let pad_bottom_int = new_im_height - pad_bottom;
        mul_reshape
            .slice(s![.., pad_top..pad_bottom_int, pad_left..pad_right_int])
            .into_owned()
    } else {
        mul_reshape.into_owned()
    };
    add_bias(&output, bias)
}
