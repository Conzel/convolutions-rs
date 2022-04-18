//! Module that contains classical convolutions, as used f.e. in convolutional neural networks.
//!
//! More can be read here:
//! - <https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53?gi=f4a37beea40b>

use std::ops::AddAssign;

use crate::{DataRepresentation, Padding};
use ndarray::*;
use num_traits::Float;

/// Rust implementation of a convolutional layer.
/// The weight matrix shall have dimension (in that order)
/// (input channels, output channels, kernel width, kernel height),
/// to comply with the order in which pytorch weights are saved.
pub struct ConvolutionLayer<F: Float> {
    /// Weight matrix of the kernel
    pub(in crate) kernel: Array4<F>,
    pub(in crate) bias: Option<Array1<F>>,
    pub(in crate) stride: usize,
    pub(in crate) padding: Padding,
}

impl<F: 'static + Float + std::ops::AddAssign> ConvolutionLayer<F> {
    /// Creates new convolution layer.
    /// The weights are given in Pytorch layout.
    /// (out channels, in channels, kernel height, kernel width)
    /// Bias: (output height * output width, 1)
    pub fn new(
        weights: Array4<F>,
        bias_array: Option<Array1<F>>,
        stride: usize,
        padding: Padding,
    ) -> ConvolutionLayer<F> {
        assert!(stride > 0, "Stride of 0 passed");
        ConvolutionLayer {
            kernel: weights,
            bias: bias_array,
            stride,
            padding,
        }
    }

    /// Creates new convolution layer. The weights are given in
    /// Tensorflow layout.
    /// (kernel height, kernel width, in channels, out channels)
    pub fn new_tf(
        weights: Array4<F>,
        bias_array: Option<Array1<F>>,
        stride: usize,
        padding: Padding,
    ) -> ConvolutionLayer<F> {
        let permuted_view = weights.view().permuted_axes([3, 2, 0, 1]);
        // Hack to fix the memory layout, permuted axes makes a
        // col major array / non-contiguous array from weights
        let permuted_array: Array4<F> =
            Array::from_shape_vec(permuted_view.dim(), permuted_view.iter().copied().collect())
                .unwrap();
        ConvolutionLayer::new(permuted_array, bias_array, stride, padding)
    }

    /// Analog to conv2d.
    pub fn convolve(&self, image: &DataRepresentation<F>) -> DataRepresentation<F> {
        conv2d(
            &self.kernel,
            self.bias.as_ref(),
            image,
            self.padding,
            self.stride,
        )
    }
}

pub(in crate) fn get_padding_size(
    input_h: usize,
    input_w: usize,
    stride: usize,
    kernel_h: usize,
    kernel_w: usize,
) -> (usize, usize, usize, usize, usize, usize) {
    let pad_along_height: usize;
    let pad_along_width: usize;
    let idx_0: usize = 0;

    if input_h % stride == idx_0 {
        pad_along_height = (kernel_h - stride).max(idx_0);
    } else {
        pad_along_height = (kernel_h - (input_h % stride)).max(idx_0);
    };
    if input_w % stride == idx_0 {
        pad_along_width = (kernel_w - stride).max(idx_0);
    } else {
        pad_along_width = (kernel_w - (input_w % stride)).max(idx_0);
    };

    let pad_top = pad_along_height / 2;
    let pad_bottom = pad_along_height - pad_top;
    let pad_left = pad_along_width / 2;
    let pad_right = pad_along_width - pad_left;

    // yes top/bottom and right/left are swapped. No, I don't know
    // why this change makes it conform to the pytorchn implementation.
    (
        pad_along_height,
        pad_along_width,
        pad_bottom,
        pad_top,
        pad_right,
        pad_left,
    )
}

pub(in crate) fn im2col_ref<'a, T, F: 'a + Float>(
    im_arr: T,
    ker_height: usize,
    ker_width: usize,
    im_height: usize,
    im_width: usize,
    im_channel: usize,
    stride: usize,
) -> Array2<F>
where
    // Args:
    //   im_arr: image matrix to be translated into columns, (C,H,W)
    //   ker_height: filter height (hh)
    //   ker_width: filter width (ww)
    //   im_height: image height
    //   im_width: image width
    //
    // Returns:
    //   col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
    //         new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    T: AsArray<'a, F, Ix3>,
{
    let im2d_arr: ArrayView3<F> = im_arr.into();
    let new_h = (im_height - ker_height) / stride + 1;
    let new_w = (im_width - ker_width) / stride + 1;
    let mut cols_img: Array2<F> =
        Array::zeros((new_h * new_w, im_channel * ker_height * ker_width));
    let mut cont = 0_usize;
    for i in 1..new_h + 1 {
        for j in 1..new_w + 1 {
            let patch = im2d_arr.slice(s![
                ..,
                (i - 1) * stride..((i - 1) * stride + ker_height),
                (j - 1) * stride..((j - 1) * stride + ker_width),
            ]);
            let patchrow_unwrap: Array1<F> = Array::from_iter(patch.map(|a| *a));

            cols_img.row_mut(cont).assign(&patchrow_unwrap);
            cont += 1;
        }
    }
    cols_img
}

/// Performs a convolution on the given image data using this layers parameters.
/// We always convolve on flattened images and expect the input array in im2col
/// style format.
///
/// Read more here:
/// - <https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster>
///
/// Input:
/// -----------------------------------------------
/// - kernel_weights: weights of shape (F, C, HH, WW)
/// - im2d: Input data of shape (C, H, W)
/// -----------------------------------------------
/// - 'stride': The number of pixels between adjacent receptive fields in the
///     horizontal and vertical directions, must be int
/// - 'pad': "Same" or "Valid"

/// Returns:
/// -----------------------------------------------
/// - out: Output data, of shape (F, H', W')
pub fn conv2d<'a, T, V, F: 'static + Float + std::ops::AddAssign>(
    kernel_weights: T,
    bias: Option<&Array1<F>>,
    im2d: V,
    padding: Padding,
    stride: usize,
) -> DataRepresentation<F>
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
    let im_col: Array2<F>; // output of fn: im2col_ref()
    let new_im_height: usize;
    let new_im_width: usize;
    let weight_shape = kernel_weights_arr.shape();
    let num_filters = weight_shape[0] as usize;
    let num_channels_out = weight_shape[1] as usize;
    let kernel_height = weight_shape[2] as usize;
    let kernel_width = weight_shape[3] as usize;

    // Dimensions: C, H, W
    let im_channel = im2d_arr.len_of(Axis(0));
    let im_height = im2d_arr.len_of(Axis(1));
    let im_width = im2d_arr.len_of(Axis(2));

    // Calculate output shapes H', W' for two types of Padding
    if padding == Padding::Same {
        // https://mmuratarat.github.io/2019-01-17/implementing-padding-schemes-of-tensorflow-in-python
        // H' = H / stride
        // W' = W / stride

        let h_float = im_height as f32;
        let w_float = im_width as f32;
        let stride_float = stride as f32;

        let new_im_height_float = (h_float / stride_float).ceil();
        let new_im_width_float = (w_float / stride_float).ceil();

        new_im_height = new_im_height_float as usize;
        new_im_width = new_im_width_float as usize;
    } else {
        // H' =  ((H - HH) / stride ) + 1
        // W' =  ((W - WW) / stride ) + 1
        new_im_height = ((im_height - kernel_height) / stride) + 1;
        new_im_width = ((im_width - kernel_width) / stride) + 1;
    };

    // weights.reshape(F, HH*WW*C)
    let filter_col = kernel_weights_arr
        .into_shape((num_filters, kernel_height * kernel_width * num_channels_out))
        .unwrap();

    // fn:im2col() for different Paddings
    if padding == Padding::Same {
        // https://mmuratarat.github.io/2019-01-17/implementing-padding-schemes-of-tensorflow-in-python
        let (pad_num_h, pad_num_w, pad_top, pad_bottom, pad_left, pad_right) =
            get_padding_size(im_height, im_width, stride, kernel_height, kernel_width);
        let mut im2d_arr_pad: Array3<F> = Array::zeros((
            num_channels_out,
            im_height + pad_num_h,
            im_width + pad_num_w,
        ));
        let pad_bottom_int = (im_height + pad_num_h) - pad_bottom;
        let pad_right_int = (im_width + pad_num_w) - pad_right;
        // https://github.com/rust-ndarray/ndarray/issues/823
        im2d_arr_pad
            .slice_mut(s![.., pad_top..pad_bottom_int, pad_left..pad_right_int])
            .assign(&im2d_arr);

        let im_height_pad = im2d_arr_pad.len_of(Axis(1));
        let im_width_pad = im2d_arr_pad.len_of(Axis(2));

        im_col = im2col_ref(
            im2d_arr_pad.view(),
            kernel_height,
            kernel_width,
            im_height_pad,
            im_width_pad,
            im_channel,
            stride,
        );
    } else {
        im_col = im2col_ref(
            im2d_arr,
            kernel_height,
            kernel_width,
            im_height,
            im_width,
            im_channel,
            stride,
        );
    };
    let filter_transpose = filter_col.t();
    let mul = im_col.dot(&filter_transpose);
    let output = mul
        .into_shape((new_im_height, new_im_width, num_filters))
        .unwrap()
        .permuted_axes([2, 0, 1]);

    add_bias(&output, bias)
}

pub(in crate) fn add_bias<F>(x: &Array3<F>, bias: Option<&Array1<F>>) -> Array3<F>
where
    F: 'static + Float + std::ops::AddAssign,
{
    if let Some(bias_array) = bias {
        assert!(
            bias_array.shape()[0] == x.shape()[0],
            "Bias array has the wrong shape {:?} for vec of shape {:?}",
            bias_array.shape(),
            x.shape()
        );
        // Yes this is really necessary. Broadcasting with ndarray-rust
        // starts at the right side of the shape, so we have to add
        // the axes by hand (else it thinks that it should compare the
        // output width and the bias channels).
        (x + &bias_array
            .clone()
            .insert_axis(Axis(1))
            .insert_axis(Axis(2))
            .broadcast(x.shape())
            .unwrap())
            .into_dimensionality()
            .unwrap()
    } else {
        x.clone()
    }
}
