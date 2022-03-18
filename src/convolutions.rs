//! Module that contains classical convolutions, as used f.e. in convolutional neural networks.
//!
//! More can be read here:
//! - <https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53?gi=f4a37beea40b>
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
    pub(in crate) bias: Option<Array2<F>>,
    pub(in crate) stride: usize,
    pub(in crate) padding: Padding,
}

impl<F: 'static + Float> ConvolutionLayer<F> {
    /// Creates new convolution layer. 
    /// The weights are given in Pytorch layout.
    /// (out channels, in channels, kernel height, kernel width)
    /// Bias: (output height * output width, 1)
    pub fn new(weights: Array4<F>, bias_array: Option<Array2<F>>, stride: usize, padding: Padding) -> ConvolutionLayer<F> {
        assert!(stride > 0, "Stride of 0 passed");
        // match bias_array{
        //     Some(x) => x,
        //     None => None,
        // }
        ConvolutionLayer {
            kernel: weights,
            bias:bias_array,
            stride,
            padding,
        }
    }

    /// Creates new convolution layer. The weights are given in
    /// Tensorflow layout.
    /// (kernel height, kernel width, in channels, out channels)
    pub fn new_tf(weights: Array4<F>, bias: Option<Array2<F>>, stride: usize, padding: Padding) -> ConvolutionLayer<F> {
        let permuted_view = weights.view().permuted_axes([3, 2, 0, 1]);
        // Hack to fix the memory layout, permuted axes makes a
        // col major array / non-contiguous array from weights
        let permuted_array: Array4<F> =
            Array::from_shape_vec(permuted_view.dim(), permuted_view.iter().copied().collect())
                .unwrap();
        ConvolutionLayer::new(permuted_array, bias, stride, padding)
    }

    /// Analog to conv2d.
    pub fn convolve(&self, image: &DataRepresentation<F>) -> DataRepresentation<F> {
        conv2d(&self.kernel, self.bias.clone(), image, self.padding, self.stride)
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

    (
        pad_along_height,
        pad_along_width,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
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

fn col2im_ref<'a, T, F: 'a + Float>(
    mat: T,
    height_prime: usize,
    width_prime: usize,
    _channels: usize,
) -> DataRepresentation<F>
where
    T: AsArray<'a, F, Ix2>,
{
    let img_vec: ArrayView2<F> = mat.into();
    let filter_axis = img_vec.len_of(Axis(1));
    let mut img_mat: Array3<F> = Array::zeros((filter_axis, height_prime, width_prime));
    // C = 1
    for i in 0..filter_axis {
        let col = img_vec.slice(s![.., i]).to_vec();
        let col_reshape = Array::from_shape_vec((height_prime, width_prime), col).unwrap();
        img_mat
            .slice_mut(s![i, 0..height_prime, 0..width_prime])
            .assign(&col_reshape);
    }
    img_mat
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
pub fn conv2d<'a, T, V, F: 'static + Float>(
    kernel_weights: T,
    bias: Option<Array2<F>>,
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
    let bias_array:Array2<F>;
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

    // initialize if bias
    if bias.is_none(){
        bias_array = Array::zeros((new_im_height * new_im_height, 1));
    } 
    else {
        // let bias_vec = bias.into_iter().flatten().collect();
        bias_array = bias.unwrap();
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
    let mul = im_col.dot(&filter_transpose) + bias_array;
    col2im_ref(&mul, new_im_height, new_im_width, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let bias = Array::zeros((9, 1));
        let conv_layer = ConvolutionLayer::new(testker, Some(bias), 1, Padding::Valid);
        let output = arr3(&[[
            [57.0, 75.0, 93.0],
            [111.0, 129.0, 141.0],
            [138.0, 156.0, 162.0],
        ]]);
        let convolved_image = conv_layer.convolve(&test_img);

        assert_eq!(convolved_image, output);

        let test_img1 = array![
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
        let kernel1 = Array::from_shape_vec(
            (1, 3, 2, 2),
            vec![1., 2., 1., 2., 1., 2., 1., 2., 1., 2., 1., 2.],
        );
        let testker1 = kernel1.unwrap();
        let bias1 = Array::zeros((16, 1));
        let conv_layer1 = ConvolutionLayer::new(testker1, Some(bias1), 1, Padding::Same);
        let output1 = arr3(&[[
            [57.0, 75.0, 93.0, 33.0],
            [111.0, 129.0, 141.0, 48.0],
            [138.0, 156.0, 162.0, 54.0],
            [69.0, 78.0, 81.0, 27.0],
        ]]);
        let convolved_image1 = conv_layer1.convolve(&test_img1);

        assert_eq!(convolved_image1, output1);
    }

    #[test]
    fn test_conv2d_tf_layout() {
        let weights_pt = Array::from_shape_vec(
            (2, 1, 3, 3),
            vec![
                0.06664403, 0.65961174, 0.49895822, 0.80375346, 0.20159994, 0.25319365, 0.0520944,
                0.33067411, 0.76843672, 0.08252145, 0.22638044, 0.09291164, 0.63277792, 0.50181511,
                0.40393298, 0.19495441, 0.30511827, 0.28940649,
            ],
        )
        .unwrap();

        let weights_tf = Array::from_shape_vec(
            (3, 3, 1, 2),
            vec![
                0.06664403, 0.08252145, 0.65961174, 0.22638044, 0.49895822, 0.09291164, 0.80375346,
                0.63277792, 0.20159994, 0.50181511, 0.25319365, 0.40393298, 0.0520944, 0.19495441,
                0.33067411, 0.30511827, 0.76843672, 0.28940649,
            ],
        )
        .unwrap();

        let im = array![[
            [0.56494069, 0.3395626, 0.71270928],
            [0.04827336, 0.12623257, 0.30822787],
            [0.82976574, 0.8590054, 0.90254945]
        ]];
        let conv_pt = ConvolutionLayer::new(weights_pt, None, 1, Padding::Valid);
        let conv_tf = ConvolutionLayer::new_tf(weights_tf, None, 1, Padding::Valid);
        assert_eq!(conv_pt.convolve(&im), conv_tf.convolve(&im));
    }
}
