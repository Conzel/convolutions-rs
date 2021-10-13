//! This package provides normal convolutions as well as transposed convolutions.
//! We provide both in the form of free functions as well as something resembling a neural network layer.
//! This crate also requires ndarray to use the functions, as input and output are in the form of ndarrays.
//!
//! In all implementations, we conform to the Pytorch implementation of convolutions (which agrees with the Tensorflow implementation, up to shapes).
//! We have used the technique described in this blog to achieve a fast implementation:
//! - <https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster>
//!
//! Example:
//! ```
//! use convolutions_rs::convolutions::*;
//! use ndarray::*;
//! use convolutions_rs::Padding;
//!
//! // Input has shape (channels, height, width)
//! let input = Array::from_shape_vec(
//!     (1, 4, 4),
//!     vec![1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.]
//! )
//! .unwrap();
//!
//! // Kernel has shape (channels out, channels in, height, width)
//! let kernel: Array4<f32> = Array::from_shape_vec(
//!     (2,1,2,2),
//!     vec![1.,1.,1.,1.,1.,1.,1.,1.]
//! )
//! .unwrap();
//!
//! let conv_layer = ConvolutionLayer::new(kernel.clone(), 1, Padding::Valid);
//! let output_layer: Array3<f32> = conv_layer.convolve(&input);
//! let output_free = conv2d(kernel, &input, Padding::Valid, 1);
//!
//! println!("Layer: {:?}", output_layer);
//! println!("Free: {:?}", output_free);
//! ```

use ndarray::Array3;

pub mod convolutions;
pub mod transposed_convolutions;

pub type WeightPrecision = f32;
pub type ImagePrecision = f32;
pub type InternalDataRepresentation = Array3<f32>;
pub type ConvKernel = ndarray::Array4<WeightPrecision>;

/// Padding (specific way of adding zeros to the input matrix) kind used in the convolution.
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Padding {
    /// Output has the same shape as input.
    Same,
    /// Padding is only used to make input fit the kernel.
    Valid,
}
