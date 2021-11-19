# convolutions-rs
convolutions-rs is a crate that provides a fast, well-tested convolutions library for machine learning written entirely in Rust with minimal dependencies. In particular, this is the first convolutions crate that has absolutely no dependencies on native C libraries. We provide both transposed convolutions (also called deconvolutions), as well as normal convolutions.

This crate has been developed in the course of the ZipNet project (https://github.com/Conzel/zipnet), where we required a C-free implementation of convolutions in order to compile our code to WebASM.

## Features 
- [x] Minimal dependencies, especially no C-dependencies
- [x] Extensively tested through randomly generated unit tests
- [x] 100% compatible with Tensorflow and Pytorch implementations
- [x] Generics to ensure smooth usage
- [x] Speed verified by benchmarking

As of now, this crate is as fast as Pytorch on small images, but has a noticeable slowdown on large and medium images (takes ~20-50x as much time). We are still reaonsably fast enough for research/sample applications, but we aim to improve the speed to get close to PyTorch. Benchmarks can be found at https://github.com/Conzel/convolutions-rs-benchmarks/.

## Usage
As mentioned, this package provides normal convolutions as well as transposed convolutions. We provide both in the form of free functions as well as something resembling a neural network layer. This crate also requires ndarray to use the functions, as input and output are in the form of ndarrays.

Example:
```
use convolutions_rs::convolutions::*;
use ndarray::*;
use convolutions_rs::Padding;

// Input has shape (channels, height, width)
let input = Array::from_shape_vec(
    (1, 4, 4),
    vec![1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.]
)
.unwrap();

// Kernel has shape (channels out, channels in, height, width)
let kernel: Array4<f32> = Array::from_shape_vec(
    (2,1,2,2),
    vec![1.,1.,1.,1.,1.,1.,1.,1.]
)
.unwrap();

let conv_layer = ConvolutionLayer::new(kernel.clone(), 1, Padding::Valid);
let output_layer: Array3<f32> = conv_layer.convolve(&input);
let output_free = conv2d(&kernel, &input, Padding::Valid, 1);

println!("Layer: {:?}", output_layer);
println!("Free: {:?}", output_free);
```
