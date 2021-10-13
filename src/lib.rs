use ndarray::Array3;

pub mod convolutions;
pub mod transposed_convolutions;

pub type WeightPrecision = f32;
pub type ImagePrecision = f32;
pub type InternalDataRepresentation = Array3<f32>;
pub type ConvKernel = ndarray::Array4<WeightPrecision>;

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Padding {
    Same,
    Valid,
}
