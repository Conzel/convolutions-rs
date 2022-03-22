#!/usr/bin/env python
import jinja2
import numpy as np
import itertools
import tensorflow as tf
import os
import torch


class RandomArrayTest:
    def __init__(self, test_name, layer_name, random_test_objects):
        """Struct that represents one Random Array Test.
        test_name: str, name of the test case
        layer_name: str, name of the layer to test
        random_test_objects: [TestObject]
        """
        assert layer_name == "ConvolutionLayer" or layer_name == "TransposedConvolutionLayer", "Layer name unknown"
        self.test_name = test_name
        self.layer_name = layer_name
        if self.layer_name == "ConvolutionLayer":
            self.function_name = "convolve"
        elif self.layer_name == "TransposedConvolutionLayer":
            self.function_name = "transposed_convolve"
        self.random_test_objects = random_test_objects


class RandomArrayTestObject:
    def __init__(self, input_arr, kernel, output_arr, bias, padding, stride=1):
        """Struct that represents one test case for the random array tests.
        input_arr: ndarray, 3-Dimensional floating point numpy array
        output_arr: ndarray, 3-Dimensional floating point numpy array
        kernel: ndarray, 3-Dimensional floating point numpy array, weights of
        the convolutional layer
        stride: int
        padding: str, valid or same. Valid padding just applies the kernel directly,
        same padding ensures that inputsize = outputsize
        """
        if padding == "VALID":
            self.padding = "Padding::Valid"
        elif padding == "SAME":
            self.padding = "Padding::Same"
        else:
            raise ValueError(f"Illegal padding value {padding}")

        self.input_arr = numpy_array_to_rust(input_arr, shape_vec=True)
        self.output_arr = numpy_array_to_rust(output_arr, shape_vec=True)
        self.kernel = numpy_array_to_rust(kernel, shape_vec=True)
        if bias is None:
            self.bias = "None"
        else:
            self.bias = numpy_array_to_rust(
                bias, shape_vec=True, bias=True)
        self.stride = stride


def numpy_array_to_rust(x, shape_vec=False, bias=False):
    """
        Outputs a numpy array as a Rust ndarray.
        If shape_vec is set to true, outputs
        the array creationg through the shape_vec Rust function.
        The Rust array macro seems broken for 4-D arrays, so this is a
        workaround.
    """
    # This removes the "dtype=..." info in the representation,
    # if needed
    if x.dtype == np.float64:
        ending_delimiter = -1
    elif x.dtype == np.float32:
        ending_delimiter = -16
    else:
        raise ValueError("array has an unsupported datatype: {x.dtype}")

    if shape_vec:
        x_shape = x.shape
        x = x.flatten()
    # removes leading array and closing paren tokens
    array_repr = f"{repr(x)}"[6:][:ending_delimiter].replace("\n", "\n\t\t")
    if shape_vec:
        if bias:
            return f"Some(Array::from_shape_vec({x_shape}, vec!{array_repr}).unwrap())"
        else:
            return f"Array::from_shape_vec({x_shape}, vec!{array_repr}).unwrap()"
    else:
        return f"array!{array_repr}".rstrip().rstrip(",")


def torch_to_tf_img(x):
    return np.moveaxis(x, 0, 2)


def tf_to_torch_img(x):
    return np.moveaxis(x, 2, 0)


def tf_to_torch_ker(k):
    return np.moveaxis(k, [2, 3], [1, 0])


def tf_to_torch_ker_transpose(k):
    return np.moveaxis(k, [2, 3], [0, 1])


def transform_img(orig, dest, x):
    """Transforms img between orig and dest data formats.

    orig: str, tf, rust, or pt
    dest: str, tf, rust, or pt
    x: ndarray to transform
    """
    return transform(orig, dest, x, is_kernel=False)


def transform_ker(orig, dest, x):
    """Transforms ker between orig and dest data formats.

    orig: str, tf, rust, or pt
    dest: str, tf, rust, or pt
    x: ndarray to transform
    """
    return transform(orig, dest, x, is_kernel=True)


def transform(orig, dest, x, is_kernel):
    """
    Transforms an array from the origin dataformat to the destination data format.
    Used to convert between the different dataformats used.

    These are (im, ker, im_transpose, ker_transpose):
    B = Batch, H = Height, W = Width, C = Channel, I = Input channels, O = Output channels
    TF = Tensorflow, PT = Pytorch, Rust = our implementation

    rust: CHW, OIHW, CHW, IOHW
    tf: BHWC, HWIO, BHWC, HWOI
    pt: BCHW, OIHW, BCHW, IOHW

    orig: str, tf, rust, or pt
    dest: str, tf, rust, or pt
    x: ndarray to transform
    transpose: whether
    kernel: bool, True if x is a kernel
    """
    orig = orig.lower()
    dest = dest.lower()

    if orig == dest:
        print("Warning: Tried to convert orig to same dest.")
        return x

    # We first convert everything to PT
    if orig == "rust":
        if not is_kernel:
            x = np.expand_dims(x, axis=0)

    elif orig == "tf":
        if not is_kernel:
            x = np.moveaxis(x, 3, 1)
        if is_kernel:
            x = np.moveaxis(x, [2, 3], [1, 0])

    # now x is in PT format
    if dest == "rust":
        if not is_kernel:
            x = np.squeeze(x, axis=0)
        # if is_kernel:
            # x = np.moveaxis(x, 0, 1)

    elif dest == "tf":
        if not is_kernel:
            x = np.moveaxis(x, 1, 3)
        else:
            x = np.moveaxis(x, [0, 1], [3, 2])

    return x


def conv2d_random_array_test(img_shapes, kernel_shapes, num_arrays_per_case=3, use_torch=False, transpose=False, seed=260896, padding="VALID", stride=1, bias=False, compare_impls=True):
    """Returns a Test case that can be rendered with the
    test_py_impl_random_arrays_template.rs into a Rust test
    that tests the conv2d Rust implementation against tf.nn.conv2d.

    num_arrays_per_case: int, number of different random arrays generated
    per (img_shape, kernel_shape) combination
    use_torch: bool, set to true if we should use the pytorch implementation to compare against.
    False for the tensorflow implementation"""
    # The implementation works as follows with data formats:
    #                     IMG, KER (Rust)  ---------------
    #                   /                 \               \
    #           to tf  /                   \ to pt         \
    #                 v                     v               \
    #              IMG, KER (TF)          IMG, KER (PT)      \
    #                 /                       \               \
    #      tf conv2d /                         \  pt conv2d    \  our implementation
    #               v                           v               \
    #              OUT (TF)                    OUT (PT)          \
    #              /                             \                \
    #     to rust /                               \ to rust        \
    #            v                                 v                \
    #           out1  <-- Compare (in python) --> out2               \
    #                             |                                   v
    #                             ------------------------------->  out3
    #                                  Compare (in rust testcase)

    np.random.seed(seed)

    if transpose and padding == "VALID" and (compare_impls or not use_torch):
        raise ValueError(
            "Valid padding not useable with tensorflow transposed convolution.")
    if stride > 1 and padding.lower() == "same" and (compare_impls or not use_torch):
        raise ValueError("Stride > 1 with same padding: It is known, that Pytorch and Tensorflow have differing implementations. See https://stackoverflow.com/questions/52975843/comparing-conv2d-with-padding-between-tensorflow-and-pytorch. We abide by the Pytorch implementation. Aborting test generation.")

    objects = []
    for im_shape, ker_shape in list(itertools.product(img_shapes, kernel_shapes)):
        if not transpose and im_shape[0] != ker_shape[1] or transpose and im_shape[0] != ker_shape[0]:
            continue  # shapes are not compatible, channel size mismatch

        for i in range(num_arrays_per_case):
            im = np.random.rand(*im_shape).astype(dtype=np.float32)
            ker = np.random.rand(*ker_shape).astype(dtype=np.float32)
            if bias:
                if transpose:
                    bias_shape = ker_shape[1]
                else:
                    bias_shape = ker_shape[0]
                bias_vec = np.random.rand(bias_shape).astype(dtype=np.float32)
            else:
                bias_vec = None

            # Generating the PT and TF parts

            if compare_impls or use_torch:
                im_pt = torch.FloatTensor(transform_img("rust", "pt", im))
                ker_pt = torch.FloatTensor(transform_ker("rust", "pt", ker))
                if bias:
                    bias_pt = torch.FloatTensor(bias_vec)
                else:
                    bias_pt = None

                if transpose:
                    if padding == "SAME":
                        # This simulates padding = "SAME"
                        out_pt = torch.nn.functional.conv_transpose2d(
                            im_pt, ker_pt, bias_pt, output_padding=stride-1, padding=(ker_pt.shape[2] // 2, ker_pt.shape[3]//2), stride=stride)
                    else:
                        out_pt = torch.nn.functional.conv_transpose2d(
                            im_pt, ker_pt, bias_pt, padding=0, stride=stride)
                else:
                    if padding == "SAME" and stride > 1:
                        out_pt = torch.nn.functional.conv2d(
                            im_pt, ker_pt, bias_pt, padding=(ker_pt.shape[2] // 2, ker_pt.shape[3]//2), stride=stride)
                    else:
                        out_pt = torch.nn.functional.conv2d(
                            im_pt, ker_pt, bias_pt, padding=padding.lower(), stride=stride)

                out_pt_numpy = transform_img("pt", "rust", out_pt.numpy())

            if compare_impls or not use_torch:
                # axis 0 is batch dimension, which we need to remove and add back in
                im_tf = tf.constant(transform_img(
                    "rust", "tf", im), dtype=tf.float32)
                ker_tf = tf.constant(transform_ker(
                    "rust", "tf", ker), dtype=tf.float32)

                if transpose:
                    output_shape = (
                        1, stride*im_shape[1], stride*im_shape[2], ker_shape[1])
                    # conv2d transpose expected filters as [height, width, out, in]
                    # https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose
                    out_tf = tf.nn.conv2d_transpose(
                        im_tf, ker_tf, output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)
                else:
                    out_tf = tf.nn.conv2d(im_tf, ker_tf, strides=[
                        1, stride, stride, 1], padding=padding)
                out_tf_numpy = transform_img("tf", "rust", out_tf.numpy())

            # Comparing implementations
            if compare_impls:
                assert np.allclose(
                    out_tf_numpy, out_pt_numpy), f"Torch and Tensorflow implementations didn't match.\nTorch: {out_pt_numpy}\n Tensorflow:{out_tf_numpy}"

            if use_torch:
                out = out_pt_numpy
            else:
                out = out_tf_numpy

            # TODO() write out bias with shape (output_channel, 1)
            # Rust expects Array2<F>
            # Writing the test objects
            test_obj = RandomArrayTestObject(
                im, ker, out, bias_vec, padding, stride=stride)
            objects.append(test_obj)

    if transpose:
        transpose_string = "_transpose"
        layer_name = "TransposedConvolutionLayer"
    else:
        transpose_string = ""
        layer_name = "ConvolutionLayer"

    if use_torch:
        test_name = "conv2d_torch"
    else:
        test_name = "conv2d"

    if stride != 1:
        stride_string = f"_stride{stride}"
    else:
        stride_string = ""
    return RandomArrayTest(f"{test_name}{transpose_string}{stride_string}", layer_name, objects)


def write_test_to_file(ml_test_folder, test_content, test_name):
    test_filename = f"{test_name}_automated_test.rs"
    test_output_filepath = os.path.join(
        ml_test_folder, test_filename)
    with open(test_output_filepath, "w+") as conv2d_output_file:
        conv2d_output_file.write(test_content)
        print(f"Successfully wrote {test_name} test to {test_output_filepath}")
    os.system(f"rustfmt {test_output_filepath}")
    print(f"Formatted {test_name} test.")


def main():
    # Tensorflow conv2d inputs are given as
    # - batch_shape + [in_height, in_width, in_channels]
    # and weights as
    # - [filter_height * filter_width * in_channels, output_channels]
    # See also: https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
    # analog for conv2d_transpose:
    # https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose

    # Im shapes: Channels, Height, Width. Ker shapes are Out, In, Height, Width
    # (the format used in Rust)

    # Note: Tensorflow does not provide an option to input bias; so bias is tested with pytorch

    img_shapes = [(1, 5, 12), (1, 10, 15), (1, 15, 10),
                  (3, 6, 6), (3, 10, 15), (3, 15, 10)]
    kernel_shapes = [(2, 1, 3, 4), (2, 1, 5, 5),
                     (2, 3, 3, 3), (2, 3, 5, 5)]

    img_shapes_trans = [(2, 5, 4), (2, 4, 3), (2, 6, 6), (1, 4, 5), (1, 3, 3)]
    kernel_shapes_trans = [(2, 1, 4, 4), (1, 1, 4, 4),
                           (2, 1, 3, 3), (2, 1, 5, 5)]
    # tests for different output channels
    img_shapes_trans_test_different_channels = [(1, 3, 3), (3, 2, 2)]
    kernel_shapes_trans_test_different_channels = [
        (3, 2, 4, 4), (3, 2, 6, 6), (1, 2, 3, 3), (1, 2, 5, 5)]

    np.set_printoptions(suppress=True)
    # loading Jinja with the random array test template
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loader = jinja2.FileSystemLoader(
        os.path.join(project_root, "scripts", "templates"))
    env = jinja2.Environment(loader=loader)
    template = env.get_template("test_py_impl_random_arrays_template.rs")
    ml_test_folder = os.path.join(project_root, "tests")

    # stride = 1, padding= "VALID", normal conv
    conv2d_test_case = conv2d_random_array_test(
        img_shapes, kernel_shapes, padding="VALID", compare_impls=True)
    conv2d_test_content = template.render(
        random_tests=[conv2d_test_case], file=__file__)
    write_test_to_file(ml_test_folder, conv2d_test_content,
                       "conv2d_stride1_valid_both")

    # writing out the conv2d_stride2 test cases with stride 2
    # we need to skip the uneven kernel, as same padding in torch
    # only really works with uneven kernels (else we would need
    # a full pad operation)
    # We cannot compare the impls, as PT and TF disagree here. By convention,
    # we will stick to the PT implementation.
    # See: https://stackoverflow.com/questions/52975843/comparing-conv2d-with-padding-between-tensorflow-and-pytorch
    conv2d_stride2_test_case = conv2d_random_array_test(
        img_shapes, kernel_shapes[1:], stride=2, padding="SAME", compare_impls=False, use_torch=True)
    conv2d_stride2_test_content = template.render(
        random_tests=[conv2d_stride2_test_case], file=__file__)
    write_test_to_file(
        ml_test_folder, conv2d_stride2_test_content, "conv2d_stride2_same_torch")

    # stride 2, padding = "VALID", conv2 transpose, no bias
    conv2d_transpose_test_case = conv2d_random_array_test(
        img_shapes_trans, kernel_shapes_trans, transpose=True, padding="VALID", compare_impls=False, use_torch=True, stride=2)
    conv2d_transpose_test_content = template.render(
        random_tests=[conv2d_transpose_test_case], file=__file__)
    write_test_to_file(ml_test_folder, conv2d_transpose_test_content,
                       "conv2d_transpose_stride2_valid_torch")

    # transpose, padding same, stride2. Uses torch, as impls cant be compared
    # only uneven kernel sizes usable
    # stride 2, padding = "SAME", conv2 transpose, no bias
    conv2d_stride2_same_transpose_test_case = conv2d_random_array_test(
        img_shapes_trans, kernel_shapes_trans[2:], transpose=True, padding="SAME", compare_impls=False, use_torch=True, stride=2)
    conv2d_stride2_transpose_test_content = template.render(
        random_tests=[conv2d_stride2_same_transpose_test_case], file=__file__)
    write_test_to_file(ml_test_folder, conv2d_stride2_transpose_test_content,
                       "conv2d_transpose_stride2_same_torch")

    # writing out the conv2d_tranposed test cases for
    # a change in channel (output != input channel)
    conv2d_transpose_test_case = conv2d_random_array_test(
        img_shapes_trans_test_different_channels, kernel_shapes_trans_test_different_channels[2:], transpose=True, padding="SAME", compare_impls=False, stride=2, use_torch=True)
    conv2d_transpose_test_content = template.render(
        random_tests=[conv2d_transpose_test_case], file=__file__)
    write_test_to_file(ml_test_folder, conv2d_transpose_test_content,
                       "conv2d_transpose_stride2_test_different_in_out_channel")

    # transpose, padding valid, stride1
    conv2d_transpose_test_case = conv2d_random_array_test(
        img_shapes_trans, kernel_shapes_trans, transpose=True, padding="VALID", compare_impls=False, use_torch=True, bias=True)
    conv2d_transpose_test_content = template.render(
        random_tests=[conv2d_transpose_test_case], file=__file__)
    write_test_to_file(ml_test_folder, conv2d_transpose_test_content,
                       "conv2d_transpose_stride1_valid_torch_bias")

    # stride 1, padding = "VALID", normal conv, bias
    conv2d_stride2_torch_test_case = conv2d_random_array_test(
        img_shapes, kernel_shapes, use_torch=True, stride=1, padding="VALID", bias=True, compare_impls=False)
    conv2d_stride2_torch_test_content = template.render(
        random_tests=[conv2d_stride2_torch_test_case], file=__file__)
    write_test_to_file(
        ml_test_folder, conv2d_stride2_torch_test_content, "conv2d_stride2_valid_torch_bias")


if __name__ == "__main__":
    main()
