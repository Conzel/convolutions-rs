# Scripts
This directory contains scripts there automatic test generation. 

## Automated unit tests
We have implemented automated unit test generation for convolution and transposed convolutions. These have to conform tightly to the tensorflow implementation (https://www.tensorflow.org/api_docs/python/tf/nn/conv2d and https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose respectively), which is why automatic code generation provides useful. We create random arrays, pass them through the tensorflow outputs and check the output of the Rust implementation against them. 

As always, tests can be run via `cargo test`. The tests can be generated through the script by simply running the script in `scripts/generate_tests.py` and modified through the file in the `templates` folder.

For this, we recommend `tensorflow 2.5.0`, `pytorch 1.10.2`, `numpy 1.19.5` and `Jinja2 3.0.0`.

We have provided a yaml file with a conda environment for this purpose. Install it by running `conda env create -f environment.yml` in this folder. Don't forget to activate the environment running `conda activate convolutions-rs`.