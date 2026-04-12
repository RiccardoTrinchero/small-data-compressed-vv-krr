
## Overview

This repository contains the MATLAB code used to reproduce the results of the first benchmark example presented in the paper "Small-Data Modeling of Electromagnetic Structures via Compressed Vector-Valued Kernel Ridge Regression" (submitted), including:

- the proposed compressed VV-KRR model,
- the PCA+GPR baseline,
- the DNN baseline.

The code is intended for small-data multi-output regression problems arising in electromagnetic and microwave modeling, where training samples are expensive to generate and output responses are high-dimensional.

## Included material

This public release currently includes:

- the dataset for Example 1,
- the MATLAB implementation of the proposed compressed VV-KRR method,
- the MATLAB implementation of the PCA+GPR comparison model,
- the MATLAB implementation of the DNN comparison model,

The other examples discussed in the paper are not included in this release.

## Main features

- Compressed training of separable VV-KRR models
- Support for input kernels:
  - RBF
  - ARD RBF
  - Matérn 5/2
  - ARD Matérn 5/2
  - Matérn 1/2
  - ARD Matérn 1/2
- Support for:
  - analytic output kernels
  - data-driven output kernels based on output correlation
- K-fold cross-validation for hyperparameter estimation
- Baseline comparison with:
  - PCA + Gaussian-process regression
  - fully connected deep neural network

