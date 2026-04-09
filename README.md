# compressed-vv-krr

MATLAB implementation of compressed vector-valued kernel ridge regression (VV-KRR) for data-efficient surrogate modeling of electromagnetic structures and microwave components.

## Overview

This repository contains the MATLAB code used to reproduce the results of the first benchmark example presented in the paper, including:

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
- scripts for training, prediction, error evaluation, and figure generation for Example 1.

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

## Repository structure

```text
.
├── data/              % dataset for Example 1
├── examples/          % scripts to reproduce Example 1
├── kernels/           % kernel functions
├── training/          % VV-KRR training routines
├── prediction/        % VV-KRR prediction routines
├── baselines/         % PCA+GPR and DNN baseline scripts
├── utils/             % utility functions
├── README.md
├── LICENSE
