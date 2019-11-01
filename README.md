# End-to-end Microphone Permutation and Number Invariant Multi-channel Speech Separation

This repository provides the model implementation and dataset generation scripts for the paper "End-to-end Microphone Permutation and Number Invariant Multi-channel Speech Separation" by Yi Luo, Zhuo Chen, Nima Mesgarani and Takuya Yoshioka. The paper introduces ***transform-average-concatenate (TAC)***, a simple module to allow end-to-end multi-channel separation systems to be invariant to microphone permutation (indexing) and number. Although designed for ad-hoc array configuration, TAC also provides significant performance improvement in fixed geometry microphone configuration, showing that it can serve as a general design paradigm for end-to-end multi-channel processing systems.

## Model

We implement TAC in the framework of ***filter-and-sum network (FaSNet)***, a recently proposed multi-channel speech separation model operated in time-domain. FaSNet is a neural beamformer that performs the standard filter-and-sum beamforming in time domain, while the beamforming coefficients are estimated by a neural network in an end-to-end fashion. For details please refer to the original paper: ["FaSNet: Low-latency Adaptive Beamforming for Multi-microphone Audio Processing"](https://arxiv.org/abs/1909.13387).

In this paper we make two main modifications to the original FaSNet:
1) Instead of the original two-stage architecture, we change it into a single-stage architecture.
2) TAC is applied throughout the filter estimation module to synchronize the information in different microphones and allow the model to perform *global* decision while estimating the filter coeffients.

We show that such modifications lead to significantly better separation performance in both ad-hoc array with varying number of microphones and fixed circular array configurations.

The building blocks for the filter estimation modules are based on ***dual-path RNNs (DPRNNs)***, a simple yet effective method for organizing RNN layers to allow successful modeling of extremely long sequential data. For details about DPRNN please refer to ["Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation"](https://arxiv.org/abs/1910.06379). The implementation of DPRNN, as well as the combination of DPRNN and TAC, can be found in [*utility/models*](https://github.com/yluo42/TAC/blob/master/utility/models.py).

## Dataset

The evaluation of the model is on both ad-hoc array and fixed geometry array configurations. We simulate two datasets on the public available Librispeech corpus. For data generation please refer to the *data* folder.
