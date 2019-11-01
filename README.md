# End-to-end Microphone Permutation and Number Invariant Multi-channel Speech Separation

This repository provides the model implementation and dataset generation scripts for the paper "End-to-end Microphone Permutation and Number Invariant Multi-channel Speech Separation" by Yi Luo, Zhuo Chen, Nima Mesgarani and Takuya Yoshioka. The paper introduces ***transform-average-concatenate (TAC)***, a simple module to allow end-to-end multi-channel separation systems to be invariant to microphone permutation (indexing) and number. Although designed for ad-hoc array configuration, TAC also provides significant performance improvement in fixed geometry microphone configuration, showing that it can serve as a general design paradigm for end-to-end multi-channel processing systems.

## Model

We implement TAC in the framework of ***filter-and-sum network (FaSNet)***, a recently proposed multi-channel speech separation model operated in time-domain. FaSNet is a neural beamformer that performs the standard filter-and-sum beamforming in time domain, while the beamforming coefficients are estimated by a neural network in an end-to-end fashion. Here we provide the original FaSNet implementation and the modified FaSNet with TAC applied. For details please refer to the paper.

The building blocks in the FaSNet models are ***dual-path RNNs (DPRNNs)***, a simple yet effective method for organizing RNN layers to allow successful modeling of extremely long sequential data. For details about DPRNN please refer to ["Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation"](https://arxiv.org/abs/1910.06379). The implementation of DPRNN, as well as the combination of DPRNN and TAC, can be found in [*utility/models*](https://github.com/yluo42/TAC/blob/master/utility/models.py).

## Dataset

The evaluation of the model is on both ad-hoc array and fixed geometry array configurations. For data generation please refer to the *data* folder.
