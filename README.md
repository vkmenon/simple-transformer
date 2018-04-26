# Simple Transformer

## Overview
The goal of this repository is to a version of the Transformer (https://arxiv.org/abs/1706.03762) model that is:

* easy to read & understand
* easy to modify & add to existing pipelines

I will also work on a script to compile a Keras compatible model for maximum ease of use (however, it will only work using the TensorFlow backend since I am using Tensorflow specific functions and not K.backend).

## Model Overview

<p align="center">
  <img width="200" src="imgs/transformer_image.png">
</p>

Unlike most sequence to sequence models today, the Transformer uses Self-Attention (or Intra-Attention) as both the encoding unit and decoding units. In addition, there is a "traditional" attention module in the Decoder that attends to the outputs of the Encoder.

### Why Self-Attention (vs RNN vs CNN)?

For more information on attention in seq2seq models, check out Section 8 of Graham Neubig's great (both comprehensive and concise!) [tutorial on Neural Machine Translation](https://arxiv.org/abs/1703.01619). Of course, Attention is used outside of just NMT (pretty much any seq2seq model benefits from it, such as Text-to-Speech, Speech-to-Text, Q&A, and even convolutional models).

Self Attention is simply a form of attention in which each sample of the input attends to every other sample in the input (rather than each hidden state of a decoder attending to every hidden state of an encoder). 

Because each sample in the input is directly attending to all other inputs, the information flow between samples that are far from each other in the sequence itself is very short / direct.

In a recurrent neuron, the information flow between samples occurs through the changing hidden states, which can lose information if the distance between samples (or the sequence length) gets too long - even in improved RNN's such as Long-Short-Term-Memory Units or Gated Recurrent Units. Convolutional seq2seq models help in that convolutional kernels can have fixed width but are limited by the kernel width (at least at the first layer). 

In addition, Self-Attention as implemented in the paper (and this repo) is very easy to compute since it is a simple matrix multiplication. This allows for fast, parallelized training due to a lack of dependencies in the computational graph. 

## Usage

(WIP)

## TO DO:

|    **Item**    | **Status** |
|----------------|------------|
| Build Model    | :thumbsup: |
| Input Pipeline |            |
| Add NMT Tools  |            |
| Music Tools    |            |
