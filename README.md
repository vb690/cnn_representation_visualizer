
# Anatomy of a Convolutional Neural Network

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/vb690/cnn_representation_visualizer/main/visualizer_app.py)

## Motivation

This App aims to visualize changes in filters and representation produced by a Convolutional Artificial Neural Network (LeNet-5) while learning how to classify different clothing categories present in the Fashion-MNIST dataset.  
  
For maximizing the observed differences and reducing memory overhead, the observed training period is represented by the sequence of random batches observed during a single training epoch.

## Features

* Script for trainining-on-batch of LeNet-5 on Fashion-MNIST.
  * Extracting learned filters over training.
  * Extracting learned embeddings oser training.
  * Running AlignedUMAP ovser sequence of extracted embeddings.
* Streamlit App
  * Visulize changes in learned filters over training.
  * Visulize changes in UMAP reduction of learned embedding over training.

## How to use  

Go to the live app clicking on the streamlit badge on top of this page.

## License

[The MIT License](https://github.com/vb690/bazaar/blob/master/LICENSE)
