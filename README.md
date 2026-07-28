[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/doughdemon/modular-addition/blob/master/notebook.ipynb)

# Geometry of neural networks performing modular addition

This repo contains scripts and a notebook to train a neural network on a modular addition task and to visualize its embeddings (in 2d and 3d) and algorithms.

The neural net architecture is from [^1]. Among others, the notebook reproduces evidence for the Pizza algorithm from [^2].

In addition, there also are stand-alone scripts to train the model (`train_cyclic.py`) and to analyze the net.

## How to run

1. Open the notebook in Colab
2. For training, set the runtime to T4 GPU
3. Run the setup (this connects to Google Drive for checkpoint storage)
4. Run the rest of the notebook


[^1]: A Toy Model of Universality: Reverse Engineering How Networks Learn Group Operations
Bilal Chughtai, Lawrence Chan, Neel Nanda
https://arxiv.org/abs/2302.03025

[^2]: The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks
Ziqian Zhong, Ziming Liu, Max Tegmark, Jacob Andreas
https://arxiv.org/abs/2306.17844
