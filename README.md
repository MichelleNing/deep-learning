# Deep Learning (PyTorch)

This repository contains exercises related to Udacity's Deep Learning Nanodegree program. It consists of a bunch of excercise notebooks for various deep learning topics. The notebooks implement models such as convolutional networks, recurrent networks, and GANs. There are other topics covered such as weight initialization and batch normalization. 

## Table of Content
### Introduction to Nerual Networks
- Introduction to Neural Networks: implement gradient descent and apply it to predicting patterns in student admissions data.
- Introduction to PyTorch: build neural networks in PyTorch and use pre-trained networks for state-of-the-art image classifiers.
### Convolutional Nerual Networks
- Convolutional Neural Networks: Visualize the output of layers that make up a CNN. Define and train a CNN for classifying MNIST data, a handwritten digit database that is notorious in the fields of machine and deep learning. Also, define and train a CNN for classifying images in the CIFAR10 dataset.
- Transfer Learning. In practice, most people don't train their own networks on huge datasets; they use pre-trained networks such as VGGnet. Here use VGGnet to help classify images of flowers without training an end-to-end network from scratch.
- Weight Initialization: Explore how initializing network weights affects performance.
- Autoencoders: Build models for image compression and de-noising, using feedforward and convolutional networks in PyTorch.
- Style Transfer: Extract style and content features from images, using a pre-trained network. Implement style transfer according to the paper, Image Style Transfer Using Convolutional Neural Networks by Gatys et. al. Define appropriate losses for iteratively creating a target, style-transferred image of your own design!
### Recurrent Nerual Networks
- Intro to Recurrent Networks (Time series & Character-level RNN): Recurrent neural networks are able to use information about the sequence of data, such as the sequence of characters in text; learn how to implement these in PyTorch for a variety of tasks.
- Embeddings (Word2Vec): Implement the Word2Vec model to find semantic representations of words for use in natural language processing.
- Sentiment Analysis RNN: Implement a recurrent neural network that can predict if the text of a moview review is positive or negative.
- Attention: Implement attention and apply it to annotation vectors.
### Generative Adversarial Networks
- Generative Adversarial Network on MNIST: Train a simple generative adversarial network on the MNIST dataset.
- Batch Normalization: Learn how to improve training rates and network stability with batch normalizations.
- Deep Convolutional GAN (DCGAN): Implement a DCGAN to generate new images based on the Street View House Numbers (SVHN) dataset.
- CycleGAN: Implement a CycleGAN that is designed to learn from unpaired and unlabeled data; use trained generators to transform images from summer to winter and vice versa.
### Deploying a Model (with AWS SageMaker)
