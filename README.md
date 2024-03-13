# ResNet-152 Model Testing with PyTorch

This repository aims to demonstrate image classification using a pre-trained ResNet-152 model provided by PyTorch. 

![ResNet152 testing](https://github.com/HR-Fahim/ResNet-152-Model-Testing-with-PyTorch/assets/66734379/ff33c7d5-c238-4145-839c-48d2198c531c)

## Purpose
The purpose of this project is to showcase how deep learning models, specifically convolutional neural networks (CNNs), can be used for image classification tasks. By leveraging the ResNet-152 architecture, which is known for its exceptional performance on various image recognition benchmarks, we aim to illustrate the effectiveness of pre-trained models in recognizing objects within images.

## Methodology
The project involves the following steps:
- Loading a pre-trained ResNet-152 model provided by PyTorch.
- Preprocessing an image to match the model's input requirements.
- Performing inference on the preprocessed image to obtain predictions.
- Interpreting the model's output to identify the top predicted classes and their probabilities.

## Why ResNet-152?
ResNet-152 is chosen for its depth and performance. With 152 layers, it can capture intricate features in images, making it suitable for a wide range of classification tasks. Additionally, it has been pre-trained on large-scale datasets like ImageNet, which contributes to its ability to generalize well to unseen data.

<img width="958" alt="resnet_architecture" src="https://github.com/HR-Fahim/ResNet-152-Model-Testing-with-PyTorch/assets/66734379/5e7ca7f8-7a33-469a-9e77-619b0a8479f0">

## Why PyTorch?
PyTorch is a widely used deep learning framework known for its flexibility and ease of use. By utilizing PyTorch, this project provides an accessible implementation of image classification with ResNet-152, enabling researchers and practitioners to easily experiment with deep learning models for image recognition tasks.

## Future Improvements
Potential future improvements for this project include:
- Integration with web or mobile applications for real-time image classification.
- Fine-tuning the pre-trained model on custom datasets to improve performance on specific domains.
- Exploring other pre-trained models or architectures for comparison and benchmarking.

## Acknowledgments
- The pre-trained ResNet-152 model is provided by [PyTorch](https://pytorch.org/).
- Here used one common source PyTorch torchvision repository, where they provided a [imagenet_classes.txt](https://github.com/pytorch/hub.git) file containing the class labels for the ImageNet dataset.

