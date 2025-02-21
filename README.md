# Deep Learning for Cryo-EM Image Classification
Welcome! This repository contains the code for my final degree project. I'm Josh—feel free to reach out with any questions!

This project explores whether a similarity network can learn from cryo-electron microscopy images with Python. The code is still being cleaned up, but I aim to complete it soon (last update: 21/02/2025).

## Libraries and configuration

### GPU Configuration
This code was developed on a Windows laptop with an RTX 3050 GPU. Training on the CPU was too slow, so I configured TensorFlow to use the GPU.

To simplify the setup, I installed WSL (Windows Subsystem for Linux), which made TensorFlow-GPU installation much easier. I followed the official TensorFlow WSL2 guide for setup: https://www.tensorflow.org/install/pip .

### Virtual Environment
A key skill I learned during my internship was using virtual environments to manage dependencies. To create one, run:

`python -m venv /path/to/new/virtual/environment`

After setting it up, install the required packages by doing `pip install <library>`. The versions I used are (if there are any other package version you want me to check, feel free to contact me):

- numpy==2.0.2
- PyQt6==6.8.0
- PyQt6-Qt6==6.8.1
- scikit-learn==1.6.0
- scipy==1.14.1
- tensorflow==2.18.0

## Model Architecture
The model is inspired by an article on the Keras website. This project applies the same approach to cryo-EM images. As a former physics student, I found deep learning challenging, but Deep Learning with Python by François Chollet helped immensely.

## Dataset
The dataset was generated with my advisor, Dr. David Maluenda Niubó. It includes:

- Noisy images: Simulate real low signal-to-noise conditions.
- Clear images: Represent ideal conditions.
Both sets help assess the model’s capabilities.

## Code Overview
- model_train.py – Trains and evaluates on similar unseen data.
- model_predict.py – Evaluates on a different type of unseen data.
Since the test dataset cannot be saved after training, the trained model must be loaded separately for inference.

Additionally:

*SiameseModel.py* & *DistanceLayer.py* – Supporting scripts for the main models.
This was my first experience with Object-Oriented Programming—I'll refine it as I learn more.

