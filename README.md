# Deep Learning for Cryo-EM Image Classification
Welcome! This repository contains the code for my final degree project. I'm Josh—feel free to reach out with any questions!

This project explores whether a similarity network can learn from cryo-electron microscopy images. The code is still being cleaned up, but I aim to complete it soon (last update: 06/02/2025).

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

