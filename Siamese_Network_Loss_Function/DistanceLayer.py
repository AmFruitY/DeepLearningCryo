import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from pathlib import Path
from keras import applications
from keras import layers
from keras import losses
from keras import ops
from keras import optimizers
from keras import metrics
from keras import Model
from keras.applications import resnet


target_shape = (200, 200)

class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    """ * The original code was done using 
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):

        cosine_similarity = metrics.CosineSimilarity() # We want the to train it using the cosine similarity since
                                                    # it is the metric that we will be using to evaluate the similarity

        ap_distance = cosine_similarity(anchor, positive)
        an_distance = cosine_similarity(anchor, negative)

        # The commented part of this code is the Euclidean distance.
        # ap_distance = ops.sum(tf.square(anchor - positive), -1)
        # an_distance = ops.sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)