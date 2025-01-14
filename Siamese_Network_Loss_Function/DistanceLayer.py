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
from keras import losses


target_shape = (128, 128)

    
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

        #cosine_similarity = metrics.CosineSimilarity() # We want the to train it using the cosine similarity since
                                                    # it is the metric that we will be using to evaluate the similarity
        # cosine_similarity.reset_state()
        # ap_distance = cosine_similarity(anchor, positive).numpy()
        # cosine_similarity.reset_state()
        # an_distance = cosine_similarity(anchor, negative).numpy()
        # cosine_similarity.reset_state()
        

        # Compute cosine similarity
        # ap_distance = 1 + losses.cosine_similarity(anchor, positive, axis=-1)
        # an_distance = 1 + losses.cosine_similarity(anchor, negative, axis=-1)

        # The commented part of this code is the Euclidean distance.
        ap_distance = ops.sum(tf.square(anchor - positive), -1)
        an_distance = ops.sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

    def get_config(self):
        # Return the config dictionary
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        # Create the layer from the config dictionary
        return cls(**config)