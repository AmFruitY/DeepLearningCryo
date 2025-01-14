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


target_shape = (128, 128)



class SiameseModel(Model):
    def __init__(self, siamese_network, margin=0.75, **kwargs):
        super().__init__(**kwargs)
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        ap_distance, an_distance = self.siamese_network(data)
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]

    def get_config(self):
        config = super().get_config()
        config.update({
            "siamese_network": tf.keras.utils.serialize_keras_object(self.siamese_network),
            "margin": self.margin,
        })
        return config

    @classmethod
    def from_config(cls, config):
        siamese_network_config = config.pop("siamese_network")
        siamese_network = tf.keras.utils.deserialize_keras_object(siamese_network_config)
        return cls(siamese_network=siamese_network, **config)

    def get_build_config(self):
        return {
            "input_shape": self.siamese_network.input_shape,
        }

    def build_from_config(self, config):
        input_shape = config.get("input_shape")
        if input_shape:
            if not self.siamese_network.built:
                self.siamese_network.build(input_shape)
        else:
            raise ValueError("Missing `input_shape` in build config")

