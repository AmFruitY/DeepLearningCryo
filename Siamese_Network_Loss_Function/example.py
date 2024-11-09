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

# Hides warning to rebuild TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras import mixed_precision

# Set the global mixed precision policy to 'mixed_float16'
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


# Runtime initialization will not alocate all the memory on the device
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Classes
from SiameseModel import SiameseModel
from DistanceLayer import DistanceLayer

target_shape = (200, 200)


# Paths! 
cache_dir = Path("/mnt/c/Users/joshu/Desktop/TFG/DeepLearningCryo/Siamese_Network_Loss_Function/.keras/") # ".keras"
anchor_images_path = Path("/mnt/c/Users/joshu/Desktop/TFG/DeepLearningCryo/Siamese_Network_Loss_Function/.keras/left/")
positive_images_path = Path("/mnt/c/Users/joshu/Desktop/TFG/DeepLearningCryo/Siamese_Network_Loss_Function/.keras/right/")

def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image


def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )

# We need to make sure both the anchor and positive images are loaded in
# sorted order so we can match them together. 
anchor_images = sorted(
    [str(anchor_images_path / f) for f in os.listdir(anchor_images_path)]
)
# This is to create the full path to the image we want to use. To explain it we must do it in step by step:
# This is a list comprehension in which 1) Using a for loop, for every f in the path we acquire the name of the image
# 2) Then we concatenate the image file with the path we have already specified from before.
# 3) Finally, we want to sort the list since it is much more convenient.


positive_images = sorted(
    [str(positive_images_path / f) for f in os.listdir(positive_images_path)]
)

image_count = len(anchor_images)

anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)

# To generate the list of negative images, let's randomize the list of
# available images and concatenate them together.
rng = np.random.RandomState(seed=42)
rng.shuffle(anchor_images)
rng.shuffle(positive_images)

negative_images = anchor_images + positive_images
np.random.RandomState(seed=32).shuffle(negative_images)

negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
negative_dataset = negative_dataset.shuffle(buffer_size=4096)

dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.map(preprocess_triplets)

# Let's now split our dataset in train and validation.
train_dataset = dataset.take(round(image_count * 0.8))
val_dataset = dataset.skip(round(image_count * 0.8))

train_dataset = train_dataset.batch(32, drop_remainder=False)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.batch(32, drop_remainder=False)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

def visualize(anchor, positive, negative):
    """Visualize a few triplets from the supplied batches."""

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(9, 9))

    axs = fig.subplots(3, 3)
    for i in range(3):
        show(axs[i, 0], anchor[i])
        show(axs[i, 1], positive[i])
        show(axs[i, 2], negative[i])

    plt.show()


visualize(*list(train_dataset.take(1).as_numpy_iterator())[0])

#%% Setting the embedding model

base_cnn = resnet.ResNet50(
    weights="imagenet", input_shape=target_shape + (3,), include_top=False
)

flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(512, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)

embedding = Model(base_cnn.input, output, name="Embedding")

trainable = False
for layer in base_cnn.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable

# Setting up the Siamese Network

anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
positive_input = layers.Input(name="positive", shape=target_shape + (3,))
negative_input = layers.Input(name="negative", shape=target_shape + (3,))

distances = DistanceLayer()(
    embedding(resnet.preprocess_input(anchor_input)),
    embedding(resnet.preprocess_input(positive_input)),
    embedding(resnet.preprocess_input(negative_input)),
)

siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)

#%% Training


siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.0001))
siamese_model.summary()
siamese_model.fit(train_dataset, epochs=2, validation_data=val_dataset,batch_size=2)

# Saving the model in my local directory
# savedir = Path("/mnt/c/Users/joshu/Desktop/TFG/DeepLearningCryo/Siamese_Network_Loss_Function/siamesetlktrained/")
# siamese_model.save(savedir)

# Saving the model weights in my local directory
# saveweightsdir = Path("/mnt/c/Users/joshu/Desktop/TFG/DeepLearningCryo/Siamese_Network_Loss_Function/siamesetlktrainedweights/")
# siamese_model.save_weights(saveweightsdir)


#%% Inspecting

sample = next(iter(train_dataset))
visualize(*sample)


anchor, positive, negative = sample
anchor_embedding, positive_embedding, negative_embedding = (
    embedding(resnet.preprocess_input(anchor)),
    embedding(resnet.preprocess_input(positive)),
    embedding(resnet.preprocess_input(negative)),
)


cosine_similarity = metrics.CosineSimilarity()

positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
print("Positive similarity:", positive_similarity.numpy())

negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
print("Negative similarity", negative_similarity.numpy())
