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
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)


# Runtime initialization will not alocate all the memory on the device
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Classes
from SiameseModel import SiameseModel
from DistanceLayer import DistanceLayer

# The example target shape is 200, 200
# I will be using 100,100 for the cryogenic molecules
target_shape = (100, 100)


# Paths! 
cache_dir = Path("/mnt/c/Users/joshu/Desktop/TFG/DeepLearningCryo/Siamese_Network_Loss_Function/data/")
images_path = [cache_dir / "clear1", cache_dir / "clear2", cache_dir / "clear3", cache_dir / "clear4"]

# anchor_images_path = cache_dir / "clear1"
# other_folders = [cache_dir / "clear2", cache_dir / "clear3", cache_dir / "clear4"]

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

# Use this seed so we can use reproducibility
random.seed(42)

# Given a list of elements, it creates another list in which the no element is in the same position as the original list.
def get_randomized_images(images):
    randomized = images[:] # Creates a copy of the list we want to randomize
    while True:
        random.shuffle(randomized) # Randomize the objects inside the said list
        if all(randomized[i] != images[i] for i in range(len(images))): # Only when each and every object in the list are different does the loop stops
            return randomized

# Generating the negative dataset - This part differs from the example as we want the negative dataset to be from different classes
# Function to get a random image from other folders
def get_random_image_from_folders(folders, num_images):
    all_images = []
    for folder in folders: # folders must be a list of the other folder paths
        all_images.extend([str(folder / f) for f in os.listdir(folder)]) # We use extend here to add the element to the back of the list
                                                                        # If we use the append method, it will add a list inside the list
    return [random.choice(all_images) for _ in range(num_images)]

# This function is to automize the creation of dataset to be able to create a larger dataset
# Since we want to use not only one class 
def create_dataset(list_of_image_paths):

    anchor_images = []
    positive_images = []
    negative_images =[]

    for i, path in enumerate(list_of_image_paths):
        anchor_images_individual = sorted([str(path / f) for f in os.listdir(path)]) # Get sorted list of anchor images - The anchor 
        positive_images_indivual = get_randomized_images(anchor_images_individual) # Function to get a randomized version of the list such that no image is in the same index
        
        anchor_images.extend(anchor_images_individual)
        positive_images.extend(positive_images_indivual)

        image_count = len(anchor_images_individual)

        other_paths = list_of_image_paths[:i] + list_of_image_paths[i+1:] # This excludes the current anchor path 
        negative_images_individual = get_random_image_from_folders(other_paths, image_count)
        negative_images.extend(negative_images_individual)

    return anchor_images, positive_images, negative_images

anchor_images, positive_images, negative_images = create_dataset(images_path)

image_count = len(anchor_images)


# Create the dataset for TensorFlow
anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)
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


# visualize(*list(train_dataset.take(1).as_numpy_iterator())[0])

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

# Create a list of the embedding vectors given a TensorFlow dataset
def embedding_vectors(dataset):

    embedding_vectors = []
    for sample in dataset:
        sample_image, _, _ = sample
        sample_image_embedding = embedding(resnet.preprocess_input(sample_image)) 
        embedding_vectors.extend(sample_image_embedding)
        print(len(embedding_vectors[0]))
        # It goes through this loop 11 times: 11x32

    return embedding_vectors

def confusion_similarity_matrix(embedding_vectors):
    cosine_similarity = metrics.CosineSimilarity()
    num_embeddings = int(len(embedding_vectors)/32)
    similarity_matrix = np.zeros((num_embeddings, num_embeddings))

    for i in range(num_embeddings):
        for j in range(num_embeddings):
            similarity = cosine_similarity(embedding_vectors[i], embedding_vectors[j])
            similarity_matrix[i, j] = similarity.numpy()


    return np.array(similarity_matrix)

# Returns the mean of the diagonal and the mean of the non-diagonal elements of a squared array.
def diagonal_non_diagonal_mean(array):

    diagonal_elements = np.diagonal(array) # Returns a list of the diagonal of an array

    diagonal_average = np.average(np.diagonal(array)) # Returns the mean of the elements of an array

    # Create a mask for the diagonal elements
    mask = np.eye(array.shape[0], dtype=int)
    weights = np.ones_like(array, dtype=float)
    weights[mask] = 0  # Set diagonal weights to 0

    # Compute the weighted average excluding the diagonal elements
    non_diagonal_elements_average = np.average(array, weights=weights)

    return diagonal_average, non_diagonal_elements_average

embedding_vectors_valdataset = embedding_vectors(val_dataset)
confusion_matrix = confusion_similarity_matrix(embedding_vectors_valdataset)
res_confusion_matrix = diagonal_non_diagonal_mean(confusion_matrix)
print(res_confusion_matrix)
print(confusion_matrix.shape)
exit()
#%% Training


siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.0001))
siamese_model.summary()
# If we want to troubleshoot problems, we might want to use smaller epochs and smaller batch sizes so that we can make sure that it is not overloading the system.

history = siamese_model.fit(train_dataset, epochs=10, validation_data=val_dataset, batch_size=1)

print(history.history.keys())

# Saving the model in my local directory
# savedir = Path("/mnt/c/Users/joshu/Desktop/TFG/DeepLearningCryo/Siamese_Network_Loss_Function/siamesetlktrained/")
# siamese_model.save(savedir)

# Saving the model weights in my local directory
# saveweightsdir = Path("/mnt/c/Users/joshu/Desktop/TFG/DeepLearningCryo/Siamese_Network_Loss_Function/siamesetlktrainedweights/")
# siamese_model.save_weights(saveweightsdir)


#%% Inspecting

sample = next(iter(train_dataset))
# visualize(*sample)


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
