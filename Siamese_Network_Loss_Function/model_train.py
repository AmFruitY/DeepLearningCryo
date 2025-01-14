import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import copy
import cv2
from pathlib import Path
from keras import applications
from keras import layers
from keras import losses
from keras import ops
from keras import optimizers
from keras import metrics
from keras import Model
from keras import callbacks
from keras.applications import resnet
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


# Classes
from SiameseModel import SiameseModel
from DistanceLayer import DistanceLayer

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

# The example target shape is 200, 200
# I will be using 100,100 for the cryogenic molecules
target_shape = (128, 128)
    
#%% Paths
cache_dir = Path("/mnt/c/Users/joshu/Desktop/TFG/DeepLearningCryo/Siamese_Network_Loss_Function/data/")
images_path = [cache_dir / "clear1", cache_dir / "clear2", cache_dir / "clear3", cache_dir / "clear4"]
# images_path = [cache_dir / "noisy1", cache_dir / "noisy2", cache_dir / "noisy3", cache_dir / "noisy4"]


save_path = Path("/mnt/c/Users/joshu/Desktop/TFG/DeepLearningCryo/Siamese_Network_Loss_Function/Visuals/")

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

def preprocess_triplets(anchor, positive, negative, labels):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    # The original function only had the first part of the tuple but since I want to know
    # The labels of each image, I had to modify it a little bit
    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative), labels, )

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


"""
This function is to automize the creation of dataset to be able to create a larger 
dataset since we want to use not only one class. It also includes a ground truth label variable that
makes sure we know what class they came from.
"""

def create_dataset(list_of_image_paths):

    anchor_images = []
    positive_images = []
    negative_images =[]
    labels = []

    for i, path in enumerate(list_of_image_paths):

        # Anchor and Positive dataset creation
        anchor_images_individual = sorted([str(path / f) for f in os.listdir(path)]) # Get sorted list of anchor images - The anchor 
        positive_images_indivual = get_randomized_images(anchor_images_individual) # Function to get a randomized version of the list such that no image is in the same index
        
        anchor_images.extend(anchor_images_individual)
        positive_images.extend(positive_images_indivual)

        image_count = len(anchor_images_individual)

        # Negative dataset creation
        other_paths = list_of_image_paths[:i] + list_of_image_paths[i+1:] # This excludes the current anchor path 
        negative_images_individual = get_random_image_from_folders(other_paths, image_count)
        negative_images.extend(negative_images_individual)

        # Labels
        class_label = [i] * image_count
        labels.extend(class_label)
    return anchor_images, positive_images, negative_images, labels

anchor_images, positive_images, negative_images, labels = create_dataset(images_path)

image_count = len(anchor_images)

# Create the dataset for TensorFlow - which are all paths to the said images
anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)
negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
# negative_dataset = negative_dataset.shuffle(buffer_size=4096)
label_dataset = tf.data.Dataset.from_tensor_slices(labels)


dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset, label_dataset))
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.map(preprocess_triplets) # Turn the paths into the images

# Let's now split our dataset in train and validation.
train_dataset = dataset.take(round(image_count * 0.8))
val_dataset = dataset.skip(round(image_count * 0.8))

# Saving this dataset to calculate the confusion matrix for later, since we want to know the labels
val_dataset_conf = dataset.skip(round(image_count * 0.8))

# Selecting only the first three elements of the tuple before passing batching - in other words ignoring the labels
train_dataset = train_dataset.map(lambda anchor, positive, negative, label: (anchor, positive, negative))
val_dataset = val_dataset.map(lambda anchor, positive, negative, label: (anchor, positive, negative))

# We will batch the data to make the training more efficient
train_dataset = train_dataset.batch(8, drop_remainder=False) # If TRUE the last batch is smaller than 32, then it drops it.
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE) # This just means that it prepares the later data for more efficient training

val_dataset = val_dataset.batch(8, drop_remainder=False)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset_conf = val_dataset_conf.batch(8, drop_remainder=False)
val_dataset_conf = val_dataset_conf.prefetch(tf.data.AUTOTUNE)


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

# Before anything to optimize the epoch number of training, we will be using the technique of Early-Stopping
# Since the number of epochs determine quite substancially the difference in magnitude of the confusion matrix

# Early Stopping to stop the training once it starts to overfit
callback = callbacks.EarlyStopping(
    monitor='val_loss',    # Metric to monitor
    patience=5,            # Number of epochs with no improvement before stopping
    mode='min',            # Lower 'val_loss' is better
    verbose=1              # Print a message when stopping
)

import keras

data_augmentation = keras.Sequential(
    [
        layers.GaussianNoise(0.1),  # Add Gaussian noise with standard deviation 0.1
        # layers.RandomTranslation(0.1, 0.1),  # Translate images by 20% horizontally and vertically
    ],
    name="data_augmentation",
)

base_cnn = resnet.ResNet50(
    weights="imagenet", input_shape=target_shape + (3,), include_top=False
)

# Augment the input data
input_layer = layers.Input(shape=target_shape + (3,))
augmented_input = data_augmentation(input_layer)

# Pass augmented data to the base CNN
cnn_output = base_cnn(augmented_input)

flatten = layers.Flatten()(cnn_output)
dense1 = layers.Dense(256, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(128, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(128)(dense2)

embedding = Model(input_layer, output, name="Embedding")

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
siamese_model.compile(optimizer=optimizers.Adam(0.00001))
# If we want to troubleshoot problems, we might want to use smaller epochs and smaller batch sizes so that we can make sure that it is not overloading the system.

# train_triplets, labels = strain_dataset
history = siamese_model.fit(train_dataset, epochs=200, validation_data=val_dataset, batch_size=8, callbacks=[callback])

siamese_model.summary()  # Check the summary

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss', fontsize = 20)
plt.xlabel('epoch', fontsize = 20)
plt.legend(['train', 'test'], loc='upper left', fontsize =  20)
training_history_path = save_path / "training_history.png"
plt.savefig(training_history_path)
# plt.show()

# Build the model by calling it on dummy data
dummy_input = [tf.random.uniform((1, 128, 128, 3)) for _ in range(3)]  # Example for triplet inputs
siamese_model(dummy_input)

# Saving the embedding layer since this is the output we want
embedding_keras = Path("/mnt/c/Users/joshu/Desktop/TFG/DeepLearningCryo/Siamese_Network_Loss_Function/siamesetlktrained/new_training_008_clear.keras")
embedding.save(embedding_keras, include_optimizer=False)

""" Extract embeddings sorted into different classes from val_dataset.

    input:
        dataset: It has to be a TensorFlow quadruplet (triplet with labels) dataset.
    output:
        embedding_vector_result: array The embeddings of the images in their respective classes
        embeddings: array The embeddings without putting classifying.
        images: array The images used in the dataset without classifying"""

def extract_embeddings(dataset):
    images = []
    embeddings = []
    labels = []

    for samples in dataset:
        anchor, _, _, label = samples
        images.extend(anchor)
        anchor_embeddings = embedding(resnet.preprocess_input(anchor))  # Obtain embeddings from your model
        embeddings.extend(anchor_embeddings.numpy())
        labels.extend(label.numpy())  # Collect labels for grouping

    # Organize embeddings by class
    class_embeddings = {}
    for embedding_ind, label in zip(embeddings, labels):
        if label not in class_embeddings:
            class_embeddings[label] = []
        class_embeddings[label].append(embedding_ind)

    # Create the resulting list
    embedding_vectors_result = []

    min_length = min(len(values) for values in class_embeddings.values())

    for key in sorted(class_embeddings.keys()):  # Ensure order of keys is consistent
        # Randomly select x elements from the list corresponding to the current key
        # This is important since the CosineSimilarity receives input in batches

        selected_arrays = random.sample(class_embeddings[key], min_length)
        embedding_vectors_result.append(selected_arrays)

    return np.array(embedding_vectors_result), np.array(embeddings), np.array(images), np.array(labels)

# Dimension reduction - PCA from 256 to 50 - t-SNE from 50 to 2 
"""

    output:
        reduced_embeddings: the samples with their reduced embeddings to 50 dimensions used for k-means clustering
        embeddings_2d: using the reduced embeddings, we further reduce to 2 dimensions using t-SNE for visualization"""

def dimension_reduction(embeddings):

    embeddings = np.array(embeddings)  # Convert to numpy array if not already

    # Reduce dimensionality to 2D using PCA
    pca = PCA(n_components=50)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(reduced_embeddings)

    return reduced_embeddings, embeddings_2d

def confusion_similarity_matrix(embedding_vectors):
    cosine_similarity = metrics.CosineSimilarity()
    num_embeddings = embedding_vectors.shape[0]
    similarity_matrix = np.zeros((num_embeddings, num_embeddings))

    for i in range(num_embeddings):
        for j in range(num_embeddings):
            similarity = cosine_similarity(tf.convert_to_tensor(embedding_vectors[i], dtype=tf.float64), tf.convert_to_tensor(embedding_vectors[j], dtype=tf.float64))
            similarity_matrix[i, j] = similarity.numpy()
            cosine_similarity.reset_state()


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

embedding_vectors_valdataset = extract_embeddings(val_dataset_conf)
# Not passing through k-means: Just a way to know that the model is correctly trained
confusion_matrix = confusion_similarity_matrix(embedding_vectors_valdataset[0])
res_confusion_matrix = diagonal_non_diagonal_mean(confusion_matrix)
fig, ax = plt.subplots()
im = ax.imshow(confusion_matrix, cmap=plt.get_cmap('hot'))
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
fig.colorbar(im)
conf_matrix_path = save_path / "conf_matrix.png"
fig.savefig(conf_matrix_path)
# plt.show()

embeddings = dimension_reduction(embedding_vectors_valdataset[1])

def visualize_embeddings(embeddings, labels):

    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(embeddings[idx, 0], embeddings[idx, 1], label=f'Class {label+1}', alpha=0.7)

    plt.xlabel("t-SNE Dimension 1", fontsize=25)
    plt.ylabel("t-SNE Dimension 2", fontsize=25)
    plt.legend(fontsize = 20)
    dim_reduction_path = save_path / "dim_reduction.png"
    plt.savefig(dim_reduction_path)
    # plt.show()

visualize_embeddings(embeddings[1], embedding_vectors_valdataset[3])

"""Input: Embeddings without classification,
        rel: given a relative range ratio, it picks the optimized k clusters to use for clustering
        
        It also outputs the elbow plot to visualize the optimal k value for kmeans clustering"""
def optimize_k_means(data, max_k):
    means = []
    inertias = []

    for k in range(1, max_k): # The max_k is not included
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)
    max_range_y = np.max(inertias)
    # Temporary k-chooser based on the range
    for i, element in enumerate(inertias):
        range_y = np.max(inertias[i:]) - np.min(inertias[i:])
        rel = range_y/max_range_y
        if rel < 0.012:
            optimized_k = i # I have already considered that the enumeration starts with index 0
             # It is not perfect, I still have to find a way how to do it correctly.
            break


    # Generate the elbow plot
    fig = plt.subplots(figsize = (10, 7))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Number of Clusters', fontsize = 20)
    plt.ylabel('Inertia', fontsize = 20)
    plt.grid(True)
    elbow_plot_path = save_path / "elbow_plot.png"
    plt.savefig(elbow_plot_path)
    # plt.show()

    return optimized_k


optimal_k = optimize_k_means(embeddings[1], 10)
kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto").fit(embedding_vectors_valdataset[1])
images = embedding_vectors_valdataset[2]
klabels = kmeans.labels_

def kmeans_images(images, klabels):
    classed_images = {}
    for image_ind, klabel in zip(images, klabels):
        if klabel not in classed_images:
            classed_images[klabel] = []
        classed_images[klabel].append(image_ind)

    # Create the resulting list
    images_result = []

    min_length = min(len(values) for values in classed_images.values())

    for key in sorted(classed_images.keys()):  # Ensure order of keys is consistent
        # Randomly select x elements from the list corresponding to the current key
        # This is important since the CosineSimilarity receives input in batches

        selected_arrays = random.sample(classed_images[key], min_length)
        images_result.append(selected_arrays)

    return np.array(images_result), min_length

def compute_average(list_of_images):
    """
    Compute the average image for each class.

    Parameters:
        list_of_images (list of lists): A list where each index corresponds to a list of images for a specific class.

    Returns:
        list: A list of average images, one for each class.
    """
    avg_list = []  # List to store the average image for each class

    for class_images in list_of_images:
        avg = np.stack(class_images, axis=0).mean(axis=0)
        avg_list.append(avg)
    return np.array(avg_list)


# Results

# After passing through K-means
kmeans_results = kmeans_images(images, klabels)[0]
average_kmeans = compute_average(kmeans_results)

# Average N-images

N = kmeans_images(images, klabels)[1]

random_images_to_avg = []

for class_folder in images_path:
    # Get all image paths in the current class folder
    class_images_path = [str(Path(class_folder) / f) for f in os.listdir(class_folder)]

    # Select N random images without replacement
    selected_paths = random.sample(class_images_path, min(N, len(class_images_path)))

    # Load and preprocess the selected images
    images = [np.float64(preprocess_image(path).numpy()) for path in selected_paths]

    random_images_to_avg.append(images)

average_random_images = compute_average(random_images_to_avg)

def save_images(images_list, filename):
    num_images_list = len(images_list)

    for i in range(num_images_list):
        avg_path = save_path / f"{filename}_{i+1}.png"
        plt.imsave(avg_path,images_list[i])

#%% Random N images average

average_random_per_class_images = []
for i in range(4):
    ims_paths = get_random_image_from_folders(images_path, N)
    ims = [preprocess_image(filename) for filename in ims_paths]
    average_random_class_images = np.stack(ims, axis=0).mean(axis=0)
    average_random_per_class_images.append(average_random_class_images)

save_images(average_kmeans, "kmeans_average")
save_images(average_random_images, "random_average")
save_images(average_random_per_class_images, "random_class_average")

exit()






