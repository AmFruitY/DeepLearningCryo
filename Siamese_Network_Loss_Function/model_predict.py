#%% Inspecting
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
from keras import callbacks
from keras import models
from keras.applications import resnet
import copy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Classes
from SiameseModel import SiameseModel
from DistanceLayer import DistanceLayer

# Hides warning to rebuild TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Load model

custom_objects = {'DistanceLayer': DistanceLayer, 'SiameseModel': SiameseModel}

embedding_dir = Path("/mnt/c/Users/joshu/Desktop/TFG/DeepLearningCryo/Siamese_Network_Loss_Function/siamesetlktrained/embedding_clear.keras")
embedding = models.load_model(embedding_dir, custom_objects=custom_objects)



# Create the dataset
target_shape = (128,128)

cache_dir = Path("/mnt/c/Users/joshu/Desktop/TFG/DeepLearningCryo/Siamese_Network_Loss_Function/data/")
images_path = cache_dir / "EMD18199_clear"
# images_path = cache_dir / "EMD18199_noisy"

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


images = sorted([str(images_path / f) for f in os.listdir(images_path)])
dataset = tf.data.Dataset.from_tensor_slices(images)
dataset = dataset.map(preprocess_image)
dataset = dataset.batch(8, drop_remainder=False)
dataset = dataset.prefetch(tf.data.AUTOTUNE)


# Results 
def visualize_embeddings(embeddings):
    """
    Visualizes the embeddings in 2D space using PCA.
    Args:
        embeddings: List of embedding vectors.
        labels: Corresponding labels for the embeddings.
    """
    embeddings = np.array(embeddings)  # Convert to numpy array if not already

    # Reduce dimensionality to 2D using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Plot the embeddings
    plt.figure(figsize=(10, 8))
    plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        cmap='tab10',
        alpha=0.7
    )
    plt.title("2D Visualization of Embeddings")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.show()

def extract_embeddings(dataset):
    images = []
    embeddings = []


    for samples in dataset:
        anchor = samples
        images.extend(anchor)
        anchor_embeddings = embedding(resnet.preprocess_input(anchor))  # Obtain embeddings from your model
        embeddings.extend(anchor_embeddings.numpy())
          # Collect labels for grouping

    
    # Visualize embeddings after extracting them
    visualize_embeddings(embeddings)

    return np.array(embeddings), np.array(images)


"""Input: Embeddings without classification"""
def optimize_k_means(data, max_k):
    means = []
    inertias = []

    for k in range(1, max_k): # The max_k is not included
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)
    
    print(inertias)
    max_range_y = np.max(inertias)
    # Temporary k-chooser based on the range
    for i, element in enumerate(inertias):
        range_y = np.max(inertias[i:]) - np.min(inertias[i:])
        rel = range_y/max_range_y
        print(rel)
        if rel < 0.012:
            optimized_k = i # I have already considered that the enumeration starts with index 0
            print(optimized_k) # It is not perfect, I still have to find a way how to do it correctly.
            break


    # derivative = finite_differences(means, inertias)
    # print(derivative)
    # Temporary k-chooser based on the way how the derivatives decline
    # for i, element in enumerate(derivative):
        # if element < 0.40:
            # optimized_k = i+1
            # print(optimized_k)
            # break

    # Generate the elbow plot
    fig = plt.subplots(figsize = (10, 5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()


embedding_vectors_valdataset = extract_embeddings(dataset)
optimize_k_means(embedding_vectors_valdataset[0], 20)
kmeans = KMeans(n_clusters=14, random_state=0, n_init="auto").fit(embedding_vectors_valdataset[0])
images = embedding_vectors_valdataset[1]
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

    return images_result

def compute_and_plot_average(images):
    image_count = len(images) # We will use this to automate the step_checkpoint and not rely on our own inputs
                                # Since we know that shuffling makes the number of training images random
    step_checkpoint = image_count // 4 # We want an integer intead of a float

    running_avg = np.zeros_like(images[0], dtype=np.float64)
    checkpoints = [] 
    plots = []  # List to store the running averages at checkpoints

    for i, img in enumerate(images, start=1):
        running_avg += (img - running_avg) / i
        
        # Save the current average at the checkpoints
        if i % step_checkpoint == 0:
            checkpoints.append(i)
            plots.append(running_avg.copy())  # Save a copy at this iteration

    # Plotting
    fig, axes = plt.subplots(2, 2)  # Create 1x4 subplot grid
    axes = axes.flatten()
    for ax, avg, step in zip(axes, plots, checkpoints):
        ax.imshow(avg)  # Convert to uint8 for visualization
        ax.set_title(f"Iteration {step}")
        ax.axis("off")  # Turn off axes

    plt.tight_layout()
    plt.show()
    return running_avg

results = kmeans_images(images, klabels)

for i in range(14):
    compute_and_plot_average(results[i])



