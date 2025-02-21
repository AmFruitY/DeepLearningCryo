#%% Libraries
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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Classes
from SiameseModel import SiameseModel
from DistanceLayer import DistanceLayer

#%% Configuration
# Hides warning to rebuild TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#%% Loading the model
custom_objects = {'DistanceLayer': DistanceLayer, 'SiameseModel': SiameseModel} # Since we are using a model that uses custom class objects
                                                                                # We need to create this variable to load the model correctly - This is called a subclassed
embedding_dir = Path("/mnt/c/Users/joshu/Desktop/TFG/DeepLearningCryo/Siamese_Network_Loss_Function/siamesetlktrained/new_training_008_noisy.keras")
embedding = models.load_model(embedding_dir, custom_objects=custom_objects)

#%% Loading the dataset
target_shape = (128,128) # I don't think we need this.

# The same workflow as before
cache_dir = Path("/mnt/c/Users/joshu/Desktop/TFG/DeepLearningCryo/Siamese_Network_Loss_Function/data/")
# images_path = cache_dir / "EMD18199_clear"
images_path = cache_dir / "EMD18199_noisy"

save_path = Path("/mnt/c/Users/joshu/Desktop/TFG/DeepLearningCryo/Siamese_Network_Loss_Function/Visuals/") # We also want to analize how the model works with a different unseen dataset

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


#%% Analizing the results of infering the model
# The first step is to extract the embeddings from the dataset
def extract_embeddings(dataset):
    """ Given a dataset, extract embeddings using the embedding model. """

    images = []
    embeddings = []


    for samples in dataset:
        anchor = samples
        images.extend(anchor)
        anchor_embeddings = embedding(resnet.preprocess_input(anchor))  # Obtain embeddings from your model
        embeddings.extend(anchor_embeddings.numpy())
          # Collect labels for grouping

    return np.array(embeddings), np.array(images)

# The second step is to reduce the dimensionality of the embeddings to 2D using PCA and t-SNE
def dimension_reduction(embeddings):
    """ Given a set of embeddings, reduce their dimensionality to 2D using PCA and t-SNE. """

    embeddings = np.array(embeddings)  # Convert to numpy array if not already

    # Reduce dimensionality to 2D using PCA
    pca = PCA(n_components=50)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(reduced_embeddings)

    return reduced_embeddings, embeddings_2d

# The third step is to optimize the number of clusters using the elbow method
def optimize_k_means(data, max_k):
    """ Given a dataset, optimize the number of clusters by visualizing the elbow method. 
        Returns the optimized number of clusters but using the relative differences between the inertias. I still have to find a way to do it correctly.
    """
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
            print(optimized_k) # It is not perfect, I still have to find a way how to do it correctly.
            break


    # Generate the elbow plot
    fig = plt.subplots(figsize = (10, 5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Number of Clusters', fontsize = 25)
    plt.ylabel('Inertia', fontsize = 25)
    plt.grid(True)
    plt.show()

    return optimized_k

embedding_vectors_valdataset = extract_embeddings(dataset) # Extract embeddings from the dataset
embeddings = dimension_reduction(embedding_vectors_valdataset[0]) # Use the function dimension_reduction to reduce the dimensionality of the embeddings.

embeddings_no_labels = embeddings[1] # Save the embeddings without labels for visualization

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_no_labels[:, 0], embeddings_no_labels[:, 1], alpha=0.7)
plt.xlabel("t-SNE Dimension 1", fontsize = 25)
plt.ylabel("t-SNE Dimension 2", fontsize = 25)
dim_reduction_path = save_path / "dim_reduction_no_labels.png"
plt.savefig(dim_reduction_path)

k_cluster = optimize_k_means(embedding_vectors_valdataset[0], 20)
kmeans = KMeans(n_clusters=k_cluster, random_state=0, n_init="auto").fit(embedding_vectors_valdataset[0])
images = embedding_vectors_valdataset[1]
klabels = kmeans.labels_

def visualize_embeddings(embeddings, labels):
    """ Visualize the embeddings in 2D space with the labels given."""
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(embeddings[idx, 0], embeddings[idx, 1], label=f'Class {label+1}', alpha=0.7)

    plt.xlabel("t-SNE Dimension 1", fontsize = 25)
    plt.ylabel("t-SNE Dimension 2", fontsize = 25)
    plt.legend(fontsize = 16)
    dim_reduction_path = save_path / "dim_reduction.png"
    plt.savefig(dim_reduction_path)
    # plt.show()

visualize_embeddings(embeddings[1], klabels)


exit()
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




