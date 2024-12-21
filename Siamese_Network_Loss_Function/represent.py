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
from keras.applications import resnet
import copy
from sklearn.decomposition import PCA



# Hides warning to rebuild TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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

def visualize_embeddings(embeddings, labels):
    """
    Visualizes the embeddings in 2D space using PCA.
    Args:
        embeddings: List of embedding vectors.
        labels: Corresponding labels for the embeddings.
    """
    embeddings = np.array(embeddings)  # Convert to numpy array if not already
    labels = np.array(labels)

    # Reduce dimensionality to 2D using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Plot the embeddings
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.7
    )
    plt.colorbar(scatter, label='Class Label')
    plt.title("2D Visualization of Embeddings")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.show()


# Extract embeddings sorted into different classes from val_dataset
def extract_embeddings(dataset):
    embeddings = []
    labels = []

    for samples in dataset:
        anchor, _, _, label = samples

        anchor_embeddings = embedding(resnet.preprocess_input(anchor))  # Obtain embeddings from your model
        embeddings.extend(anchor_embeddings.numpy())
        labels.extend(label.numpy())  # Collect labels for grouping

    
    # Visualize embeddings after extracting them
    visualize_embeddings(embeddings, labels)

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

    return np.array(embedding_vectors_result)


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
confusion_matrix = confusion_similarity_matrix(embedding_vectors_valdataset)
res_confusion_matrix = diagonal_non_diagonal_mean(confusion_matrix)
print(confusion_matrix)
print(res_confusion_matrix)
print(confusion_matrix.shape)

fig, ax = plt.subplots()
im = ax.imshow(confusion_matrix, cmap=plt.get_cmap('hot'))
fig.colorbar(im)
plt.show()


exit()

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