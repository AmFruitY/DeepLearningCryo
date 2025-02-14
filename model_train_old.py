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
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import cv2

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
    
# Paths! 
cache_dir = Path("/mnt/c/Users/joshu/Desktop/TFG/DeepLearningCryo/Siamese_Network_Loss_Function/data/")
images_path = [cache_dir / "clear1", cache_dir / "clear2", cache_dir / "clear3", cache_dir / "clear4"]
# images_path = [cache_dir / "noisy1", cache_dir / "noisy2", cache_dir / "noisy3", cache_dir / "noisy4"]

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
dataset since we want to use not only one class. It also includes a label variable that
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

visualize(*list(train_dataset.take(1).as_numpy_iterator())[0])

#%% Setting the embedding model

# Before anything to optimize the epoch number of training, we will be using the technique of Early-Stopping
# Since the number of epochs determine quite substancially the difference in magnitude of the confusion matrix

# Early Stopping to stop the training once it starts to overfit
callback = callbacks.EarlyStopping(
    monitor='val_loss',    # Metric to monitor
    patience=2,            # Number of epochs with no improvement before stopping
    mode='min',            # Lower 'val_loss' is better
    verbose=1              # Print a message when stopping
)

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
# If we want to troubleshoot problems, we might want to use smaller epochs and smaller batch sizes so that we can make sure that it is not overloading the system.

# train_triplets, labels = strain_dataset
history = siamese_model.fit(train_dataset, epochs=200, validation_data=val_dataset, batch_size=8, callbacks=[callback])

siamese_model.summary()  # Check the summary

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Build the model by calling it on dummy data
dummy_input = [tf.random.uniform((1, 128, 128, 3)) for _ in range(3)]  # Example for triplet inputs
siamese_model(dummy_input)


# Saving the embedding layer since this is the output we want
# embedding_h5 = Path("/mnt/c/Users/joshu/Desktop/TFG/DeepLearningCryo/Siamese_Network_Loss_Function/siamesetlktrained/new_embedding_clear.h5")
# embedding.save(embedding_h5, include_optimizer=False)
embedding_keras = Path("/mnt/c/Users/joshu/Desktop/TFG/DeepLearningCryo/Siamese_Network_Loss_Function/siamesetlktrained/default_model_clear.keras")
embedding.save(embedding_keras, include_optimizer=False)

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

    return np.array(embedding_vectors_result), np.array(embeddings), np.array(images)


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
fig.colorbar(im)
plt.show()

exit()

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





embedding_vectors_valdataset = extract_embeddings(val_dataset_conf)
optimize_k_means(embedding_vectors_valdataset[1], 8)
kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(embedding_vectors_valdataset[1])
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

    return images_result, min_length


def compute_and_plot_average(images):
    image_count = len(images)  # Automate step_checkpoint calculation
    step_checkpoint = image_count // 4  # Integer division for checkpoint steps

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
    fig, axes = plt.subplots(2, 2)  # Create 2x2 subplot grid
    axes = axes.flatten()
    for ax, avg, step in zip(axes, plots, checkpoints):
        ax.imshow(avg, cmap='gray')  # Ensure grayscale visualization
        ax.set_title(f"Iteration {step}")
        ax.axis("off")  # Turn off axes

    plt.tight_layout()
    plt.show()

    # Resize the final running average to 128x128
    resized_avg = cv2.resize(running_avg, (128, 128), interpolation=cv2.INTER_LINEAR)

    # Normalize the image to [0, 255] and convert to uint8
    normalized_avg = cv2.normalize(resized_avg, None, 0, 255, cv2.NORM_MINMAX)
    normalized_avg = normalized_avg.astype(np.uint8)

    # Save the resized and normalized running average as a PNG
    cv2.imwrite('running_avg_128x128_last_class.png', normalized_avg)

    return running_avg


# Results

# After passing through K-means
results = kmeans_images(images, klabels)[0]

for i in range(4):
    compute_and_plot_average(results[i])



# Average N-images

N = kmeans_images(images, klabels)[1]
list_images = get_random_image_from_folders(images_path, N)

rand_image_to_avg = []
counter = 0 
for path in list_images:
    image = np.float64(preprocess_image(path).numpy())
    rand_image_to_avg.append(image)

def compute_and_plot_average(images):
    image_count = len(images)  # Automate step_checkpoint calculation
    step_checkpoint = image_count // 4  # Integer division for checkpoint steps

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
    fig, axes = plt.subplots(2, 2)  # Create 2x2 subplot grid
    axes = axes.flatten()
    for ax, avg, step in zip(axes, plots, checkpoints):
        ax.imshow(avg, cmap='gray')  # Ensure grayscale visualization
        ax.set_title(f"Iteration {step}")
        ax.axis("off")  # Turn off axes

    plt.tight_layout()
    plt.show()

    # Resize the final running average to 128x128
    resized_avg = cv2.resize(running_avg, (128, 128), interpolation=cv2.INTER_LINEAR)

    # Normalize the image to [0, 255] and convert to uint8
    normalized_avg = cv2.normalize(resized_avg, None, 0, 255, cv2.NORM_MINMAX)
    normalized_avg = normalized_avg.astype(np.uint8)

    # Save the resized and normalized running average as a PNG
    cv2.imwrite('running_avg_128x128_random.png', normalized_avg)

    return running_avg

compute_and_plot_average(rand_image_to_avg)