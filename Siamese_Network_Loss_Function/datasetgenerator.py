# Given the path of a folder containing the similar images, we create an object function that creates
# a the positive dastaset which  of the 3 necessary inputs for the Siamese Model Network using a Triplet Loss Function

import os
import tensorflow as tf
import random
from pathlib import Path

class TripleDatasetGenerator:
    def __init__(self, dataset_path):
        """
        Initializes the TripleDatasetGenerator class.
        
        Args:
            dataset_path (str or Path): Path to the directory containing dataset folders.
        """
        self.dataset_path = Path(dataset_path)
        self.folder_datasets = self._load_datasets_from_folders()

    def _load_datasets_from_folders(self):
        """
        Loads all datasets from folders into a dictionary.
        
        Returns:
            dict: A dictionary where keys are folder names and values are lists of file paths.
        """
        folder_datasets = {}
        for folder in self.dataset_path.iterdir():
            if folder.is_dir():  # Ensure it's a folder
                folder_datasets[folder.name] = sorted(
                    [str(folder / f) for f in os.listdir(folder) if not f.startswith(".")]
                )
        return folder_datasets

    def create_triplet_dataset(self, anchor_folder, positive_folder, negative_folder):
        """
        Creates a tf.data.Dataset containing triplets (anchor, positive, negative).
        
        Args:
            anchor_folder (str): Folder name for the anchor dataset.
            positive_folder (str): Folder name for the positive dataset.
            negative_folder (str): Folder name for the negative dataset.
        
        Returns:
            tf.data.Dataset: A TensorFlow dataset of triplets (anchor, positive, negative).
        """
        anchor_images = self.folder_datasets.get(anchor_folder)
        positive_images = self.folder_datasets.get(positive_folder)
        negative_images = self.folder_datasets.get(negative_folder)

        if not (anchor_images and positive_images and negative_images):
            raise ValueError("One or more specified folders do not exist in the dataset.")

        # Randomize positive and negative datasets to exclude corresponding anchor files
        positive_randomized = []
        negative_randomized = []

        for i, anchor in enumerate(anchor_images):
            # Randomize positive (excluding the corresponding anchor file)
            remaining_positives = positive_images[:i] + positive_images[i + 1 :]
            positive_randomized.append(random.choice(remaining_positives))

            # Randomize negative (choose randomly from the negative folder)
            negative_randomized.append(random.choice(negative_images))

        # Zip the triplets together
        triplets = list(zip(anchor_images, positive_randomized, negative_randomized))

        # Convert to tf.data.Dataset
        triplet_dataset = tf.data.Dataset.from_tensor_slices(triplets)
        return triplet_dataset
