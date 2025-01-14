N = 100
list_images = get_random_image_from_folders(images_path, N)

image_avg = []
counter = 0 
for path in list_images:
    image = np.float64(preprocess_image(path).numpy())
    image_avg.append(image)

import matplotlib.pyplot as plt
import numpy as np

""" This function is used to compute the running average at
different iterations and plots it in a 2,2 plots."""
def compute_and_plot_average(images, step_checkpoint):
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