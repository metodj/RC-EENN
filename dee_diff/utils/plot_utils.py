import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

def create_image_grid(images, nrow=8):
    """
    Create a grid of images from a batch.

    Parameters:
    images (Tensor): Tensor of images in B, C, H, W format.
    nrow (int): Number of images in each row of the grid.

    Returns:
    grid (Tensor): A single image grid combining the input images.
    """
    # Use make_grid to create a grid of images
    grid = make_grid(images, nrow=nrow)
    return grid


def show_image_grid(grid):
    """
    Display an image grid using matplotlib for RGB images.

    Parameters:
    grid (Tensor): A single image grid combining the input images.
    """
    # Convert grid to numpy, normalize, and transpose axes for matplotlib
    np_grid = grid.cpu().numpy()
    np_grid = np_grid.transpose((1, 2, 0))  # Rearrange the axes for HWC format expected by imshow
    np_grid = np.clip(np_grid, 0, 1)  # Normalize to [0, 1] for displaying

    # Show image grid
    plt.figure(figsize=(10, 10))
    plt.imshow(np_grid)  # No need to specify 'cmap' for RGB images
    plt.axis('off')
    plt.show()

def unnormalize(field_values, mean, std):
    """
    Unnormalize the field values using the mean and standard deviation.
    """
    return field_values * std + mean

def show_images(images, num_rows=4, epoch='none', save=False, img_file=None):
    """
    Show a grid of images.
    """
    fig, axs = plt.subplots(nrows=num_rows, ncols=len(images) // num_rows, figsize=(12, 12))
    axs = axs.flatten()

    for img, ax in zip(images, axs):
        ax.imshow(img.squeeze(), cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    if save:
        plt.savefig(img_file)
##### Training Loop #####
# Example usage:
# Assuming 'sampled_images' is your tensor of MNIST images in B, 1, H, W format
# grid = create_image_grid(sampled_images, nrow=8)
# show_image_grid(grid)


def plot_heatmap_from_coordinates(coordinates, values, bins=64):
    # Extract x and y coordinates
    x = coordinates[:, 0]
    y = coordinates[:, 1]

    # Create a 2D histogram (heatmap)
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=values)

    # Return the heatmap plot
    return heatmap, xedges, yedges