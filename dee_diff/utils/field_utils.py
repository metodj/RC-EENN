import math
from scipy.stats import multivariate_normal
import numpy as np
import torch_geometric
from torch_geometric.nn import radius_graph
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.distributions as D
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F


def create_time_embedding(time_steps, num_datapoints, frequency=64, max_time=1.0):
    """
    Create time embeddings using Fourier features for time steps in range [0, max_time],
    and repeat these embeddings for each data point in a batch.

    :param time_steps: A tensor of time steps.
    :param num_datapoints: The number of data points for which to repeat the time embeddings.
    :param frequency: The number of frequencies to use for encoding.
    :param max_time: The maximum value for time steps.
    :return: A tensor containing the repeated time embeddings.
    """
    positions = time_steps.unsqueeze(1) / max_time
    div_term = torch.exp(torch.linspace(0, frequency - 1, frequency).float() * (-math.log(max_time) / frequency)).to(time_steps.device)
    sinusoids = torch.cat((torch.sin(positions * div_term), torch.cos(positions * div_term)), dim=1)

    # Repeat the time embeddings for each data point
    repeated_sinusoids = sinusoids.unsqueeze(1).repeat(1, num_datapoints, 1)
    return repeated_sinusoids

def create_space_embedding(coordinates, frequency=10):
    """
    Create space embeddings using Fourier features for a batch of 2D coordinates.

    :param coordinates: A tensor of coordinates of shape [batch, num_datapoints, 2].
    :param frequency: The number of frequencies to use for encoding.
    :return: A tensor containing the space embeddings for each batch.
    """
    batch_size, num_datapoints, dim = coordinates.shape
    # Flatten batch and num_datapoints dimensions
    flat_coordinates = coordinates.view(-1, dim)

    # Create div_term for the frequency encoding
    div_term = torch.exp(torch.linspace(0, frequency - 1, frequency).float() * (-math.log(2.0) / frequency)).to(coordinates.device)
    div_term = div_term.view(1, -1)

    # Apply the sinusoidal functions separately to x and y coordinates
    sinusoids_x = torch.cat((torch.sin(math.pi * flat_coordinates[:, 0:1] * div_term), torch.cos(math.pi * flat_coordinates[:, 0:1] * div_term)), dim=1)
    sinusoids_y = torch.cat((torch.sin(math.pi * flat_coordinates[:, 1:2] * div_term), torch.cos(math.pi * flat_coordinates[:, 1:2] * div_term)), dim=1)
    if dim == 3:
        sinusoids_z = torch.cat((torch.sin(math.pi * flat_coordinates[:, 2:3] * div_term), torch.cos(math.pi * flat_coordinates[:, 2:3] * div_term)), dim=1)
        sinusoids = torch.cat((sinusoids_x, sinusoids_y, sinusoids_z), dim=1)
    else:
        sinusoids = torch.cat((sinusoids_x, sinusoids_y), dim=1)

    # Reshape back to [batch, num_datapoints, frequency * 4]
    return sinusoids.view(batch_size, num_datapoints, -1)
