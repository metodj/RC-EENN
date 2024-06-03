import os
import argparse
import time

import torch
import torch.nn.functional as F
import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

import wandb
from networks.ViT import UViT, DeeDiff_UViT
from ddpm_core import NoiseScheduler
from utils.plot_utils import create_image_grid, show_image_grid
from utils.simple_utils import get_device, load_config

def parse_args():
    parser = argparse.ArgumentParser(description="Script for training a model with a YAML config")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    return parser.parse_args()

args = parse_args()
config = load_config(args.config)

########### Getting the model parameters ###########
model_dict = config['model_params']
model_name = model_dict.pop('name')
model_path = model_dict.pop('path')
NUM_TRAIN_TIMESTEPS = config['train_params']['num_train_timesteps']
USE_AMP = config['train_params']['use_amp']
AMP_DTYPE = getattr(torch, config['train_params']['amp_dtype'])  # Convert string to torch dtype
BATCH_SIZE = config['train_params']['batch_size']
NUM_EPOCHS = config['train_params']['num_epochs']
LR = config['train_params']['learning_rate']
NUM_WARMUP_STEPS = config['train_params']['num_warmup_steps']

########### Getting the dataset parameters ###########
dataset_name = config['dataset']['name']
mean = config['dataset']['mean']
std = config['dataset']['std']
original_resolution = config['dataset']['original_resolution']
target_resolution = config['dataset']['target_resolution']

########### Setting up the model directory ###########
USE_CLUSTER = True
if USE_CLUSTER:
    root = '/nvmestore/thadziv/data/'
else:
    root = os.path.join(os.getcwd(), 'data')
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir)
model_directory = os.path.join(base_dir, "models", f"{model_name}_{dataset_name}")
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
    print(f"Created directory {model_directory}")
current_time = time.strftime("%Y%m%d-%H%M%S")
########### Log hyperparameters ###########
device = get_device()
print(f'Using device: {device}')

########### Initialize the model ###########
if model_name == 'UViT':
    model = UViT(**model_dict)
elif model_name == 'DeeDiff_UViT':
    model = DeeDiff_UViT(**model_dict)

model.load_state_dict(torch.load(model_path))
model = model.to(device)

########### Load the dataset ###########
target_H, target_W = target_resolution[0], target_resolution[1]
std_tensor = torch.tensor(std).view(3, 1, 1).to(device)
mean_tensor = torch.tensor(mean).view(3, 1, 1).to(device)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std), transforms.Resize((target_H, target_W))])

########### Initialize the NoiseScheduler ###########
noise_scheduler = NoiseScheduler(beta_steps=NUM_TRAIN_TIMESTEPS)
noise_scheduler.set_device(device)
scaler = GradScaler()
num_params = sum(p.numel() for p in model.parameters())


def sample(model, lambda_threshold=0.1, batch_size=256, num_batches=8, lambda_label=''):
    H, W = target_resolution
    data_shape = (3, H, W)
    num_steps = NUM_TRAIN_TIMESTEPS
    num_samples = batch_size
    save_folder = 'Generated_samples/DeeDiff_UViT_CIFAR10'
    folder_name = f'CIFAR_10_{lambda_label}'
    save_dir = os.path.join(save_folder, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    for batch in range(num_batches):
        if lambda_threshold is not None:
            samples, exit_layer_list = noise_scheduler.early_exit_sample(model=model,
                                                                         num_steps=num_steps,
                                                                         data_shape=data_shape,
                                                                         num_samples=num_samples,
                                                                         lambda_threshold=lambda_threshold,
                                                                         model_type='DeeDiff_UViT')
        else:
            samples = noise_scheduler.sample(model=model,
                                             num_steps=num_steps,
                                             data_shape=data_shape,
                                             num_samples=num_samples,
                                             model_type='DeeDiff_UViT')

        # Normalize samples and save
        samples = (samples * std_tensor + mean_tensor).clamp(0, 1)
        for idx, sample in enumerate(samples):
            filename = os.path.join(save_dir, f'sample_{batch * batch_size + idx}.png')
            save_image(sample, filename)

        # Save exit layers data
        exit_layers_filename = os.path.join(save_dir, f'exit_layers_batch_{batch}.txt')
        with open(exit_layers_filename, 'w') as f:
            flattened_exit_layers = torch.cat(exit_layer_list).numpy().flatten()
            np.savetxt(f, flattened_exit_layers, fmt='%d')


########### Train the model ###########
if __name__ == '__main__':
    N = 10  # Define the number of steps for lambda
    lambdas = np.linspace(0, 1, N)

    for lam in lambdas:
        print(f'Sampling for lambda = {lam:.2f}')
        sample(model, lambda_threshold=lam, lambda_label=f'{lam:.2f}')
        print(f'Finished sampling for lambda = {lam:.2f}')