import os
import argparse
import time

import torch
import torch.nn.functional as F
import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
NUM_TRAIN_TIMESTEPS = config['train_params']['num_train_timesteps']
USE_AMP = config['train_params']['use_amp']
AMP_DTYPE = config['train_params']['amp_dtype']
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
SAVE_ONLY_BEST = True
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
wandb.init(project=f'{model_name} {dataset_name}')
wandb.config.update(config)

########### Initialize the model ###########
if model_name == 'UViT':
    model = UViT(**model_dict)
elif model_name == 'DeeDiff_UViT':
    model = DeeDiff_UViT(**model_dict)

########### Load the dataset ###########
target_H, target_W = target_resolution[0], target_resolution[1]
std_tensor = torch.tensor(std).view(3, 1, 1).to(device)
mean_tensor = torch.tensor(mean).view(3, 1, 1).to(device)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std), transforms.Resize((target_H, target_W))])
if dataset_name == 'CIFAR10':
    train = datasets.CIFAR10(root=os.path.join(os.getcwd(), 'data'), train=True, download=True, transform=transform)
elif dataset_name == 'CelebA':
    train = datasets.CelebA(root=root, split='all', download=True, transform=transform)
trainloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

########### Initialize the optimizer and learning rate scheduler ###########
accelerator = Accelerator(mixed_precision=AMP_DTYPE)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
num_epochs = NUM_EPOCHS
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=NUM_WARMUP_STEPS,
    num_training_steps=(len(trainloader) * num_epochs))

model, optimizer, trainloader, lr_scheduler = accelerator.prepare(
    model, optimizer, trainloader, lr_scheduler)
devoce = accelerator.device
noise_scheduler = NoiseScheduler(beta_steps=NUM_TRAIN_TIMESTEPS)
noise_scheduler.set_device(device)
num_params = sum(p.numel() for p in model.parameters())
print(f'Number of parameters: {num_params}')
wandb.log({'Number of parameters': num_params})

########### Training loop ###########
def train(model, trainloader, optimizer, scheduler, epochs):
    model.train()
    best_loss = float('inf')
    for epoch in range(epochs):
        total_loss = 0
        # These losses are only for deediff
        u_loss = 0
        UAL_loss = 0
        simple_loss = 0
        for batch_idx, (data, _) in enumerate(tqdm.tqdm(trainloader)):
            data = data.to(device)
            bs = data.size(0)
            clean_images = data
            timesteps = torch.randint(
                0, NUM_TRAIN_TIMESTEPS, (bs,), device=clean_images.device
            ).long()
            timesteps_normalized = timesteps.float() / NUM_TRAIN_TIMESTEPS
            noise, noisy_images = noise_scheduler.add_noise(clean_images, timesteps)
            if model_name == 'UViT':
                predicted_noise = model(noisy_images, timesteps_normalized)
                loss = F.mse_loss(predicted_noise, noise)
            elif model_name == 'DeeDiff_UViT':
                predicted_noise, u_i, g_i = model(noisy_images, timesteps_normalized)
                L = g_i.size(1)  # Number of layers
                # The shape of g_i is (bs, L, 3, 32, 32), same for u_i
                # The shape of noise is (bs, 3, 32, 32)
                eps_t = noise.unsqueeze(1).repeat(1, g_i.size(1), 1, 1, 1)
                u_i_bar = F.tanh(torch.abs(g_i - eps_t))
                loss_simple = F.mse_loss(predicted_noise, noise)
                loss_u = F.mse_loss(u_i_bar.detach(), u_i) * L
                UAL = (1 - u_i.detach()) * (g_i - eps_t) ** 2
                loss_UAL = torch.mean(UAL) * L
                loss = loss_simple + loss_u + loss_UAL
            if model_name == 'DeeDiff_UViT':
                u_loss += loss_u.item()
                UAL_loss += loss_UAL.item()
                simple_loss += loss_simple.item()
            total_loss += loss.item()
            optimizer.zero_grad()
            accelerator.backward(loss)
            max_grad_norm = 1.0
            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()

        if model_name == 'UViT':
            loss_dict = {'Train loss': total_loss/(batch_idx+1)}
        elif model_name == 'DeeDiff_UViT':
            loss_dict = {'Train loss': total_loss/(batch_idx+1),
                         'Simple loss': simple_loss/(batch_idx+1),
                         'U loss': u_loss/(batch_idx+1),
                         'UAL loss': UAL_loss/(batch_idx+1)}

        print(f'Epoch {epoch}, Losses: {loss_dict}')
        if epoch % 5 == 0:
            H, W = target_resolution
            data_shape = (3, H, W)
            num_steps = NUM_TRAIN_TIMESTEPS
            num_samples = 16
            samples = noise_scheduler.sample(model, num_steps, data_shape, num_samples, model_type=model_name)
            for idx in range(len(samples)):
                samples[idx] = samples[idx] * std_tensor + mean_tensor
            grid = create_image_grid(samples, nrow=4)
            show_image_grid(grid)
            avg_train_loss = total_loss/(batch_idx+1)
            if SAVE_ONLY_BEST:
                if avg_train_loss < best_loss:
                    best_loss = avg_train_loss
                    torch.save(model.state_dict(), os.path.join(model_directory, f'{model_name}_{dataset_name}_{current_time}.pt'))
            else:
                torch.save(model.state_dict(), os.path.join(model_directory, f'{model_name}_{dataset_name}_{current_time}.pt'))
            wandb.log({'Samples': [wandb.Image(grid, caption=f'Epoch {epoch}')],
                          **loss_dict})
        else:
            wandb.log(loss_dict)

########### Train the model ###########
if __name__ == '__main__':
    train(model, trainloader, optimizer, noise_scheduler, epochs=num_epochs)