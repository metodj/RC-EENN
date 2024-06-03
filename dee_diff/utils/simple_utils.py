import torch
import yaml

def get_device():
    if torch.cuda.is_available():
        # CUDA available, use GPU (NVIDIA)
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # MPS available, use GPU (Apple Silicon)
        return torch.device("mps")
    else:
        # Neither CUDA nor MPS available, use CPU
        return torch.device("cpu")

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
