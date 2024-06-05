import os
import random
import numpy
import torch
import logging


def set_wd(target_wd: str):
    curr_wd = os.getcwd()
    if curr_wd != target_wd:
        print(f"Changing wd: {curr_wd} -> {target_wd}")
        os.chdir(target_wd)
    else:
        print("Already in target wd.")


def set_seed(seed: int, logger, verbose: bool = True):
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # https://pytorch.org/docs/stable/backends.html
    torch.backends.cudnn.benchmark = True  # type: ignore # Randomness but better performance
    torch.backends.cudnn.deterministic = False  # type: ignore # Randomness but better performance
    if verbose:
        logger.info(f"Setting {seed=}.")


def set_device(device: str):
    print(f"Requested {device=}.")
    gpu_avail = torch.cuda.is_available()

    if (device == "cpu") and (not gpu_avail):
        pass
    elif (device == "cpu") and gpu_avail:
        print("'cpu' requested but 'cuda' is available.")
    elif (device == "cuda") and (not gpu_avail):
        print("'cuda' requested but not available, using 'cpu' instead.")
        device = "cpu"
    elif (device == "cuda") and gpu_avail:
        print(f"Using 'cuda', {torch.cuda.device_count()} devices available.")
        print(f"Using GPU {torch.cuda.get_device_name()}.")

    return device


def get_logger(dir_log, fname_log):
    loggy = logging.getLogger("loggy")
    loggy.setLevel(logging.DEBUG)
    loggy.propagate = False
    file_handler = logging.FileHandler(os.path.join(dir_log, fname_log))
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(name)s][%(asctime)s][%(filename)s|%(funcName)s|%(lineno)d]: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    # formatter = logging.Formatter("%(name)s|%(message)s")s
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    loggy.addHandler(file_handler)
    loggy.addHandler(console_handler)

    return loggy


def get_cityscapes_classes():
    # https://www.cityscapes-dataset.com/dataset-overview/
    # index of the class in the list == class id
    return [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]
