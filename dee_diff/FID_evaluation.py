import os
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image

def save_cifar10_images(directory):
    # Ensure the output directory exists
    os.makedirs(directory, exist_ok=True)

    # Define the transformation: Convert PIL images to Tensors
    transform = transforms.ToTensor()

    # Load the CIFAR10 dataset
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Loop over the dataset and save each image
    for i, (image, label) in enumerate(dataset):
        # File path to save the image, e.g., 'original_data/0.png'
        filename = os.path.join(directory, f'{i}.png')
        # Save image; 'save_image' expects a batch dimension, so use 'unsqueeze(0)'
        save_image(image, filename)


if __name__ == '__main__':
    # Directory to save the images
    output_dir = 'Generated_samples/CIFAR10/original_data'
    save_cifar10_images(output_dir)
    print(f"All CIFAR10 images have been saved in '{output_dir}'.")
