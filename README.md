# RC-EENN
Code for paper [Fast yet Safe: Early-Exiting with Risk Control](https://arxiv.org/abs/2405.20915)

## Main Dependencies
TODO

## Setup 
1. Clone or download this repo. `cd` yourself to it's root directory.
2. Create and activate python [conda](https://www.anaconda.com/) enviromnent: `conda create --name rc-eenn python=3.10`
3. Activate conda environment:  `conda activate rc-eenn`
4. Install dependencies, using `pip install -r requirements.txt`

## Code
Code for each experiment can be found in:
- image classification (ImageNet) --> `img_cls`
- semanting segmentation (CityScapes) --> `sem_seg`
- language modeling (SQuAD, CNN/DM) --> `calm`
- image generation with diffusion (CelebA, CIFAR) --> `dee_diff`
 
## Acknowledgements
The [Robert Bosch GmbH](https://www.bosch.com) is acknowledged for financial support.
