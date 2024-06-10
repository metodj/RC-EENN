# RC-EENN
This is the public code repository for our paper: [Fast yet Safe: Early-Exiting with Risk Control](https://arxiv.org/abs/2405.20915)


## Setup 
1. Clone or download this repo. `cd` yourself to it's root directory.
2. Create and activate python [conda](https://www.anaconda.com/) enviromnent: `conda create --name rc-eenn python=3.10`
3. Activate conda environment:  `conda activate rc-eenn`
4. Install dependencies, using `pip install -r requirements.txt`

TODO: add requirements for `dee_diff` and `sem_seg` experiments

## Code
Code for each experiment can be found in its respective subfolder:
- Image classification (ImageNet) --> `img_cls`
- Semantic segmentation (Cityscapes, GTA5) --> `sem_seg`
- Language modeling (SQuAD, CNN/DM) --> `calm`
- Image generation with early-exit diffusion (CelebA, CIFAR) --> `dee_diff`
 
## Acknowledgements
The [Robert Bosch GmbH](https://www.bosch.com) is acknowledged for financial support.

## License
TODO

## Citation
If you find this repository helpful, please consider citing:
```
@article{jazbec2024fast,
    title = {Fast yet Safe: Early-Exiting with Risk Control}, 
    author = {Metod Jazbec and Alexander Timans and Tin Hadži Veljković and Kaspar Sakmann and Dan Zhang and Christian A. Naesseth and Eric Nalisnick},
    journal = {Arxiv Preprint},
    year = {2024},
}
```