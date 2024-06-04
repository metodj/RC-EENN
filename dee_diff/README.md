## Diffusion model training & inference

### Training
Here you can find instructions on how to train diffusion models and the early-exit DeeDiff baseline models. Configuration files for the experiments are located in the [config](config) folder. To train the model, use the `ddpm_train.py` script with the following command:
    
    python ddpm_train.py --config config/<config_file>.yaml

In the `ddpm_core.py` script, we provide a custom implementation of a simple linear diffusion noise scheduler used in the experiments. The baseline model for both early-exit DeeDiff diffusion and regular diffusion is the UViT model. We modified the [original implementation](https://github.com/baofff/U-ViT) to be compatible with the early-exit diffusion models.

### Inference
Once a model is trained, you can use the `ddpm_EE_sample.py` script to generate samples for different threshold values, which can be selected in the script itself. To run the script for a model trained with a specific configuration file, use the following command:

    python ddpm_EE_sample.py --config config/<config_file>.yaml

If you have any questions, please open an issue and tag @stases.