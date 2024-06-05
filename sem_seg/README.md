## Semantic segmentation experiments
Image-level early-exiting with ADP-C model and risk control framework on Cityscapes and GTA5 datasets.

### Setup
- Download pretrained ADP-C models from the [original repo](https://github.com/liuzhuang13/anytime) and store them in the `./pretrained_models/` folder, or add symbolic links here. This includes both the base models as well as the early-exit extensions (see their instructions). Do not change any model names. 
- ADP-C models finetuned on GTA5 for 50 epochs can be downloaded [here](https://drive.google.com/drive/folders/1KKxYbpXGHhkAb7Zn1xC02fG-ZMIHTpsX?usp=sharing). Also store them in the same `./pretrained_models/` folder, or add symbolic links here. Do not change any model names. 
- Download [Cityscapes](https://www.cityscapes-dataset.com/) and [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) datasets from the respective websites and store them in the `./data/` folder, or add symbolic links here. For Cityscapes, adhere to the default folder structures. For GTA5, the data comes in 10 zips of 2500 samples. Unzip and use the provided script `./data/gta5/group.sh` to group the data and labels into single folders.
- The config files for the experiments are located in the [config](config) folder, under the respective dataset subfolders in the `cfg.yaml` file.  Default configs are specified in `config_rcp.py`, and new ones can be added here and optionally linked to command line args for overriding. The other config files are used by the ADP-C model internally and should be modified minimally and with caution (e.g., to change the pretrained model's path).
- Note: We use a subset of the original GTA5 validation set for calibration and test data, i.e., we use `val.lst` which is a subset of 1000 random samples from `allval.lst`. This is only relevant for the risk control experiments, not finetuning.

### Early-exiting with risk control
After setting the desired config settings (incl. specifying the risk control procedures, calibration/test splits, number of trials etc.), you can use the `sem_seg.py` script to run the experiments.  Depending on which settings are specified in the config file and which given via the command line, the run command will look similar to the following:

    python RC-EENN/sem_seg/sem_seg.py --config_file=cfg --config_dir=RC-EENN/sem_seg/config/cityscapes --exp_suffix=_img --run_pred --conf_measure=top1 --conf_measure_aggr=mean --risk=miscoverage --risk_conf=brier_pred --model=w48 --device=cuda

Visit the script or run `python RC-EENN/sem_seg/sem_seg.py -h` for more info on the available arguments.  Changing the path to the correct config dir will run experiments on the respective dataset.  All the results, incl. intermediate results and automatically generated risk/efficiency plots, are saved to a subfolder in the provided output directory. 

### Finetuning on GTA5
We provide pre-tuned models [here](https://drive.google.com/drive/folders/1KKxYbpXGHhkAb7Zn1xC02fG-ZMIHTpsX?usp=sharing). For own finetuning, the script can be found in `tools/train_ee.py` and is a modified version of the original training script adapted to the GTA5 dataset and our experiment setup. The script can be run with the following command (specifying the correct config file for ADP-C-W18 or ADP-C-W48 finetuning):

    python -m torch.distributed.launch RC-EENN/sem_seg/tools/train_ee.py --cfg "RC-EENN/sem_seg/config/gta5/w48_adpc_train" --local-rank=0

It will generate a separate folder in the root directory called `output_train_w18/48` where the finetuned models, intermediate checkpoints and other training details are saved. The finetuned models can then be used for the early-exiting experiments on the GTA5 dataset.

- The finetuning script is `finetune_gta5.py`.  It uses the same config files as the main experiments, but with the `finetune` flag set to True.  The script will automatically load the pretrained model and finetune it on the GTA5 dataset.  The finetuned model is saved in the output directory specified in the config file.  The finetuned model can then be used for the early-exiting experiments on the GTA5 dataset.

If you have any questions, please open an issue and tag @alextimans.