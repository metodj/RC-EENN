from yacs.config import CfgNode

########################
### BASE CONFIG FILE ###
########################

CFG = CfgNode()

CFG.PROJECT = CfgNode()
CFG.PROJECT.CODE_DIR = "sem_seg"
CFG.PROJECT.CONFIG_FILE = "cfg"
CFG.PROJECT.CONFIG_DIR = "sem_seg/config"
CFG.PROJECT.OUTPUT_DIR = "/media/atimans/hdd/output_rcp"
CFG.PROJECT.PLOT_DIR = "plots"
CFG.PROJECT.SEED = 666666

CFG.DATASET = CfgNode()
CFG.DATASET.DIR = "/media/atimans/hdd/datasets"
CFG.DATASET.NAME = "cityscapes"
CFG.DATASET.NUM_CLASSES = 19
CFG.DATASET.IGNORE_LABEL = 255
CFG.DATASET.CHANNELS = "RGB"
CFG.DATASET.CAL_LABELS_AVAIL = True

CFG.MODEL = CfgNode(new_allowed=True)
CFG.MODEL.DEVICE = "cpu"
CFG.MODEL.TYPE = "w18"  # w48

CFG.RISK_CONTROL = CfgNode(new_allowed=True)
CFG.RISK_CONTROL.DELTA = 0.1
CFG.RISK_CONTROL.DELTA_CONF = 0.1
CFG.RISK_CONTROL.CONF_MEASURE = "top1"
CFG.RISK_CONTROL.CONF_MEASURE_AGGR = "mean"
CFG.RISK_CONTROL.RISK = "miscoverage"
CFG.RISK_CONTROL.RISK_CONF = "abs_diff"
CFG.RISK_CONTROL.LAMBDA_START = 1.0
CFG.RISK_CONTROL.LAMBDA_END = 0.0
CFG.RISK_CONTROL.LAMBDA_STEP = 0.01
CFG.RISK_CONTROL.LAMBDA_SELECT = "min_lambda"
CFG.RISK_CONTROL.N_TRIALS = 100
CFG.RISK_CONTROL.N_CAL = 400
CFG.RISK_CONTROL.EPSILON_START = 0.0
CFG.RISK_CONTROL.EPSILON_END = 0.51
CFG.RISK_CONTROL.EPSILON_STEP = 0.01
CFG.RISK_CONTROL.PROCEDURE = ["naive", "crc", "ucb", "ltt"]
CFG.RISK_CONTROL.PERM = "random" 

CFG.FILE = CfgNode(new_allowed=True)
CFG.FILE.SAVE_LABELS = False
CFG.FILE.SAVE_PREDS = False
CFG.FILE.SAVE_CONF = False

CFG.EXPERIMENT = CfgNode(new_allowed=True)
CFG.EXPERIMENT.LOAD_DIR = "auto" # path to load files from
CFG.EXPERIMENT.DIR = "auto"  # subfolder in output_dir
CFG.EXPERIMENT.SUFFIX = ""  # suffix for output files in subfolder
CFG.EXPERIMENT.FULL_DIR = ""  # full path to output_dir/dataset/experiment_dir_suffix
CFG.EXPERIMENT.RUN_PRED = False
CFG.EXPERIMENT.PLOT = True


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    return CFG.clone()


def update_from_args(cfg, args):
    """Update config from CLI args.

    cfg: yacs.CfgNode object
    args: argparse.Namespace object
    """
    args_to_cfg = {
        "config_file": "PROJECT.CONFIG_FILE",
        "config_dir": "PROJECT.CONFIG_DIR",
        "load_dir": "EXPERIMENT.LOAD_DIR",
        "exp_dir": "EXPERIMENT.DIR",
        "exp_suffix": "EXPERIMENT.SUFFIX",
        "run_pred": "EXPERIMENT.RUN_PRED",
        "n_trials": "RISK_CONTROL.N_TRIALS",
        "n_cal": "RISK_CONTROL.N_CAL",
        "delta": "RISK_CONTROL.DELTA",
        "delta_conf": "RISK_CONTROL.DELTA_CONF",
        "conf_measure": "RISK_CONTROL.CONF_MEASURE",
        "conf_measure_aggr": "RISK_CONTROL.CONF_MEASURE_AGGR",
        "risk": "RISK_CONTROL.RISK",
        "risk_conf": "RISK_CONTROL.RISK_CONF",
        "model": "MODEL.TYPE",
        "device": "MODEL.DEVICE"
    }

    # Filter out entries with value None
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    # Bring into format for merge_from_list
    args_list = [
        item
        for sublist in [[args_to_cfg[k], v] for k, v in args_dict.items()]
        for item in sublist
    ]
    cfg.merge_from_list(args_list)
    return cfg, args_list
