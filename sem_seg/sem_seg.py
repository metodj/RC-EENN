import sys
import os

# add import module paths
code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # RC-EENN
sys.path.append(code_dir)
sys.path.append(os.path.join(code_dir, "sem_seg"))
sys.path.append(os.path.join(code_dir, "sem_seg", "lib"))

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config_rcp import get_cfg_defaults, update_from_args
import plot_util
from tools import io_file, util
import rcp

from lib.datasets.cityscapes import Cityscapes
from lib.datasets.gta5 import GTA5
from lib.models import model_anytime


class ImageRiskControl:

    def __init__(self, cfg, cfg_model, logger):
        self.cfg = cfg
        self.cfg_model = cfg_model
        self.logger = logger
        
        self.ignore_label = self.cfg.DATASET.IGNORE_LABEL
        self.device = self.cfg.MODEL.DEVICE

        self.conf_measure = self.cfg.RISK_CONTROL.CONF_MEASURE
        self.conf_measure_aggr = self.cfg.RISK_CONTROL.CONF_MEASURE_AGGR
        
        self.risk = self.cfg.RISK_CONTROL.RISK
        self.risk_conf = self.cfg.RISK_CONTROL.RISK_CONF
        self.delta = self.cfg.RISK_CONTROL.DELTA
        self.delta_conf = self.cfg.RISK_CONTROL.DELTA_CONF
        
        self.eps_start = self.cfg.RISK_CONTROL.EPSILON_START
        self.eps_end = self.cfg.RISK_CONTROL.EPSILON_END
        self.eps_step = self.cfg.RISK_CONTROL.EPSILON_STEP 
        
        self.lambda_start = self.cfg.RISK_CONTROL.LAMBDA_START
        self.lambda_end = self.cfg.RISK_CONTROL.LAMBDA_END
        self.lambda_step = self.cfg.RISK_CONTROL.LAMBDA_STEP
        self.lambda_select = self.cfg.RISK_CONTROL.LAMBDA_SELECT
        
        self.n_trials = self.cfg.RISK_CONTROL.N_TRIALS
        self.n_cal = self.cfg.RISK_CONTROL.N_CAL
        self.rcp_types = self.cfg.RISK_CONTROL.PROCEDURE
        self.rcp_perm = self.cfg.RISK_CONTROL.PERM

    def get_pred(self, model, dataloader, label_avail: bool = True):
        self.logger.info(
            f"""
            Running 'get_pred' | (conf, conf_aggr)=({self.conf_measure, self.conf_measure_aggr}) |  Label availability: {label_avail} | Dataset size: {len(dataloader)}.
            """
        )
        _labels, _preds, _conf, _conf_aggr, _dist_loss = [], [], [], [], []
        
        use_label = True if "gt" in self.risk_conf else False # for dist loss
        self.logger.info(f"Setting {use_label=} for confidence risk.")

        with torch.no_grad(), tqdm(dataloader, desc="Images") as loader:
            for batch in loader:
                
                if label_avail:
                    image, label, _, _ = batch
                    label = label.squeeze(0).to(self.device) # (H, W)
                else:
                    image, _, _ = batch
                    label = None
                
                preds = model(image.to(self.device))  # L * (C, H_downsamp, W_downsamp)

                size = image.size()  # (1, 3, H, W)
                preds_l, conf_l, conf_aggr_l, dist_loss_l = [], [], [], []
                
                # last exit pred dist for dist loss
                if preds[-1].size()[-2] != size[-2] or preds[-1].size()[-1] != size[-1]:
                    pred_L = F.interpolate(preds[-1], (size[-2], size[-1]), mode="bilinear")  # (1, C, H, W)
                pred_L = F.softmax(pred_L.squeeze(0).movedim(0, -1), dim=-1) # (H, W, C)
                
                # other exits
                for i, pred in enumerate(preds[:-1]):
                    if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                        pred = F.interpolate(pred, (size[-2], size[-1]), mode="bilinear")  # (1, C, H, W)
                    pred = F.softmax(pred.squeeze(0).movedim(0, -1), dim=-1) # (H, W, C)

                    # image-level confidence measure
                    conf = rcp.conf_measure(pred, self.conf_measure)  # (H, W)
                    conf_aggr = rcp.conf_measure_aggr(conf, self.conf_measure_aggr, self.device)  # (1,)
                    
                    # predictive distribution loss calculation
                    if i == 0: # collect last exit once
                        dist_loss, dist_loss_L = rcp.brier_loss(pred, pred_L, label, use_label, self.ignore_label, keepdim=False) # (1,)
                    else:
                        dist_loss, _ = rcp.brier_loss(pred, pred_L, label, use_label, self.ignore_label, keepdim=False) # (1,)
                                            
                    preds_l.append(torch.argmax(pred, dim=-1).to(torch.int16).detach().to("cpu"))  # (L-1) * (H, W)
                    conf_l.append(conf.to(torch.float16).detach().to("cpu"))  # (L-1) * (H, W)
                    conf_aggr_l.append(conf_aggr.to(torch.float16).detach().to("cpu"))  # (L-1)
                    dist_loss_l.append(dist_loss.to(torch.float16).detach().to("cpu"))  # (L-1)
                    
                    del pred, conf
                
                # add last exit
                conf_L = rcp.conf_measure(pred_L, self.conf_measure)  # (H, W)
                conf_aggr_L = rcp.conf_measure_aggr(conf_L, self.conf_measure_aggr, self.device)  # (1,)
                
                preds_l.append(torch.argmax(pred_L, dim=-1).to(torch.int16).detach().to("cpu"))  # L * (H, W)
                conf_l.append(conf_L.to(torch.float16).detach().to("cpu"))  # L * (H, W)
                conf_aggr_l.append(conf_aggr_L.to(torch.float16).detach().to("cpu"))  # L
                dist_loss_l.append(dist_loss_L.to(torch.float16).detach().to("cpu"))  # L 
                    
                # collect per sample
                _preds.append(torch.stack(preds_l, dim=0)) # N * (L, H, W)
                _conf.append(torch.stack(conf_l, dim=0)) # N * (L, H, W)
                _conf_aggr.append(torch.stack(conf_aggr_l, dim=0)) # N * (L,)
                _dist_loss.append(torch.stack(dist_loss_l, dim=0)) # N * (L,)
                if label_avail:
                    _labels.append(label.to(torch.int16).detach().to("cpu")) # N * (H, W)
                
                del image, label, preds, pred_L, conf_L, preds_l, conf_l

        """
        labels: int16, (N, H, W) or None
        preds: int16, (N, L, H, W)
        conf: float16, (N, L, H, W)
        conf_aggr: float16, (N, L)
        dist_loss: float16, (N, L)
        """
        labels = torch.stack(_labels, dim=0) if label_avail else None
        return labels, torch.stack(_preds, dim=0), torch.stack(_conf, dim=0), torch.stack(_conf_aggr, dim=0), torch.stack(_dist_loss, dim=0)

    def get_losses(self, lambdas, labels, preds, conf, conf_aggr, dist_loss, is_conf: bool = False):
        self.logger.info(
            f"""
            Running 'get_losses' | (conf, conf_aggr)=({self.conf_measure, self.conf_measure_aggr}) | (risk, risk_conf)={self.risk, self.risk_conf} | {is_conf=}.
            """
        )
        N, L, H, W = preds.size()
        assert conf.size() == (N, L, H, W), f"{conf.size()=}"
        assert conf_aggr.size() == (N, L), f"{conf_aggr.size()=}"
        assert dist_loss.size() == (N, L), f"{dist_loss.size()=}"
        
        N_lam = len(lambdas)
        lam_losses = torch.zeros((N_lam, N)) 
        lam_exits = torch.zeros((N_lam, N))
        
        pred_full = preds[:, -1]  # (N, H, W)
        dist_loss_full = dist_loss[:, -1].to(torch.float32) # (N,)

        with tqdm(lambdas, desc="Lambdas") as loader:
            for i, lam in enumerate(loader):

                # Get image-level exit from confidence
                mask = (conf_aggr > lam).int()  # (N, L)
                exits = mask.argmax(dim=-1)  # (N,)
                rows_with_no_threshold = torch.sum(mask, dim=-1) == 0
                exits[rows_with_no_threshold] = L - 1

                # Get early exit 
                pred_early = preds[torch.arange(N), exits]  # (N, H, W)
                dist_loss_early = dist_loss[torch.arange(N), exits].to(torch.float32) # (N,)
                exits += 1

                # Get losses
                if is_conf:
                    _, loss = rcp.compute_risk_dist(dist_loss_early, dist_loss_full, criterion=self.risk_conf)
                else:
                    _, loss = rcp.compute_risk(pred_early, pred_full, criterion=self.risk, gt_labels=labels)
                
                lam_losses[i, :] = loss
                lam_exits[i, :] = exits

                del mask, exits, pred_early, dist_loss_early
        del pred_full, dist_loss_full
        
        return lam_losses, lam_exits

    def get_permuted_losses(self, exits, labels, preds, dist_loss, is_conf: bool = False):
        self.logger.info(f"Running 'get_permuted_losses' | {self.rcp_perm=} | {is_conf=}.")
        
        N, L, H, W = preds.size()
        N_lam = len(exits)
        assert exits.size() == (N_lam, N), f"{exits.size()=}"
        assert dist_loss.size() == (N, L), f"{dist_loss.size()=}"
        
        lam_losses = torch.zeros((N_lam, N)) 
        lam_exits = torch.zeros((N_lam, N))
        
        pred_full = preds[:, -1]  # (N, H, W)
        dist_loss_full = dist_loss[:, -1].to(torch.float32) # (N,)
        
        # For each row, permute exits by some way and get corresponding losses (baseline options)
        for i in tqdm(range(N_lam), desc="Lambdas"):
            
            if self.rcp_perm == "perm":  # Exit permutation
                exits_perm = (exits[i][torch.randperm(N)] - 1).to(torch.int)
            elif self.rcp_perm == "random":  # Random exit
                exits_perm = torch.randint(0, 4, (N,))
            elif self.rcp_perm == "const":  # Constant exit
                CONST_EXIT = 1
                exits_perm = torch.full((N,), CONST_EXIT - 1)

            # Get early exit
            pred_early = preds[torch.arange(N), exits_perm]  # (N, H, W)
            dist_loss_early = dist_loss[torch.arange(N), exits_perm].to(torch.float32) # (N,)
            exits_perm += 1
            
            # Get losses
            if is_conf:
                _, loss = rcp.compute_risk_dist(dist_loss_early, dist_loss_full, criterion=self.risk_conf)
            else:
                _, loss = rcp.compute_risk(pred_early, pred_full, criterion=self.risk, gt_labels=labels)
            
            lam_losses[i, :] = loss
            lam_exits[i, :] = exits_perm

            del exits_perm, pred_early, dist_loss_early
        del pred_full, dist_loss_full
    
        return lam_losses, lam_exits

    def rcp(self, lambdas, eps_grid, losses, exits, losses_perm, exits_perm, is_conf: bool = False):
        
        N_lam, N = losses.size()
        assert exits.size() == (N_lam, N), f"{exits.size()=}"
        assert losses_perm.size() == (N_lam, N), f"{losses_perm.size()=}"
        assert exits_perm.size() == (N_lam, N), f"{exits_perm.size()=}"
        
        delta = self.delta_conf if is_conf is True else self.delta
        loss_B = 2.0 if is_conf is True else 1.0 # loss upper bound B
        self.logger.info(
            f"""
            Running rcp | rcp_types: {self.rcp_types} | (n_trials, n_cal) = {self.n_trials, self.n_cal} | {is_conf=} | {delta=}.\n
            Eps grid: {eps_grid}
            """
        )
        
        # For data collection; shape {rcp: [eps_grid] * trials}
        rc_lambdas = {r: [] for r in self.rcp_types}
        test_risk = {r: [] for r in self.rcp_types}
        test_exit = {r: [] for r in self.rcp_types}
        
        if "crc" in self.rcp_types:
            test_risk["crc_perm"], test_exit["crc_perm"] = [], []
        if "ucb" in self.rcp_types:
            test_risk["ucb_perm"], test_exit["ucb_perm"] = [], []
        if "ltt" in self.rcp_types:
            test_risk["ltt_perm"], test_exit["ltt_perm"] = [], []
        
        # Run risk-controlling procedure over trials
        for tr in tqdm(range(self.n_trials), desc="Trials"):
            
            data_idx = torch.randperm(N) # random data split
            # data_idx = torch.arange(N) # deterministic split
            cal_idx, test_idx = data_idx[:self.n_cal], data_idx[self.n_cal:]
            cal_losses, test_losses = losses[:, cal_idx], losses[:, test_idx]
            _, test_exits = exits[:, cal_idx], exits[:, test_idx]
            _, test_losses_perm = losses_perm[:, cal_idx], losses_perm[:, test_idx]
            _, test_exits_perm = exits_perm[:, cal_idx], exits_perm[:, test_idx]
            
            # STAGE 1: find \hat{\lambda} on calibration data
            lam_id_eps = {r: [] for r in self.rcp_types}
            for rc in self.rcp_types:
                for eps in eps_grid:
                    if rc == "naive":
                        lam_id, _ = rcp.get_naive_lambda(cal_losses, eps)
                    elif rc == "crc":
                        lam_id, _ = rcp.get_crc_lambda(cal_losses, eps, loss_B)
                    elif rc == "ucb":
                        lam_id, _, _ = rcp.get_ucb_lambda(cal_losses, eps, delta, loss_B)
                    elif rc == "ltt":
                        lam_id, _, _ = rcp.get_ltt_lambda(cal_losses, eps, delta, loss_B)
                    lam_id_eps[rc].append(lam_id)
                rc_lambdas[rc].append(lambdas[lam_id_eps[rc]].tolist())

            # STAGE 2: use \hat{\lambda} to compute test risk and exits
            for rc in list(test_risk.keys()):
                test_risk_eps, test_exit_eps = [], []
                for e, eps in enumerate(eps_grid):
                    if rc in ["crc_perm", "ucb_perm", "ltt_perm"]:
                        lam_id = lam_id_eps[rc[:3]][e]
                        test_risk_eps.append(test_losses_perm[lam_id].mean().item())
                        test_exit_eps.append(test_exits_perm[lam_id].mean().item())
                    else:
                        lam_id = lam_id_eps[rc][e]
                        test_risk_eps.append(test_losses[lam_id].mean().item())
                        test_exit_eps.append(test_exits[lam_id].mean().item())
                test_risk[rc].append(test_risk_eps)
                test_exit[rc].append(test_exit_eps)
        
            del cal_losses, test_losses, test_exits, test_losses_perm, test_exits_perm
        return test_risk, test_exit, rc_lambdas
    

def create_parser():
    """
    hierarchy: CLI > cfg > cfg default
    """
    parser = argparse.ArgumentParser(
        description="Parser for CLI arguments to run model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        required=False,
        help="Config file name to get settings to use for current run.",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default=None,
        required=False,
        help="Path to config file to use for current run.",
    )
    parser.add_argument(
        "--load_dir",
        type=str,
        default=None,
        required=False,
        help="Path to load files from for current run.",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        required=False,
        help="Path to experiment folder to use for current run.",
    )
    parser.add_argument(
        "--exp_suffix",
        type=str,
        default=None,
        required=False,
        help="Experiment folder suffix to use for current run.",
    )
    parser.add_argument(
        "--run_pred",
        action=argparse.BooleanOptionalAction,
        default=None,
        required=False,
        help="If run pred loop to compute losses and exits (bool).",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=None,
        required=False,
        help="Nr of experiment trials (cal/test splits).",
    )
    parser.add_argument(
        "--n_cal",
        type=int,
        default=None,
        required=False,
        help="Nr of calibration samples (per trial).",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        required=False,
        help="Risk bound probability (1 - delta).",
    )
    parser.add_argument(
        "--delta_conf",
        type=float,
        default=None,
        required=False,
        help="Risk bound probability (1 - delta).",
    )
    parser.add_argument(
        "--conf_measure",
        type=str,
        default=None,
        required=False,
        choices=["top1", "topdiff", "entropy"],
        help="Type of pixel-level confidence measure to compute from class prob.",
    )
    parser.add_argument(
        "--conf_measure_aggr",
        type=str,
        default=None,
        required=False,
        choices=["mean", "median", "quantile", "patch"],
        help="Type of aggregation to go from pixel- to image-level confidence measure.",
    )
    parser.add_argument(
        "--risk",
        type=str,
        default=None,
        required=False,
        choices=["miscoverage", "miou"],
        help="Desired loss function to compute associated risks.",
    )
    parser.add_argument(
        "--risk_conf",
        type=str,
        default=None,
        required=False,
        choices=["brier_pred", "brier_gt"],
        help="Desired loss function to compute associated risks for confidences.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        required=False,
        choices=["w18", "w48"],
        help="Which segmentation model to evalute (small: w18, large: w48).",
    )    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        required=False,
        choices=["cpu", "cuda"],
        help="Device to run code on.",
    )
    return parser


def set_dirs(cfg):
    # Determine experiment name
    if cfg.EXPERIMENT.DIR == "auto":
        cfg.EXPERIMENT.DIR = f"rcp_{cfg.MODEL.TYPE}_{cfg.RISK_CONTROL.CONF_MEASURE}_{cfg.RISK_CONTROL.CONF_MEASURE_AGGR}_{cfg.RISK_CONTROL.RISK}_{cfg.RISK_CONTROL.RISK_CONF}"
    # Create experiment dir
    full_dir = os.path.join(
        cfg.PROJECT.OUTPUT_DIR,
        cfg.DATASET.NAME,
        f"{cfg.EXPERIMENT.DIR}{cfg.EXPERIMENT.SUFFIX}",
    )
    Path(full_dir).mkdir(exist_ok=True, parents=True)
    # Create plot subdir
    if cfg.EXPERIMENT.PLOT:
        plot_dir = os.path.join(full_dir, cfg.PROJECT.PLOT_DIR)
        Path(plot_dir).mkdir(exist_ok=True, parents=True)
    # Set loading dir
    if cfg.EXPERIMENT.LOAD_DIR == "auto":
        cfg.EXPERIMENT.LOAD_DIR = full_dir
    else:
        cfg.EXPERIMENT.LOAD_DIR = os.path.join(
            cfg.PROJECT.OUTPUT_DIR,
            cfg.DATASET.NAME,
            cfg.EXPERIMENT.LOAD_DIR,
        )
        
    return cfg, full_dir, plot_dir


def load_model(cfg, cfg_model, logger):
    logger.info(f"Loading model from {cfg_model.TEST.MODEL_FILE}.")
    model = model_anytime.get_seg_model(cfg_model)
    pretrained_dict = torch.load(cfg_model.TEST.MODEL_FILE)
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(cfg.MODEL.DEVICE)
    model.eval()
    
    return model


def get_data(cfg, cfg_model, logger):
    logger.info(f"Loading dataset for {cfg.DATASET.NAME} from {cfg.DATASET.DIR}.")
    if cfg.DATASET.NAME == "gta5":
        data = GTA5(
            device=cfg.MODEL.DEVICE,
            root=cfg_model.DATASET.ROOT,
            list_path=cfg_model.DATASET.CAL_SET,
            num_classes=cfg.DATASET.NUM_CLASSES,
            multi_scale=False,
            flip=False,
            ignore_label=cfg.DATASET.IGNORE_LABEL,
            base_size=cfg_model.TEST.BASE_SIZE,
            crop_size=(cfg_model.TEST.IMAGE_SIZE[1], cfg_model.TEST.IMAGE_SIZE[0]),
            downsample_rate=1,
        )
    elif cfg.DATASET.NAME == "cityscapes":
        data = Cityscapes(
            device=cfg.MODEL.DEVICE,
            root=cfg_model.DATASET.ROOT,
            list_path=cfg_model.DATASET.CAL_SET,
            num_classes=cfg.DATASET.NUM_CLASSES,
            multi_scale=False,
            flip=False,
            ignore_label=cfg.DATASET.IGNORE_LABEL,
            base_size=cfg_model.TEST.BASE_SIZE,
            crop_size=(cfg_model.TEST.IMAGE_SIZE[1], cfg_model.TEST.IMAGE_SIZE[0]),
            downsample_rate=1,
        )
    else:
        raise ValueError("Invalid dataset.")
    
    return DataLoader(data, batch_size=1, shuffle=False, num_workers=cfg_model.WORKERS, pin_memory=True)


def get_mean_and_std(data):
    """
    Calculate mean and std values per method from a dictionary of nested lists.
    {rcp: [eps_grid] * trials} -> {rcp: [eps_grid with mean/std over trials]}
    """
    results = {}
    for method, trials in data.items():
        tensor_trials = torch.tensor(trials)
        mean_values = tensor_trials.mean(dim=0)
        std_values = tensor_trials.std(dim=0)
        results[method] = {'mean': mean_values, 'std': std_values}
    return results


def main():
    parser = create_parser()
    args = parser.parse_args()

    # cfg default -> override opts with cfg exp -> override opts with CLI -> device check
    cfg = get_cfg_defaults()
    cfg_exp_file = cfg.PROJECT.CONFIG_FILE if args.config_file is None else args.config_file
    cfg_exp_dir = cfg.PROJECT.CONFIG_DIR if args.config_dir is None else args.config_dir
    cfg_exp = io_file.load_yaml(cfg_exp_file, cfg_exp_dir, to_yacs=True)
    cfg.merge_from_other_cfg(cfg_exp)  # override cfg with cfg_exp
    cfg, _ = update_from_args(cfg, args)  # override cfg with args
    cfg.MODEL.DEVICE = util.set_device(cfg.MODEL.DEVICE)
    # ADP-C model configs
    cfg_model = io_file.load_yaml(f"{cfg.MODEL.TYPE}_adpc", cfg.PROJECT.CONFIG_DIR, to_yacs=True)

    # Set folder dirs and freeze
    cfg, full_dir, plot_dir = set_dirs(cfg)
    cfg.EXPERIMENT.FULL_DIR = full_dir
    cfg_model.OUTPUT_DIR = full_dir
    cfg.freeze()
    cfg_model.freeze()

    # Set up logger & seed
    logger = util.get_logger(cfg.EXPERIMENT.FULL_DIR, "log.txt")
    util.set_seed(cfg.PROJECT.SEED, logger)
    
    # EXPERIMENT START ########################
    
    logger.info("===== EXPERIMENT START =====")
    logger.info(f"Using config file '{cfg.PROJECT.CONFIG_FILE}'.")
    logger.info(f"Saving experiment files to '{cfg.EXPERIMENT.FULL_DIR}'.")
    logger.info(f"Loading experiment files from '{cfg.EXPERIMENT.LOAD_DIR}'.")

    # Init risk control object
    risk_control = ImageRiskControl(cfg, cfg_model, logger)

    # Collect required files for RCP
    if cfg.EXPERIMENT.RUN_PRED:
        model = load_model(cfg, cfg_model, logger)
        dataloader = get_data(cfg, cfg_model, logger)
        
        # Collect preds and confidences and optionally save
        logger.info("Computing predictions and confidences for loaded data.")
        labels, preds, conf, conf_aggr, dist_loss = risk_control.get_pred(
            model, dataloader, label_avail=cfg.DATASET.CAL_LABELS_AVAIL
        )
        del model, dataloader
        logger.info("Saving labels, preds and confidences if requested.")
        if cfg.DATASET.CAL_LABELS_AVAIL and cfg.FILE.SAVE_LABELS:
            io_file.save_tensor(labels, "labels", cfg.EXPERIMENT.FULL_DIR)
        if cfg.FILE.SAVE_PREDS:
            io_file.save_tensor(preds, "preds", cfg.EXPERIMENT.FULL_DIR)
        if cfg.FILE.SAVE_CONF:
            io_file.save_tensor(conf, "conf", cfg.EXPERIMENT.FULL_DIR)
        # Always save since small file sizes
        io_file.save_tensor(conf_aggr, "conf_aggr", cfg.EXPERIMENT.FULL_DIR)
        io_file.save_tensor(dist_loss, "dist_loss", cfg.EXPERIMENT.FULL_DIR)

        # Collect losses and exits for lambda grid
        logger.info("Computing losses and exits for loaded data.")
        lambdas = rcp.lambda_grid(risk_control.lambda_end, risk_control.lambda_start, risk_control.lambda_step)
        logger.info(f"Lambda grid: {lambdas}")
        loss_pred, exit_pred = risk_control.get_losses(lambdas, labels, preds, conf, conf_aggr, dist_loss, is_conf=False)
        loss_conf, exit_conf = risk_control.get_losses(lambdas, labels, preds, conf, conf_aggr, dist_loss, is_conf=True)
        io_file.save_tensor(loss_pred, "loss_pred", cfg.EXPERIMENT.FULL_DIR)
        io_file.save_tensor(exit_pred, "exit_pred", cfg.EXPERIMENT.FULL_DIR)
        io_file.save_tensor(loss_conf, "loss_conf", cfg.EXPERIMENT.FULL_DIR)
        io_file.save_tensor(exit_conf, "exit_conf", cfg.EXPERIMENT.FULL_DIR)
        
        # Collect permuted losses and exits (baseline options)
        loss_pred_perm, exit_pred_perm = risk_control.get_permuted_losses(exit_pred, labels, preds, dist_loss, is_conf=False)
        loss_conf_perm, exit_conf_perm = risk_control.get_permuted_losses(exit_conf, labels, preds, dist_loss, is_conf=True)
        io_file.save_tensor(loss_pred_perm, "loss_pred_perm", cfg.EXPERIMENT.FULL_DIR)
        io_file.save_tensor(exit_pred_perm, "exit_pred_perm", cfg.EXPERIMENT.FULL_DIR)
        io_file.save_tensor(loss_conf_perm, "loss_conf_perm", cfg.EXPERIMENT.FULL_DIR)
        io_file.save_tensor(exit_conf_perm, "exit_conf_perm", cfg.EXPERIMENT.FULL_DIR)
    
    else: # Load required files for RCP
        logger.info("Skipping model and data loading as no predictions required.")
        model, dataloader = None, None
        
        logger.info("Loading losses and exit files.")
        lambdas = rcp.lambda_grid(risk_control.lambda_end, risk_control.lambda_start, risk_control.lambda_step)
        logger.info(f"Lambda grid: {lambdas}")
        
        loss_pred = io_file.load_tensor("loss_pred", cfg.EXPERIMENT.LOAD_DIR)
        exit_pred = io_file.load_tensor("exit_pred", cfg.EXPERIMENT.LOAD_DIR)
        loss_conf = io_file.load_tensor("loss_conf", cfg.EXPERIMENT.LOAD_DIR)
        exit_conf = io_file.load_tensor("exit_conf", cfg.EXPERIMENT.LOAD_DIR)
        loss_pred_perm = io_file.load_tensor("loss_pred_perm", cfg.EXPERIMENT.LOAD_DIR)
        exit_pred_perm = io_file.load_tensor("exit_pred_perm", cfg.EXPERIMENT.LOAD_DIR)
        loss_conf_perm = io_file.load_tensor("loss_conf_perm", cfg.EXPERIMENT.LOAD_DIR)
        exit_conf_perm = io_file.load_tensor("exit_conf_perm", cfg.EXPERIMENT.LOAD_DIR)
        
    # Run risk-controlling procedure over trials
    eps_grid = torch.arange(risk_control.eps_start, risk_control.eps_end, risk_control.eps_step)
    test_risk_pred, test_exit_pred, rc_lambdas_pred = risk_control.rcp(lambdas, eps_grid, loss_pred, exit_pred, loss_pred_perm, exit_pred_perm, is_conf=False)
    test_risk_conf, test_exit_conf, rc_lambdas_conf = risk_control.rcp(lambdas, eps_grid, loss_conf, exit_conf, loss_conf_perm, exit_conf_perm, is_conf=True)
    
    io_file.save_json(test_risk_pred, "test_risk_pred", cfg.EXPERIMENT.FULL_DIR)
    io_file.save_json(test_exit_pred, "test_exit_pred", cfg.EXPERIMENT.FULL_DIR)
    io_file.save_json(rc_lambdas_pred, "rc_lambdas_pred", cfg.EXPERIMENT.FULL_DIR)
    io_file.save_json(test_risk_conf, "test_risk_conf", cfg.EXPERIMENT.FULL_DIR)
    io_file.save_json(test_exit_conf, "test_exit_conf", cfg.EXPERIMENT.FULL_DIR)
    io_file.save_json(rc_lambdas_conf, "rc_lambdas_conf", cfg.EXPERIMENT.FULL_DIR)
    
    if cfg.EXPERIMENT.PLOT:
        logger.info("Plotting test results.")
        plot_util.rcp_test_res(
            cfg,
            get_mean_and_std(test_risk_pred), 
            get_mean_and_std(test_exit_pred),
            get_mean_and_std(test_risk_conf),
            get_mean_and_std(test_exit_conf),
            eps_grid,
            plot_dir, "rcp_test_res.png"
        )
            
    logger.info("===== EXPERIMENT END =====")


if __name__ == "__main__":
    main()
