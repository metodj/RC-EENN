import os
import torch
import importlib
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import cv2

from plotting_style import *
# from google.colab.patches import cv2_imshow

RISK_LATEX = {
    "miscoverage": '$\mathcal{R}^{C} (\hat{y})$(Miscov.)',
    "miou": '$\mathcal{R}^{G} (\hat{y})$(mIoU)',
    "brier_pred": '$\mathcal{R}^{C} (\hat{p})$(Brier)',
    "brier_gt": '$\mathcal{R}^{G} (\hat{p})$(Brier)',
}
RISK_CONTROL_NAMES = {
    "naive": "Naive", 
    "crc": "CRC",
    "ucb": "UCB", 
    "ltt": "LTT"
}
RISK_CONTROL_COLS = {
    "naive": "tab:purple", 
    "crc": "tab:blue",
    "ucb": "tab:green", 
    "ltt": "tab:red"
}


def rcp_test_res(cfg, tr_pred, te_pred, tr_conf, te_conf, eps_grid, save_dir, save_name):
    STD_ALPHA = 0.3
    RISKS = 2
    LW = 1.5
    
    risk = cfg.RISK_CONTROL.RISK
    risk_conf = cfg.RISK_CONTROL.RISK_CONF
    perm = cfg.RISK_CONTROL.PERM
    
    if cfg.DATASET.NAME == "gta5" and risk == "miou":
        EPS_UPPER_COL_A, SPACER_COL_A, TICK_COL_A = 0.03, 0.0015, 0.005
    else:
        EPS_UPPER_COL_A, SPACER_COL_A, TICK_COL_A = 0.25, 0.01, 0.05
    EPS_UPPER_COL_B, SPACER_COL_B, TICK_COL_B = 0.25, 0.01, 0.05
    
    # ax[0,0] risk pred, ax[0,1] risk conf, ax[1,0] exit pred, ax[1,1] exit conf
    x = eps_grid
    fig, ax = plt.subplots(2, 2, figsize=(4, 2.4))
    
    for rcp in list(tr_pred.keys())[::-1]:
        
        # only plot a single baseline to avoid clutter
        if rcp in ["crc_perm", "ltt_perm"]:
            continue
        elif rcp in ["ucb_perm"]:
            RISK_CONTROL_NAMES[rcp] = f"UCB ({perm.capitalize()})"
            RISK_CONTROL_COLS[rcp] = "grey"
        
        # test risk pred
        ax[0, 0].plot(x, tr_pred[rcp]["mean"], lw=LW, label=RISK_CONTROL_NAMES[rcp], color=RISK_CONTROL_COLS[rcp])
        ax[0, 0].fill_between(x,
                                np.maximum(tr_pred[rcp]["mean"] - tr_pred[rcp]["std"], 0),
                                tr_pred[rcp]["mean"] + tr_pred[rcp]["std"],
                                alpha=STD_ALPHA, color=RISK_CONTROL_COLS[rcp]
                                )
        # test risk conf
        ax[0, 1].plot(x, tr_conf[rcp]["mean"], lw=LW, label=RISK_CONTROL_NAMES[rcp], color=RISK_CONTROL_COLS[rcp])
        ax[0, 1].fill_between(x,
                                np.maximum(tr_conf[rcp]["mean"] - tr_conf[rcp]["std"], 0),
                                tr_conf[rcp]["mean"] + tr_conf[rcp]["std"],
                                alpha=STD_ALPHA, color=RISK_CONTROL_COLS[rcp]
                                )
        # test exit pred
        ax[1, 0].plot(x, te_pred[rcp]["mean"], lw=LW, label=RISK_CONTROL_NAMES[rcp], color=RISK_CONTROL_COLS[rcp])
        ax[1, 0].fill_between(x,
                                np.maximum(te_pred[rcp]["mean"] - te_pred[rcp]["std"], 1),
                                np.minimum(te_pred[rcp]["mean"] + te_pred[rcp]["std"], 4),
                                alpha=STD_ALPHA, color=RISK_CONTROL_COLS[rcp]
                                )
        # test exit conf
        ax[1, 1].plot(x, te_conf[rcp]["mean"], lw=LW, label=RISK_CONTROL_NAMES[rcp], color=RISK_CONTROL_COLS[rcp])
        ax[1, 1].fill_between(x,
                                np.maximum(te_conf[rcp]["mean"] - te_conf[rcp]["std"], 1),
                                np.minimum(te_conf[rcp]["mean"] + te_conf[rcp]["std"], 4),
                                alpha=STD_ALPHA, color=RISK_CONTROL_COLS[rcp]
                                )
    
    ticks_a = [i for i in np.arange(0, EPS_UPPER_COL_A + SPACER_COL_A, TICK_COL_A)]
    ticks_b = [i for i in np.arange(0, EPS_UPPER_COL_B + SPACER_COL_B, TICK_COL_B)]
    
    ax[0, 0].set_ylim(-SPACER_COL_A, EPS_UPPER_COL_A + SPACER_COL_A)
    ax[0, 0].set_yticks(ticks_a)
    ax[0, 0].set_xlim(-SPACER_COL_A, EPS_UPPER_COL_A + SPACER_COL_A)
    ax[0, 0].set_xticks(ticks_a)
    ax[1, 0].set_yticks([1, 2, 3, 4])
    ax[1, 0].set_xlim(-SPACER_COL_A, EPS_UPPER_COL_A + SPACER_COL_A)
    ax[1, 0].set_xticks(ticks_a)

    ax[0, 1].set_ylim(-SPACER_COL_B, EPS_UPPER_COL_B + SPACER_COL_B)
    ax[0, 1].set_yticks(ticks_b)
    ax[0, 1].set_xlim(-SPACER_COL_B, EPS_UPPER_COL_B + SPACER_COL_B)
    ax[0, 1].set_xticks(ticks_b)
    ax[1, 1].set_yticks([1, 2, 3, 4])
    ax[1, 1].set_xlim(-SPACER_COL_B, EPS_UPPER_COL_B + SPACER_COL_B)
    ax[1, 1].set_xticks(ticks_b)
    
    for j in range(RISKS):
        if (EPS_UPPER_COL_A == EPS_UPPER_COL_B) and (j > 0):
            ax[0, j].set_yticklabels([])
            ax[1, j].set_yticklabels([])

        ax[0, j].scatter(0, 0, marker='x', color="black")
        ax[1, j].scatter(0, 4, marker='x', color="black")
        
        # add optimal risk line
        ax[0, j].plot([0, 1], [0, 1], 'k--')
        # add optimal exit line
        # ax[1, j].plot([0, 0], [4, 1], 'k--')
        # ax[1, j].plot([0, 0.5], [1, 1], 'k--')
        
    ax[0, 0].set_title(RISK_LATEX[risk])
    ax[0, 1].set_title(RISK_LATEX[risk_conf])   
    
    ax[0, 0].set_ylabel("Test Risk")
    ax[1, 0].set_ylabel("Exit Layer ($\downarrow$)")
    fig.text(0.51, -0.01, r"Risk Level $\epsilon$", ha='center')

    ax[1, 1].legend(loc='upper right')

    fname = "rcp_test_res.png" if save_name is None else save_name
    plt.savefig(
        os.path.join(save_dir, fname),
        bbox_inches="tight",
    )


def plot_lambda_risk_pval(lambdas, risk, pval, rc_lam, epsilon, delta, save_dir, save_name=None):
    """
    Plot the risk and p-value as a function of lambda.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))
    ax1.plot(lambdas, risk, color="black")
    ax1.set_xlim(max(lambdas) + 1e-2, min(lambdas) - 1e-3)
    ax1.set_xlabel(r"Threshold parameter $\lambda$")
    ax1.set_ylabel("Risk")
    ax1.axhline(y=epsilon, color="red", ls=":", label=r"Risk level $\epsilon$", lw=2)
    ax1.axvline(
        x=rc_lam, color="green", ls=":", label=r"Risk controlling $\hat{\lambda}$", lw=2
    )
    ax1.legend()

    ax2.plot(lambdas, pval, color="black")
    ax2.set_xlim(max(lambdas) + 1e-2, min(lambdas) - 1e-3)
    ax2.set_xlabel(r"Threshold parameter $\lambda$")
    ax2.set_ylabel("P-value")
    ax2.axhline(y=epsilon, color="red", ls=":", label=r"Risk level $\epsilon$", lw=2)
    ax2.axvline(
        x=rc_lam, color="green", ls=":", label=r"Risk controlling $\hat{\lambda}$", lw=2
    )
    ax2.legend()

    fig.suptitle(rf"Target guarantee: $\epsilon={epsilon}$, $\delta={delta}$")
    fname = "lambda_risk_pval.png" if save_name is None else save_name
    plt.savefig(
        os.path.join(save_dir, fname),
        bbox_inches="tight",
    )


def plot_lambda_exits(lambdas, exits, rc_lam, rc_lam_conf, rc_lam_joint, save_dir):
    """
    Plot the exit statistics as a function of lambda.
    """

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(lambdas, exits, color="black")
    ax.set_xlim(max(lambdas) + 1e-2, min(lambdas) - 1e-3)
    ax.set_xlabel(r"Threshold parameter $\lambda$")
    ax.set_ylabel("Mean exit index")
    
    ax.axvline(x=rc_lam, color="green", ls=":", label=r"Risk controlling $\hat{\lambda_P}$", lw=2)
    ax.axvline(x=rc_lam_conf, color="blue", ls=":", label=r"Risk controlling $\hat{\lambda_C}$", lw=2)
    ax.axvline(x=rc_lam_joint, color="red", ls=":", label=r"Risk controlling $\hat{\lambda_J}$", lw=2)
    
    rc_exit = exits[torch.where(lambdas == rc_lam)[0]].item()
    rc_exit_conf = exits[torch.where(lambdas == rc_lam_conf)[0]].item()
    rc_exit_joint = exits[torch.where(lambdas == rc_lam_joint)[0]].item()
    
    ax.axhline(y=rc_exit, color="green", ls=":", label=f"Exit index: {rc_exit:.2f}", lw=2)
    ax.axhline(y=rc_exit_conf, color="blue", ls=":", label=f"Exit index: {rc_exit_conf:.2f}", lw=2)
    ax.axhline(y=rc_exit_joint, color="red", ls=":", label=f"Exit index: {rc_exit_joint:.2f}", lw=2)

    ax.legend()
    fig.suptitle("Computational savings (exit index)")

    plt.savefig(os.path.join(save_dir, "lambda_exits.png"), bbox_inches="tight")


def plot_test_risk(risk, losses, exits, epsilon, delta, save_dir, save_name):
    """
    Plot the test risk and loss statistics.
    """
    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    ax.hist(
        losses,
        bins=30,
        density=True,
        color="lightgray",
        edgecolor="black",
        linewidth=1,
        alpha=0.8,
    )
    ax.set_ylabel("Density")
    ax.set_xlabel("Test losses")
    ax.set_xlim(-0.03, 1.03)
    ax.axvline(x=epsilon, color="red", ls="-", label=r"Risk level $\epsilon$", lw=2)
    ax.axvline(
        x=risk,
        color="green",
        ls="-",
        label=f"Test risk: {risk:.3f}\nMean exit: {exits.float().mean():.2f}",
        lw=2,
    )
    ax.legend()
    fig.suptitle("Risk control on test data")
    fname = "test_risk.png" if save_name is None else save_name
    plt.savefig(
        os.path.join(save_dir, fname),
        bbox_inches="tight",
    )