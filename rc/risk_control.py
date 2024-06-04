import numpy as np

from rc.bounds import ucb_wsr, ucb_hb, hb_p_value

from typing import List, Tuple, Dict


def rc_main(
    losses: np.array,
    exits: np.array,
    eps_grid: np.array,
    rcp_types: List[str],
    binary_loss: bool = False,
    loss_bound: int = 1,
    n_trials: int = 5,
    n_cal: int = 100,
    delta: float = 0.1,
    seed: int = 42,
) -> Tuple[Dict[str, List], Dict[str, List]]:
    """
    Compute the upper risk controlling exiting threshold.

    Args:
        losses: (K, n_cal) where K is the size of the lambda grid. 
                 Rows in losses array correspond to the descending ordering of lambdas.
        exits: (K,) average exit per threshold
        eps_grid: grid of risk levels
        rcp_types: list of risk control procedures
        binary_loss: whether the loss is binary
        loss_bound: upper bound on the loss
        n_trials: number of trials (calibration/test splits)
        n_cal: number of calibration datapoints
        delta: confidence level
        seed: random seed

    """
    np.random.seed(seed)

    for x in rcp_types:
        assert x in ["naive", "ltt", "ucb-wsr", "crc"]

    assert losses.shape[0] == exits.shape[0]
    _, N = losses.shape

    test_risk, eff_gains = {r: [] for r in rcp_types}, {r: [] for r in rcp_types}

    for _ in range(n_trials):
        # select n_cal datapoints from N
        cal_ids = np.random.choice(N, n_cal, replace=False)
        test_ids = np.setdiff1d(np.arange(N), cal_ids)

        cal_losses, test_losses = losses[:, cal_ids], losses[:, test_ids]

        # STAGE 1: find \hat{\lambda} on the calibration dataset
        rcp_lams = {r: [] for r in rcp_types}
        for rcp in rcp_types:
            for eps in eps_grid:
                if rcp == "naive":
                    lam_id = naive_lam(cal_losses, eps)
                elif rcp == "ucb-wsr":
                    lam_id = ucb_lam(
                        cal_losses, eps, delta, ucb_type="wsr", binary_loss=False, loss_bound=loss_bound
                    )
                elif rcp == "ltt":
                    lam_id = ltt_lam(
                        np.maximum(cal_losses, 0.0), eps, delta, binary_loss
                    )
                elif rcp == "crc":
                    lam_id = crc_lam(cal_losses, eps, loss_bound=loss_bound)
                rcp_lams[rcp].append(lam_id)

        # STAGE 2: using \hat{\lambda} from the stage 1 to compute test risk and efficiency gains
        for rcp in rcp_types:
            test_risk_e, eff_gains_e = [], []
            for e, eps in enumerate(eps_grid):
                lam_id = rcp_lams[rcp][e]
                test_risk_e.append(test_losses[lam_id].mean())
                eff_gains_e.append(exits[lam_id])
            test_risk[rcp].append(test_risk_e)
            eff_gains[rcp].append(eff_gains_e)

    return test_risk, eff_gains, rcp_lams


def naive_lam(losses: np.array, epsilon: float) -> int:
    """
    Find a naive lambda (Eq. 9)
    
    Args:
        losses: (K, n_cal) where K is the size of the lambda grid. 
                 Rows in losses array correspond to the descending ordering of lambdas.
        epsilon: tolerated risk level 
    
    """

    risk = losses.mean(axis=1)

    lams = (risk < epsilon).nonzero()[0]

    if len(lams) == 0:
        return 0
    else:
        return lams.max()


def crc_lam(losses: np.array, epsilon: float, loss_bound: float = 1.0) -> int:
    """
    Find risk controlling lambda based on Conformal Risk Control (CRC, Eq. 10)

    Args:
        losses: (K, n_cal) where K is the size of the lambda grid. 
                 Rows in losses array correspond to the descending ordering of lambdas.
        epsilon: tolerated risk level 
        loss_bound: upper bound on the loss
    
    """
    _, n_cal = losses.shape

    risk = losses.mean(axis=1)

    ucb = (n_cal + 1) * epsilon / n_cal - loss_bound / n_cal

    lams = (risk <= ucb).nonzero()[0]

    if len(lams) == 0:
        return 0
    else:
        return lams.max()


def ltt_lam(
    losses: np.array, epsilon: float, delta: float, binary_loss: bool
) -> int:
    """
    Find risk controlling lambda based on Learn-then-Test (LTT)

    Args:
        losses: (K, n_cal) where K is the size of the lambda grid. 
                 Rows in losses array correspond to the descending ordering of lambdas.
        epsilon: tolerated risk level 
        delta: confidence level
        binary_loss: whether the loss is binary
    """
    K, n_cal = losses.shape
    risk = losses.mean(axis=1)

    p_vals = []
    for k in range(K):
        p_vals.append(
            hb_p_value(
                risk=risk[k], n=n_cal, alpha=epsilon, binary_loss=binary_loss
            ).item()
        )

    lams = (np.array(p_vals) > delta).nonzero()[0]
    if len(lams) == 0:
        return -1
    else:
        return lams.min()


def ucb_lam(
    losses: np.array,
    epsilon: float,
    delta: float,
    ucb_type: str = "wsr",
    binary_loss: bool = False,
    loss_bound: float = 1,
) -> int:
    """
    Find risk-controlling lambda using upper confidence bound (UCB) calibration (Eq. 12)

    Args:
        losses: (K, n_cal) where K is the size of the lambda grid. 
                 Rows in losses array correspond to the descending ordering of lambdas.
        epsilon: tolerated risk level 
        delta: confidence level
        ucb_type: type of UCB. Options: "wsr", "hb"
        binary_loss: whether the loss is binary
        loss_bound: upper bound on the loss
    
    """
    K, n_cal = losses.shape

    ucb = []
    for k in range(K):
        if ucb_type == "wsr":
            ucb.append(ucb_wsr(losses[k], delta=delta, B=loss_bound))
        elif ucb_type == "hb":
            ucb.append(
                ucb_hb(
                    risk=losses[k].mean(),
                    delta=delta,
                    n_cal=n_cal,
                    binary_loss=binary_loss,
                )
            )
        else:
            raise ValueError(f"Invalid UCB type: {ucb_type}")

    lams = (np.array(ucb) < epsilon).nonzero()[0]

    # find the smallest lambda for which the UCB for all the larger lambdas is smaller than epsilon (Eq. 12)
    rc_lam = find_first_gap(lams)
    return rc_lam


def find_first_gap(arr: np.array) -> int:
    if len(arr) == 0:
        return 0
    for i in range(len(arr)):
        if arr[i] != i:
            return max(0, i - 1)
    return len(arr) - 1
