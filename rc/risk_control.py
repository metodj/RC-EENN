import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq

from typing import List, Tuple, Dict


# TODO: verify correctness, add unit tests. is the correctness affected by the ordering of the lambda grid?
def get_naive_lam(losses: np.array, epsilon: float) -> int:
    """
    K - lambda grid size

    losses: (K, n_cal)
    """

    risk = losses.mean(axis=1)

    lams = (risk < epsilon).nonzero()[0]

    if len(lams) == 0:
        return 0
    else:
        return lams.max()


# TODO: verify correctness, add unit tests. is the correctness affected by the ordering of the lambda grid?
def get_crc_lam(losses: np.array, epsilon: float, loss_bound: float = 1.0) -> int:
    """
    K - lambda grid size

    losses: (K, n_cal)
    """
    _, n_cal = losses.shape

    risk = losses.mean(axis=1)

    ucb = (n_cal + 1) * epsilon / n_cal - loss_bound / n_cal

    lams = (risk <= ucb).nonzero()[0]

    if len(lams) == 0:
        return 0
    else:
        return lams.max()


def find_first_gap(arr: np.array) -> int:
    if len(arr) == 0:
        return 0
    for i in range(len(arr)):
        if arr[i] != i:
            return max(0, i - 1)
    return len(arr) - 1


# TODO: verify correctness, add unit tests. is the correctness affected by the ordering of the lambda grid?
def get_ucb_lam(
    losses: np.array,
    epsilon: float,
    delta: float,
    ucb_type: str = "wsr",
    binary_loss: bool = False,
    B: float = 1,
) -> int:
    """
    K - lambda grid size

    losses: (K, n_cal)
    """
    K, n_cal = losses.shape

    ucb = []
    for k in range(K):
        if ucb_type == "wsr":
            ucb.append(ucb_wsr(losses[k], delta=delta, B=B))
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
    return find_first_gap(lams)


# TODO: verify correctness, add unit tests. is the correctness affected by the ordering of the lambda grid?
def get_ltt_lam(
    losses: np.array, epsilon: float, delta: float, binary_loss: bool
) -> int:
    """
    K - lambda grid size

    losses: (K, n_cal)
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


def rcp_main(
    losses: np.array,
    exits: np.array,
    eps_grid: np.array,
    rcp_types: List[str],
    binary_loss: bool = False,
    n_trials: int = 5,
    n_cal: int = 100,
    delta: float = 0.1,
    seed: int = 42,
    B: int = 1,
) -> Tuple[Dict[str, List], Dict[str, List]]:
    """
    K - lambda grid size
    N - number of validation datapoints (e.g., 10K for Cifar100)

    losses: (K, N)
    exits: (K, N)
    eps_grid: (E, )
    rcp_types: List of risk control types. Options: "naive", "ucb-wsr", "ltt", "crc"
        - note that "ltt" is equivalent to "ucb-hb"

    """
    np.random.seed(seed)

    for x in rcp_types:
        assert x in ["naive", "ltt", "ucb-wsr", "crc"]

    assert losses.shape[0] == exits.shape[0]
    _, N = losses.shape

    # check for marginal monotonicity
    # also check that the orders of rows in losses corresponds to the descending ordering of lambdas
    # assert np.allclose(losses.mean(axis=1), np.sort(losses.mean(axis=1))), "Losses are not in descending order: {} vs {}".format(losses.mean(axis=1), np.sort(losses.mean(axis=1)))

    test_risk, eff_gains = {r: [] for r in rcp_types}, {r: [] for r in rcp_types}

    for _ in range(n_trials):
        # select n_cal datapoints from N
        cal_ids = np.random.choice(N, n_cal, replace=False)
        test_ids = np.setdiff1d(np.arange(N), cal_ids)

        cal_losses, test_losses = losses[:, cal_ids], losses[:, test_ids]
        if len(exits.shape) == 1:
            test_exits = exits
        else:
            test_exits = exits[:, test_ids]

        # STAGE 1: find \hat{\lambda} on the calibration dataset
        rcp_lams = {r: [] for r in rcp_types}
        for rcp in rcp_types:
            for eps in eps_grid:
                if rcp == "naive":
                    lam_id = get_naive_lam(cal_losses, eps)
                elif rcp == "ucb-wsr":
                    lam_id = get_ucb_lam(
                        cal_losses, eps, delta, ucb_type="wsr", binary_loss=False, B=B
                    )
                elif rcp == "ltt":
                    lam_id = get_ltt_lam(
                        np.maximum(cal_losses, 0.0), eps, delta, binary_loss
                    )
                elif rcp == "crc":
                    lam_id = get_crc_lam(cal_losses, eps, loss_bound=B)
                rcp_lams[rcp].append(lam_id)

        # STAGE 2: using \hat{\lambda} from the stage 1 to compute test risk and efficiency gains
        for rcp in rcp_types:
            test_risk_e, eff_gains_e = [], []
            for e, eps in enumerate(eps_grid):
                lam_id = rcp_lams[rcp][e]
                test_risk_e.append(test_losses[lam_id].mean())
                if len(exits.shape) == 1:
                    eff_gains_e.append(test_exits[lam_id])
                else:
                    eff_gains_e.append(test_exits[lam_id].mean())
            test_risk[rcp].append(test_risk_e)
            eff_gains[rcp].append(eff_gains_e)

    return test_risk, eff_gains, rcp_lams


# adapted from https://github.com/aangelopoulos/rcps/blob/main/core/bounds.py
def ucb_wsr(x, delta, maxiters=1000, B=1, eps=1e-10):
    n = x.shape[0]
    muhat = (np.cumsum(x) + 0.5) / (1 + np.array(range(1, n + 1)))
    sigma2hat = (np.cumsum((x - muhat) ** 2) + 0.25) / (1 + np.array(range(1, n + 1)))
    sigma2hat[1:] = sigma2hat[:-1]
    sigma2hat[0] = 0.25
    nu = np.minimum(np.sqrt(2 * np.log(1 / delta) / n / sigma2hat), 1 / B)

    def _Kn(mu):
        return np.max(np.cumsum(np.log(1 - nu * (x - mu)))) + np.log(delta)

    if _Kn(1) < 0:
        return B
    if _Kn(eps) > 0:
        return eps
    return brentq(_Kn, eps, 1 - eps, maxiter=maxiters)


def ucb_hb(risk, delta, n_cal, binary_loss, step=0.01):
    alphas = np.arange(0.01, 1.0 + step, step)[::-1]
    for i in range(len(alphas)):
        if (
            hb_p_value(risk=risk, n=n_cal, alpha=alphas[i], binary_loss=binary_loss)
            >= delta
        ):
            return alphas[i]
    return 0.0


# adapted from https://github.com/aangelopoulos/ltt/blob/main/core/bounds.py
def hb_p_value(
    risk: float,
    n: int,
    alpha: float = 0.05,
    eps: float = 1e-3,
    binary_loss: bool = False,
):
    """Compute the p-value of the Hoeffding-Bentkus bound.

    Args:
        risk: Computed risk estimate.
        n: Number of calibration samples.
        alpha: Tolerated risk level.

    Returns:
        p-value.
    """
    if binary_loss:
        p_value = binom.cdf(np.ceil(n * risk), n, alpha)
    else:
        bentkus_p_value = np.e * binom.cdf(np.ceil(n * risk), n, alpha)
        a, b = min(risk, alpha), alpha
        h1 = a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
        hoeffding_p_value = np.exp(-n * h1)
        p_value = min(bentkus_p_value, hoeffding_p_value)

    assert 0 - eps <= p_value <= 1 + eps, "p-value must be in [0, 1]: {}".format(
        p_value
    )
    return p_value
