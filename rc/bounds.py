import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq


# adapted from https://github.com/aangelopoulos/rcps/blob/main/core/bounds.py
def ucb_wsr(x, delta, maxiters=1000, B=1, eps=1e-10):
    """
    Compute the upper confidence bound (UCB) based on the Waudby-Smith Ramdas (WSR) bound.

    Args:
        TODO

    Returns:
        TODO
    """
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
    """
    Compute the upper confidence bound (UCB) based on the Hoeffding-Bentkus (HB) bound.

    Args:
        TODO

    Returns:
        TODO
    """
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
    """
    Compute the p-value of the Hoeffding-Bentkus bound.

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
