import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binom
from scipy.optimize import brentq


def lambda_grid(min: float, max: float, step):
    """Create a grid of lambda values.

    Args:
        min (float): Minimum lambda value.
        max (float): Maximum lambda value.
        step (float): Step size between lambda values.

    Returns:
        torch.Tensor: Grid of lambda values.
    """
    lambdas = torch.arange(max, min, -step, dtype=torch.float)
    # every lambda has at most the same number of decimal points as step
    lambdas = torch.round(lambdas / step) * step
    return lambdas


def conf_measure(prob: torch.Tensor, criterion: str):
    """Compute pixel-wise confidence measure.

    Args:
        prob (torch.Tensor): Probability distribution (logits).
            Assumed to be in last dim of given tensor.
        criterion (str): Confidence criterion.

    Returns:
        torch.Tensor: Confidence measure.
    """

    if criterion == "top1":
        conf = torch.max(prob, dim=-1).values
    elif criterion == "topdiff":  # top-2 - top-1
        conf = torch.topk(prob, k=2, dim=-1).values.diff(dim=-1).abs().squeeze(dim=-1)
    elif criterion == "entropy":  # normalized entropy
        conf = -torch.sum(prob * torch.log(prob), dim=-1)
        conf /= torch.log(torch.tensor(prob.shape[-1]))
        conf = 1.0 - conf
    else:
        raise ValueError("Invalid confidence measure.")

    return conf


def conf_measure_aggr(conf: torch.Tensor, criterion: str, device: str):
    """Aggregates pixel-wise confidence measure to image level.

    Args:
        conf (torch.Tensor): Pixel-wise confidence measure.
            Assuming image (H, W) in last two dim of given tensor.
        criterion (str): Aggregation criterion.

    Returns:
        torch.Tensor: Image-level aggregated confidence measure.
    """
    if criterion == "mean":
        aggr_conf = torch.mean(conf, dim=(-2, -1))
    elif criterion == "median":
        aggr_conf = torch.quantile(conf.flatten(-2, -1), 0.5, dim=-1)
    elif criterion == "quantile":
        q = 0.25
        aggr_conf = torch.quantile(conf.flatten(-2, -1), q, dim=-1)
    elif criterion == "patch":
        patch_size = 50  # (int x int)
        qp = 0.01
        aggr_conf = patch_aggr(conf, patch_size, qp, device)
    else:
        raise ValueError("Invalid confidence aggregation.")

    return aggr_conf


def patch_aggr(conf: torch.Tensor, patch_size: int, q: float, device: str):
    """inspired by https://github.com/IML-DKFZ/values/blob/main/evaluation/uncertainty_aggregation/aggregate_uncertainties.py

    Slides a window across conf, computes mean confidence per patch, and selects
    quantile across patches as an image-level confidence measure.
    
    Can handle batch dimension (N, H, W)
    """
    patch_sizes = [patch_size, patch_size]
    kernel = torch.ones(1, 1, *patch_sizes).to(device)
    
    if conf.dim() == 2: # (H, W)
        patch_aggr = F.conv2d(conf.unsqueeze(0).unsqueeze(0).to(device), kernel, padding="valid").squeeze()
        patch_aggr /= torch.prod(torch.tensor(patch_sizes))  # mean per patch
        conf_aggr = torch.quantile(patch_aggr.flatten(), q)
    else: # existing batch dim (N, H, W)
        patch_aggr = F.conv2d(conf.unsqueeze(1).to(device), kernel, padding="valid") / torch.prod(torch.tensor(patch_sizes))
        conf_aggr = torch.quantile(patch_aggr.flatten(-2, -1), q, dim=-1)
    
    return conf_aggr


def miou_per_sample(pred: torch.Tensor, gt_labels: torch.Tensor):
    NR_CLASSES = 19
    IGNORE_LABEL = 255

    N, H, W = pred.shape
    # Adjust the pred tensor to map IGNORE_LABEL to a valid class index (NR_CLASSES)
    pred = pred.view(N, -1)
    gt_labels = gt_labels.view(N, -1)
    # Treat IGNORE_LABEL as another class
    pred[gt_labels == IGNORE_LABEL] = NR_CLASSES
    gt_labels[gt_labels == IGNORE_LABEL] = NR_CLASSES

    # Compute global index for valid predictions
    flattened_index = (gt_labels * (NR_CLASSES + 1) + pred).view(N, -1)
    # Initialize confusion matrix for all samples (including an extra row/column for IGNORE_LABEL)
    all_cm = torch.zeros((N, NR_CLASSES + 1, NR_CLASSES + 1), device=pred.device, dtype=torch.long)
    # Efficient batch fill of the confusion matrix
    max_index = (NR_CLASSES + 1) ** 2 - 1
    for i in range(N):
        all_cm[i] = torch.bincount(flattened_index[i], minlength=max_index + 1).view(NR_CLASSES + 1, NR_CLASSES + 1)

    # Exclude the IGNORE_LABEL from the IoU calculations
    all_cm = all_cm[:, :NR_CLASSES, :NR_CLASSES]
    # Calculate IoU for each class per sample
    pos = all_cm.sum(dim=2)  # Sum over predicted classes
    res = all_cm.sum(dim=1)  # Sum over true classes
    tp = torch.diagonal(all_cm, dim1=1, dim2=2)
    iou = tp / (pos + res - tp).clamp(min=1e-10)

    return iou.mean(dim=1)


def brier_loss(pred: torch.Tensor, pred_L: torch.Tensor, 
               label: torch.Tensor, use_label: bool, 
               ignore_label: int, keepdim: bool = False, 
               sample_full: bool = True, C: float = 1.0):
    
    # pred (H, W, C), pred_L (H, W, C), label (H, W)
    h, w, c = pred_L.shape

    # work with flattened long tensors for one-hot encoding
    pred_L = pred_L.flatten(0, 1) # (H*W, C)
    pred = pred.flatten(0, 1) # (H*W, C)
    
    if sample_full: # one-hot of sample from full model (unbiased)
        pred_L_oh = F.one_hot(torch.multinomial(pred_L, num_samples=1).squeeze(), num_classes=c) # (H*W, C)
    else: # one-hot of argmax from full model (biased)
        pred_L_oh = F.one_hot(pred_L.argmax(dim=-1), num_classes=c) # (H*W, C)

    if use_label: # use one-hot gt
        label = label.flatten(0, 1) # (H*W)
        replace_pix = (label == ignore_label)
        label[replace_pix] = 0 # a valid arbitrary class as place-holder
        label_oh = F.one_hot(label.to(torch.int64), num_classes=c) # (H*W, C)
        # replace ignore_label with full model one-hot
        label_oh[replace_pix] = pred_L_oh[replace_pix]
        brier_loss = C * torch.sum((pred - label_oh) ** 2, dim=-1) # (H*W)
        brier_loss_L = C * torch.sum((pred_L - label_oh) ** 2, dim=-1) # (H*W)
    else: # use one-hot full model
        brier_loss = C * torch.sum((pred - pred_L_oh) ** 2, dim=-1) # (H*W)
        brier_loss_L = C * torch.sum((pred_L - pred_L_oh) ** 2, dim=-1) # (H*W)
        
    if keepdim: # (H*W) -> (H, W)
        brier_loss = brier_loss.view(h, w)
        brier_loss_L = brier_loss_L.view(h, w)
    else: # (H*W) -> (1,)
        brier_loss = brier_loss.mean()
        brier_loss_L = brier_loss_L.mean()
        
    return brier_loss, brier_loss_L


def compute_risk_dist(dist_loss_early: torch.Tensor, dist_loss_full: torch.Tensor, criterion: str):
    # dist_loss: (N,)
    assert dist_loss_early.shape == dist_loss_full.shape, "Loss shapes must match."
    
    if (criterion == "brier_pred") or (criterion == "brier_gt"):
        all_losses = (dist_loss_early - dist_loss_full)
        # all_losses = torch.max(torch.zeros_like(dist_loss_early), (dist_loss_early - dist_loss_full))
    else:
        raise ValueError("Invalid risk measure.")
    
    risk = all_losses.mean()
    assert 0 <= risk.item() <= 1, "Risk must be in [0, 1]."
    return risk, all_losses


def compute_risk(
    pred_early: torch.Tensor,
    pred_full: torch.Tensor,
    criterion: str,
    gt_labels: torch.Tensor = None,
    batch_size: int = 100,
):
    n, h, w = pred_early.shape
    assert pred_full.size() == (n, h, w), f"{pred_full.size()=}"
    assert ((gt_labels is not None) if criterion == "miou" else True)

    n, h, w = pred_early.shape
    num_batches = (n + batch_size - 1) // batch_size

    losses = [] 
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n)

        batch_pred_early = pred_early[start_idx:end_idx]
        batch_pred_full = pred_full[start_idx:end_idx]

        if criterion == "miscoverage":
            batch_loss = 1.0 - (batch_pred_early.view(end_idx - start_idx, -1) == batch_pred_full.view(end_idx - start_idx, -1)).sum(dim=-1).div(h * w) # (N_batch,)
        elif criterion == "miou":
            batch_gt_labels = gt_labels[start_idx:end_idx]
            miou_early = miou_per_sample(batch_pred_early, batch_gt_labels) # (N_batch,)
            miou_full = miou_per_sample(batch_pred_full, batch_gt_labels) # (N_batch,)
            # batch_loss = torch.max(torch.zeros_like(miou_early), (1 - miou_early) - (1 - miou_full)) # (N_batch,)
            batch_loss = (1 - miou_early) - (1 - miou_full) # (N_batch,)
        else:
            raise ValueError("Invalid risk measure.")
        
        losses.append(batch_loss)
    
    all_losses = torch.cat(losses, dim=0) # (N,)
    risk = all_losses.mean() # (1,)

    assert 0 <= risk.item() <= 1, "Risk must be in [0, 1]."
    return risk, all_losses


def get_naive_lambda(losses: torch.Tensor, eps):
    # losses: (N_lam, N_cal)
    # select lambda closest s.t. estimated risk is controlled on calibration data
    risk = losses.mean(dim=1) # (N_lam, )
    lam_ids = (risk <= eps).nonzero(as_tuple=True)[0]
    if lam_ids.shape[0] == 0: # no accept
        lam_id = torch.tensor([0])
        lam_ids = torch.tensor([0])
    else:
        lam_id = lam_ids[-1] # smallest lambda (from below)
    return lam_id, lam_ids


def get_crc_lambda(losses: torch.Tensor, eps, loss_B):
    # losses: (N_lam, N_cal)
    # select lambda closest s.t. estimated risk is controlled for CRC bound on calibration data
    _, N_cal = losses.shape
    risk = losses.mean(dim=1) # (N_lam, )
    crc = (N_cal + 1) * eps / N_cal - loss_B / N_cal
    lam_ids = (risk <= crc).nonzero(as_tuple=True)[0]
    if lam_ids.shape[0] == 0: # no accept
        lam_id = torch.tensor([0])
        lam_ids = torch.tensor([0])
    else:
        lam_id = lam_ids[-1] # smallest lambda (from below)
    return lam_id, lam_ids


def WSR_mu_plus(x, delta, maxiters=1000, B=1, eps=1e-10):
    # Waudby-Smith-Ramdas bound for UCB risk
    n = x.shape[0]
    muhat = (np.cumsum(x) + 0.5) / (1 + np.array(range(1, n + 1)))
    sigma2hat = (np.cumsum((x - muhat)**2) + 0.25) / (1 + np.array(range(1, n + 1))) 
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


def get_ucb_lambda(losses: torch.Tensor, eps, delta, loss_B):
    # losses: (N_lam, N_cal)
    # select lambda closest s.t. UCB risk is controlled on calibration data
    N_lam, _ = losses.shape
    ucb = torch.zeros((N_lam,))
    for i in range(N_lam):
        ucb[i] = torch.tensor([WSR_mu_plus(losses[i].to(ucb.device).numpy(), delta=delta, B=loss_B)])
    
    lam_ids = (ucb <= eps).nonzero(as_tuple=True)[0]
    if lam_ids.shape[0] == 0: # no accept
        lam_id = torch.tensor([0])
        lam_ids = torch.tensor([0])
    else:
        lam_id = lam_ids[-1] # smallest lambda (from below)
    return lam_id, lam_ids, ucb


def hb_p_value(risk: float, n, epsilon):
    """Compute the p-value of the Hoeffding-Bentkus bound.
    Adapted from https://github.com/aangelopoulos/ltt/blob/main/core/bounds.py

    Args:
        risk: Computed risk estimate.
        n: Number of calibration samples.
        epsilon: Tolerated risk level.

    Returns:
        p-value.
    """
    bentkus_p_value = np.e * binom.cdf(np.ceil(n * risk), n, epsilon)
    a, b = min(risk, epsilon), epsilon
    h1 = a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
    hoeffding_p_value = np.exp(-n * h1)
    return min(bentkus_p_value, hoeffding_p_value).clip(max=1)


def get_ltt_lambda(losses, eps, delta, loss_B):
    # losses: (N_lam, N_cal)
    # select lambda closest s.t. LTT p-value is controlled on calibration data
    
    # Note that LTT requires losses in [0, 1], thus we first need to rescale the losses
    losses = torch.max(torch.zeros_like(losses), (1 / loss_B) * losses)
    
    N_lam, N_cal = losses.shape
    risk = losses.mean(dim=1) # (N_lam,)
    pval = torch.zeros((N_lam,))
    for i in range(N_lam):
        pval[i] = torch.tensor([hb_p_value(risk[i].item(), N_cal, eps)])
    
    lam_ids = (pval <= delta).nonzero(as_tuple=True)[0]
    if lam_ids.shape[0] == 0: # no accept
        lam_id = torch.tensor([0])
        lam_ids = torch.tensor([0])
    else:
        lam_id = lam_ids[-1] # smallest lambda (from below)
    return lam_id, lam_ids, pval
