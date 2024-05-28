import pickle
import numpy as np
import torch

from typing import Tuple


def img_class_inference(
    model: str, dataset: str, path: str, lambdas: np.array, risk_type: str, non_negative: bool = True) -> Tuple[np.array, np.array]:

    with open(f"{path}/{model}/{dataset}.p", "rb") as f:
        data = pickle.load(f)

    if len(data) == 2:
        logits, targets = data
    else:
        logits, targets, _ = data

    logits = torch.transpose(logits, 0, 1)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = "cpu"
    logits = logits.to(device)
    targets = targets.to(device)
    targets = targets.to(torch.int64)

    preds = get_preds_per_exit(logits)
    acc = get_acc_per_exit(preds, targets)

    print(logits.shape, targets.shape)
    print(acc)

    conf = conf_measures(
        logits, input_type="logits" if "calib" not in model else "probs"
    )

    preds = logits.argmax(dim=2)
    preds_full = preds[:, -1]
    conf_full = conf[:, -1]

    lam_losses, lam_exits = [], []
    for l in lambdas:

        exits = compute_exits(conf, threshold=l)
        preds_ee = preds[torch.arange(len(preds)), exits - 1]

        if risk_type == "prediction-consistency":
            _, losses = compute_risk(
                preds_ee,
                preds_full,
                criterion=risk_type,
                non_negative=non_negative,
            )
        elif risk_type == "prediction-gt-gap":
            _, losses = compute_risk(
                preds_ee,
                preds_full,
                criterion=risk_type,
                gt_labels=targets,
                non_negative=non_negative,
            )
        elif risk_type == "confidence-brier":
            logits_ee = logits[torch.arange(len(logits)), exits - 1]
            if "calib" in model:
                probs_ee = logits_ee
                probs_full = logits[:, -1]
            else:
                probs_ee = torch.nn.functional.softmax(logits_ee, dim=1)
                probs_full = torch.nn.functional.softmax(logits[:, -1], dim=1)

            _, losses = compute_risk(
                probs_ee,
                probs_full,
                criterion=risk_type,
                gt_labels=targets,
                non_negative=non_negative,
            )
        elif risk_type == "confidence-brier-top-pred":
            logits_ee = logits[torch.arange(len(logits)), exits - 1]
            if "calib" in model:
                probs_ee = logits_ee
                probs_full = logits[:, -1]
            else:
                probs_ee = torch.nn.functional.softmax(logits_ee, dim=1)
                probs_full = torch.nn.functional.softmax(logits[:, -1], dim=1)

            _, losses = compute_risk(
                probs_ee,
                probs_full,
                criterion=risk_type,
                non_negative=non_negative,
            )
        elif risk_type == "confidence-hellinger":
            logits_ee = logits[torch.arange(len(logits)), exits- 1]
            if "calib" in model:
                probs_ee = logits_ee
                probs_full = logits[:, -1]
            else:
                probs_ee = torch.nn.functional.softmax(logits_ee, dim=1)
                probs_full = torch.nn.functional.softmax(logits[:, -1], dim=1)

            _, losses = compute_risk(
                probs_ee,
                probs_full,
                criterion=risk_type,
                non_negative=non_negative,
            )
        else:
            raise ValueError(f"Invalid risk type: {risk_type}")
        
        lam_losses.append(losses.cpu().numpy())
        lam_exits.append(exits.cpu().numpy())

    return np.array(lam_losses), np.array(lam_exits)



def compute_risk(
    output_early: torch.Tensor,
    output_full: torch.Tensor,
    criterion: str,
    gt_labels: torch.Tensor = None,
    non_negative: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # pred_early: (N, )
    # pred_full: (N, )

    assert output_early.shape == output_full.shape, "Prediction shapes must match."
    if criterion == "confidence-brier" or criterion == "confidence-hellinger" or criterion == "confidence-brier-top-pred":
        assert output_early.dim() == 2, "Prediction must be 2D  (N, C)."
    else:
        assert output_early.dim() == 1, "Prediction must be 1D (N, )."
    if 'confidence' in criterion:
        assert 0 <= output_full.min() and output_full.max() <= 1
        assert 0 <= output_early.min() and output_early.max() <= 1

    n = output_full.shape[0]

    if criterion == "prediction-consistency":
        loss = 1.0 - (output_full == output_early).float()

    elif criterion == "prediction-gt-gap":
        assert gt_labels is not None, "Ground truth labels must be provided."
        loss = (output_early != gt_labels).float() - (output_full != gt_labels).float()

    elif criterion == "confidence-hellinger":
        loss = hellinger_distance(output_early, output_full)

    elif criterion == "confidence-brier":
        assert gt_labels is not None, "Ground truth labels must be provided."
        loss = brier_score(output_early, gt_labels) - brier_score(output_full, gt_labels)
    
    elif criterion == "confidence-brier-top-pred":
        loss = brier_score(output_early, output_full.argmax(dim=1)) - brier_score(output_full, output_full.argmax(dim=1))

    else:
        raise ValueError("Invalid risk measure.")
    
    if non_negative:
        loss = torch.max(loss, torch.tensor(0.0))
    
    risk = loss.mean()

    # assert 0 <= risk.item() <= 1, "Risk must be in [0, 1]."
    assert risk.item() <= 1

    return risk, loss


@staticmethod
def conf_measures(input_tensor: torch.Tensor, input_type: str = 'logits', conf_measure: str = 'top_softmax') -> torch.Tensor:
    """
    logits: (N, L, C)
    returns:  
        conf: (N, L)
    """
    assert len(input_tensor.shape) == 3
    assert input_type in ['logits', 'probs']

    if conf_measure == 'top_softmax':
        # convert logits to probabilities
        if input_type == 'logits':
            input_tensor = torch.softmax(input_tensor, dim=2)
        conf = torch.max(input_tensor, dim=2).values
        
    else:
        raise ValueError(f'conf_measure: {conf_measure} not implemented')
    
    return conf


def compute_exits(conf: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    conf: (N, L)
    returns:  
        exits: (N,)
    """
    assert len(conf.shape) == 2
    assert 0 <= threshold <= 1

    mask = (conf > threshold).int()
    exits = mask.argmax(axis=1)
    rows_with_no_threshold = torch.sum(mask, dim=1) == 0
    exits[rows_with_no_threshold] = mask.shape[1] - 1
    exits = exits + 1

    assert exits.shape == (conf.shape[0],)
    assert torch.all(exits >= 1) and torch.all(exits <= conf.shape[1])

    return exits


def brier_score(conf: torch.Tensor, labels: torch.Tensor, mean: bool = False, C: float = 1.) -> torch.Tensor:
    """
    conf: (N, C)
    labels: (N,)

    returns:
        brier: (N,)
    
    """
    assert len(conf.shape) == 2
    assert len(labels.shape) == 1
    assert conf.shape[0] == labels.shape[0]

    # convert labels to one-hot
    labels = torch.nn.functional.one_hot(labels, num_classes=conf.shape[1])

    if mean:
        brier = torch.mean((conf - labels) ** 2, dim=1)
    else:
        brier = torch.sum((conf - labels) ** 2, dim=1)
    return C * brier


def hellinger_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    p: (N, C)
    q: (N, C)

    returns:
        hd: (N,)
    
    """
    assert len(p.shape) == 2
    assert len(q.shape) == 2
    assert p.shape == q.shape

    hd = torch.sqrt(torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2, dim=1)) / np.sqrt(2)
    return hd



def get_preds_per_exit(logits: torch.Tensor):
    L = logits.shape[1]
    preds = torch.argmax(logits, dim=2)
    return {l: preds[:, l] for l in range(L)}


def get_acc_per_exit(
    preds, targets: torch.Tensor
):
    L = len(preds)
    return [(targets == preds[l]).sum() / len(targets) for l in range(L)]