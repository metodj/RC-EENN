import pickle
import numpy as np

from typing import Tuple


def read_img_cls_data(path: str, model: str, dataset: str) -> Tuple[np.array, np.array]:
    with open(f"{path}/{model}/{dataset}.p", "rb") as f:
        data = pickle.load(f)

    if len(data) == 2:
        logits, targets = data
    else:
        logits, targets, _ = data

    logits = np.transpose(logits, (1, 0, 2))
    targets = targets.astype(int)

    return logits, targets


def img_cls_main(
    model: str,
    dataset: str,
    path: str,
    lambdas: np.array,
    risk_type: str,
    non_negative: bool = False,
) -> Tuple[np.array, np.array]:

    logits, targets = read_img_cls_data(path, model, dataset)

    preds = get_preds_per_exit(logits)

    # acc = get_acc_per_exit(preds, targets)
    # print(logits.shape, targets.shape)
    # print(acc)

    conf = conf_measures(logits)
    preds = np.argmax(logits, axis=2)
    preds_full = preds[:, -1]

    lam_losses, lam_exits = [], []
    for l in lambdas:
        exits = compute_exits(conf, threshold=l)
        preds_ee = preds[np.arange(len(preds)), exits - 1]

        if "confidence" in risk_type:
            logits_ee = logits[np.arange(len(logits)), exits - 1]
            probs_ee = softmax(logits_ee, axis=1)
            probs_full = softmax(logits[:, -1], axis=1)
            outputs_early = probs_ee
            outputs_full = probs_full
        else:
            outputs_early = preds_ee
            outputs_full = preds_full

        gt_labels = targets if risk_type in {"prediction-gt-gap", "confidence-brier"} else None

        _, losses = compute_risk(
            outputs_early,
            outputs_full,
            criterion=risk_type,
            gt_labels=gt_labels,
            non_negative=non_negative,
        )

        lam_losses.append(losses)
        lam_exits.append(exits)

    return np.array(lam_losses), np.array(lam_exits)



def compute_risk(
    output_early: np.ndarray,
    output_full: np.ndarray,
    criterion: str,
    gt_labels: np.ndarray = None,
    non_negative: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    assert output_early.shape == output_full.shape, "Prediction shapes must match."
    if criterion in [
        "confidence-brier",
        "confidence-hellinger",
        "confidence-brier-top-pred",
    ]:
        assert output_early.ndim == 2, "Prediction must be 2D (N, C)."
    else:
        assert output_early.ndim == 1, "Prediction must be 1D (N, )."
    if "confidence" in criterion:
        assert 0 <= output_full.min() and output_full.max() <= 1
        assert 0 <= output_early.min() and output_early.max() <= 1

    n = output_full.shape[0]

    if criterion == "prediction-consistency":
        loss = 1.0 - (output_full == output_early).astype(np.float32)

    elif criterion == "prediction-gt-gap":
        assert gt_labels is not None, "Ground truth labels must be provided."
        loss = (output_early != gt_labels).astype(np.float32) - (
            output_full != gt_labels
        ).astype(np.float32)

    elif criterion == "confidence-hellinger":
        loss = hellinger_distance(output_early, output_full)

    elif criterion == "confidence-brier":
        assert gt_labels is not None, "Ground truth labels must be provided."
        loss = brier_score(output_early, gt_labels) - brier_score(
            output_full, gt_labels
        )

    elif criterion == "confidence-brier-top-pred":
        loss = brier_score(output_early, output_full.argmax(axis=1)) - brier_score(
            output_full, output_full.argmax(axis=1)
        )

    else:
        raise ValueError("Invalid risk measure.")

    if non_negative:
        loss = np.maximum(loss, 0.0)

    risk = loss.mean()

    assert risk <= 1

    return risk, loss


def conf_measures(
    input_tensor: np.ndarray,
    input_type: str = "logits",
    conf_measure: str = "top_softmax",
) -> np.ndarray:
    assert len(input_tensor.shape) == 3
    assert input_type in ["logits", "probs"]

    if conf_measure == "top_softmax":
        if input_type == "logits":
            input_tensor = softmax(input_tensor, axis=2)
        conf = np.max(input_tensor, axis=2)

    else:
        raise ValueError(f"conf_measure: {conf_measure} not implemented")

    return conf


def compute_exits(conf: np.ndarray, threshold: float) -> np.ndarray:
    assert len(conf.shape) == 2
    assert 0 <= threshold <= 1

    mask = (conf > threshold).astype(np.int32)
    exits = np.argmax(mask, axis=1)
    rows_with_no_threshold = np.sum(mask, axis=1) == 0
    exits[rows_with_no_threshold] = mask.shape[1] - 1
    exits = exits + 1

    assert exits.shape == (conf.shape[0],)
    assert np.all(exits >= 1) and np.all(exits <= conf.shape[1])

    return exits


def brier_score(
    conf: np.ndarray, labels: np.ndarray, mean: bool = False, C: float = 1.0
) -> np.ndarray:
    assert len(conf.shape) == 2
    assert len(labels.shape) == 1
    assert conf.shape[0] == labels.shape[0]

    labels_one_hot = np.eye(conf.shape[1])[labels]

    if mean:
        brier = np.mean((conf - labels_one_hot) ** 2, axis=1)
    else:
        brier = np.sum((conf - labels_one_hot) ** 2, axis=1)
    return C * brier


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    assert len(p.shape) == 2
    assert len(q.shape) == 2
    assert p.shape == q.shape

    hd = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2, axis=1)) / np.sqrt(2)
    return hd


def get_preds_per_exit(logits: np.ndarray):
    L = logits.shape[1]
    preds = np.argmax(logits, axis=2)
    return {l: preds[:, l] for l in range(L)}


def get_acc_per_exit(preds, targets: np.ndarray):
    L = len(preds)
    return [(targets == preds[l]).sum() / len(targets) for l in range(L)]


def softmax(x, axis=None):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def max_exit(model: str, dataset: str, path: str,) -> int:
    with open(f"{path}/{model}/{dataset}.p", "rb") as f:
        data = pickle.load(f)

    if len(data) == 2:
        logits, _ = data
    else:
        logits, _, _ = data

    return logits.shape[0]
