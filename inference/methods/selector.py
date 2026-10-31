"""Reference-free candidate selection; neural checkpoints require only NumPy."""

import json
from pathlib import Path

import numpy as np

from common import parse_answer


def answers_for(bank):
    return [[parse_answer(c["text"]) for c in row["candidates"]] for row in bank]


def select_choices(bank, margins=None, aggregation="majority"):
    if aggregation not in ("majority", "max", "weighted_vote"):
        raise ValueError("Unknown aggregation")
    if margins is not None:
        margins = np.asarray(margins)
        if margins.shape != (len(bank), 4) or not np.isfinite(margins).all():
            raise ValueError("Expected finite [N, 4] candidate margins")
    if aggregation != "majority" and margins is None:
        raise ValueError("Learned aggregation needs candidate margins")
    choices = []
    for q, answers in enumerate(answers_for(bank)):
        groups = {}
        for i, answer in enumerate(answers):
            if answer is not None:
                groups.setdefault(answer, []).append(i)
        if not groups:
            choices.append(0)
            continue
        if aggregation == "majority" or len(groups) == 1:
            group = max(groups.values(), key=len)
            chosen = group[0]
        elif aggregation == "max":
            valid = [i for group in groups.values() for i in group]
            chosen = max(sorted(valid), key=lambda i: margins[q, i])
        else:
            # BCE margins only: this sum is a voting weight, not a posterior.
            probabilities = 1 / (1 + np.exp(-np.clip(margins[q], -60, 60)))
            group = max(groups.values(), key=lambda g: float(probabilities[g].sum()))
            chosen = max(group, key=lambda i: margins[q, i])
        choices.append(int(chosen))
    return choices


def candidate_margins(arrays, checkpoint):
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    embeddings = np.asarray(arrays["embeddings"], dtype=np.float32)
    if embeddings.ndim != 3 or embeddings.shape[1:] != (4, 1024):
        raise ValueError("Expected [N, 4, 1024] embeddings")
    if not np.isfinite(embeddings).all():
        raise ValueError("Nonfinite embeddings")
    with np.load(checkpoint / "weights.npz", allow_pickle=False) as weights:
        features = (embeddings.reshape(-1, 1024) - weights["mean"]) / weights["scale"]
        if config["family"] == "neural":
            values = features @ weights["w0"] + weights["b0"]
            if config["hidden"]:
                values = np.maximum(values, 0) @ weights["w1"] + weights["b1"]
            margins = values.reshape(-1, 4)
        elif config["family"] == "xgboost":
            import xgboost as xgb

            model = xgb.Booster()
            model.load_model(checkpoint / "model.ubj")
            model.set_param({"nthread": 1, "device": "cpu"})
            margins = model.predict(xgb.DMatrix(features, nthread=1), output_margin=True).reshape(-1, 4)
        else:
            raise ValueError("Unknown checkpoint family")
    return margins, config


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        return select_choices(bank)
    if not bank:
        return []
    margins, config = candidate_margins(arrays, checkpoint)
    if len(margins) != len(bank):
        raise ValueError("Bank and feature counts differ")
    aggregation = config.get("aggregation", "max")
    if aggregation == "weighted_vote" and config["objective"] != "bce":
        raise ValueError("Pairwise margins are not calibrated correctness probabilities")
    return select_choices(bank, margins, aggregation)
