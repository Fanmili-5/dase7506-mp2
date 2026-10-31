"""Linear ranking with candidate-varying answer and representation context."""

import json
from pathlib import Path

import numpy as np

from selector import answers_for, select_choices

FEATURE_NAMES = ("answer_vote_fraction", "same_answer_mean_cosine", "same_answer_max_cosine",
                 "different_answer_mean_cosine", "different_answer_max_cosine")


def features(bank, embeddings, mode):
    e = np.asarray(embeddings, dtype=np.float32)
    if e.shape != (len(bank), 4, 1024) or not np.isfinite(e).all():
        raise ValueError("Expected finite [N,4,1024] input embeddings")
    if mode not in ("none", "count", "pattern", "full"):
        raise ValueError("Unknown context feature mode")
    extra = np.zeros((len(bank), 4, len(FEATURE_NAMES)), np.float32)
    if mode == "none" or not bank:
        return np.concatenate((e, extra), axis=-1)
    unit = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-6)
    cosine = np.clip(np.einsum("nid,njd->nij", unit, unit), -1, 1)
    for q, answers in enumerate(answers_for(bank)):
        for i, answer in enumerate(answers):
            if answer is None:
                continue
            same = [j for j, a in enumerate(answers) if j != i and a is not None and a == answer]
            different = [j for j, a in enumerate(answers) if a is not None and a != answer]
            extra[q, i, 0] = (1 + len(same)) / 4
            if mode == "count":
                continue
            for peers, offset in ((same, 1), (different, 3)):
                if peers:
                    values = cosine[q, i, peers]
                    extra[q, i, offset:offset + 2] = (1, 1) if mode == "pattern" else (values.mean(), values.max())
    return np.concatenate((e, extra), axis=-1)


def candidate_scores(bank, arrays, checkpoint):
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config.get("family") != "context_linear" or config.get("objective") != "pairwise":
        raise ValueError("Expected a context linear ranking checkpoint")
    x = features(bank, arrays["embeddings"], config["feature_mode"])
    with np.load(checkpoint / "weights.npz", allow_pickle=False) as weights:
        values = ((x - weights["mean"]) / weights["scale"]) @ weights["w"] + weights["b"]
    scores = values.squeeze(-1)
    if scores.shape != (len(bank), 4) or not np.isfinite(scores).all():
        raise ValueError("Malformed candidate scores")
    return scores, config


def predict(bank, arrays, checkpoint):
    if not bank:
        return []
    root = Path(checkpoint).resolve()
    if (root / "ensemble.json").is_file():
        config = json.loads((root / "ensemble.json").read_text(encoding="utf-8"))
        if config.get("aggregation") != "mean_margin" or not config.get("members"):
            raise ValueError("Expected nonempty mean-margin ensemble")
        scores, modes = [], set()
        for relative in config["members"]:
            member = (root / relative).resolve()
            if not member.is_relative_to(root):
                raise ValueError("Member must be inside the checkpoint")
            values, c = candidate_scores(bank, arrays, member)
            modes.add(c["feature_mode"])
            scores.append(values)
        if len(modes) != 1:
            raise ValueError("Keep controlled ensembles within one feature mode")
        values = np.mean(np.stack(scores), axis=0)
    else:
        values, _ = candidate_scores(bank, arrays, root)
    return select_choices(bank, values, "max")
