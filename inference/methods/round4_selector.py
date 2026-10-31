"""CPU-only kernel, contextual attention, and lexical ranking residuals."""

import json
from pathlib import Path

import numpy as np

from round2_features import raw_dense, groups
from round2_selector import text_matrix
from selector import select_choices


def raw_inputs(bank, arrays):
    return raw_dense(bank, arrays["embeddings"], {"dense": "v1_count", "embedding": True})


def standardized(raw, weights):
    return ((raw - weights["mean"]) / weights["scale"]).astype(np.float32)


def projection(x, weights):
    return (((x[..., :1024] - weights["pca_offset"]) @ weights["pca_components"].T)
            / weights["pca_scale"]).astype(np.float32)


def linear_inputs(x):
    out = x.copy(); out[..., :1024] /= np.sqrt(1024)
    return out


def stable_softmax(x, mask):
    values = np.where(mask, x, -1e6)
    values = np.exp(np.clip(values - values.max(axis=-1, keepdims=True), -60, 0)) * mask
    return values / np.maximum(values.sum(axis=-1, keepdims=True), 1e-12)


def attention_residual(z, vote_standardized, membership, valid, weights, config):
    inputs = np.concatenate((z, vote_standardized), axis=-1)
    h = np.tanh(inputs @ weights["encoder_w"] + weights["encoder_b"])
    q = h @ weights["query_w"] + weights["query_b"]
    k = h @ weights["key_w"] + weights["key_b"]
    v = h @ weights["value_w"] + weights["value_b"]
    same = np.einsum("ngi,ngj->nij", membership, membership)
    scores = np.einsum("nid,njd->nij", q, k) / np.sqrt(config["hidden"])
    scores += weights["same_bias"] * same
    mask = np.broadcast_to(valid[:, None, :], scores.shape).copy()
    if not config["context"]:
        mask &= np.eye(4, dtype=bool)[None]
    attention = stable_softmax(scores, mask)
    context = h + np.einsum("nij,njd->nid", attention, v)
    hidden = np.maximum(context @ weights["hidden_w"] + weights["hidden_b"], 0)
    return (hidden @ weights["out_w"] + weights["out_b"]).squeeze(-1)


def question_standardize(scores, valid):
    count = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    mean = (scores * valid).sum(axis=1, keepdims=True) / count
    centered = np.where(valid, scores - mean, 0)
    deviation = np.sqrt((centered * centered).sum(axis=1, keepdims=True) / count)
    return centered / np.maximum(deviation, 1e-6)


def candidate_scores(bank, arrays, checkpoint):
    root = Path(checkpoint).resolve()
    config = json.loads((root / "config.json").read_text())
    if config.get("format") != "mp2-round4-v1":
        raise ValueError("Unexpected fourth-round checkpoint format")
    if not bank:
        return np.empty((0, 4), np.float32)
    if config["family"] in ("average", "blend"):
        members = []
        for name in config["members"]:
            child = (root / name).resolve()
            if not child.is_relative_to(root):
                raise ValueError("Member outside checkpoint")
            if config["family"] == "blend" and name == "baseline":
                from context_selector import candidate_scores as baseline_scores
                score, _ = baseline_scores(bank, arrays, child)
            else:
                score = candidate_scores(bank, arrays, child)
            members.append(score)
        if config["family"] == "average":
            return np.mean(np.stack(members), axis=0)
        _, valid, _ = groups(bank)
        return np.mean(np.stack([question_standardize(s, valid) for s in members]), axis=0).astype(np.float32)
    raw = raw_inputs(bank, arrays)
    with np.load(root / "weights.npz", allow_pickle=False) as w:
        x = standardized(raw, w)
        family = config["family"]
        if family == "kernel":
            xx = linear_inputs(x).reshape(-1, x.shape[-1])
            result = xx @ w["linear_w"]
            if config["gamma_multiplier"]:
                z = projection(x, w).reshape(-1, len(w["pca_scale"]))
                support = w["support"]
                norms = np.sum(support * support, axis=1)
                for start in range(0, len(z), 256):
                    chunk = z[start:start + 256]
                    distances = np.maximum(np.sum(chunk * chunk, axis=1, keepdims=True) + norms[None] - 2 * chunk @ support.T, 0)
                    result[start:start + len(chunk)] += np.exp(-float(w["gamma"]) * distances) @ w["alpha"]
            result = result.reshape(-1, 4)
        else:
            result = (x @ w["base_w"] + w["base_b"]).squeeze(-1)
            if family == "attention":
                membership, valid, _ = groups(bank)
                result += attention_residual(projection(x, w), x[..., 1024:1025], membership, valid, w, config)
            elif family == "text":
                result += np.asarray(text_matrix(bank, config, root, w) @ w["text_coef"]).reshape(-1, 4)
            else:
                raise ValueError("Unknown fourth-round model family")
    if result.shape != (len(bank), 4) or not np.isfinite(result).all():
        raise ValueError("Malformed fourth-round scores")
    return result.astype(np.float32)


def predict(bank, arrays, checkpoint):
    return select_choices(bank, candidate_scores(bank, arrays, checkpoint), "max") if bank else []
