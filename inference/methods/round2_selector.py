"""Round-two CPU inference: no reference access and no fitting at prediction."""

import json
from pathlib import Path

import numpy as np

from round2_features import groups, raw_dense, text_rows
from selector import select_choices


def text_matrix(bank, config, root, arrays):
    from scipy import sparse
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    vocabulary = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    texts = text_rows(bank, config["text"])
    blocks = []
    for kind, gram in (("word", (1, 2)), ("char", (3, 5))):
        counter = CountVectorizer(analyzer=kind, ngram_range=gram,
            lowercase=True, vocabulary=vocabulary[kind], dtype=np.float32)
        transform = TfidfTransformer(norm="l2", sublinear_tf=True)
        transform.idf_ = arrays["idf_" + kind]
        blocks.append(transform.transform(counter.transform(texts)).astype(np.float32))
    return sparse.hstack(blocks, format="csr", dtype=np.float32)


def dense_matrix(bank, embeddings, config, weights, surface=None):
    x = raw_dense(bank, embeddings, config, surface)
    x = ((x - weights["mean"]) / weights["scale"]).astype(np.float32)
    if "pca" in weights:
        x = np.concatenate(((x[..., :1024] - weights["pca_offset"]) @ weights["pca"].T, x[..., 1024:]), axis=-1)
    return x


def group_scores(x, membership, valid, weights, config):
    h = np.maximum(x @ weights["phi_w"] + weights["phi_b"], 0)
    sizes = membership.sum(axis=-1)
    denominator = np.maximum(sizes[..., None], 1)
    if config.get("group_context", True):
        means = np.einsum("ngi,nid->ngd", membership, h) / denominator
        maxima = np.where(membership[..., None] > 0, h[:, None], -1e6).max(axis=2)
        maxima = np.where(sizes[..., None] > 0, maxima, 0)
        global_mean = (h * valid[..., None]).sum(axis=1) / np.maximum(valid.sum(axis=1, keepdims=True), 1)
        features = np.concatenate((means, maxima, np.repeat(global_mean[:, None], 4, axis=1), sizes[..., None] / 4), axis=-1)
        out = np.maximum(features @ weights["head_w"] + weights["head_b"], 0) @ weights["out_w"] + weights["out_b"]
        out = out.squeeze(-1)
    else:
        # Matched parameter count, same group objective. No pooled context is
        # supplied to the head; candidate logits are simply averaged per answer.
        vote = np.einsum("ng,ngi->ni", sizes, membership)[..., None] / 4
        features = np.concatenate((h, np.zeros_like(h), np.zeros_like(h), vote), axis=-1)
        candidate = (np.maximum(features @ weights["head_w"] + weights["head_b"], 0) @ weights["out_w"] + weights["out_b"]).squeeze(-1)
        out = np.einsum("ngi,ni->ng", membership, candidate) / np.maximum(sizes, 1)
    return np.einsum("ng,ngi->ni", out, membership)


def relative_probabilities(scores, valid):
    masked = np.where(valid, scores, -1e6)
    shifted = masked - masked.max(axis=1, keepdims=True)
    values = np.exp(np.clip(shifted, -60, 0)) * valid
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)


def fusion_features(member_scores, valid, vote):
    probabilities = np.stack([relative_probabilities(s, valid) for s in member_scores], axis=-1)
    confidence = probabilities.max(axis=1, keepdims=True)
    return np.concatenate((probabilities, probabilities * confidence, vote), axis=-1).astype(np.float32)


def candidate_scores(bank, arrays, checkpoint):
    root = Path(checkpoint).resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("format") != "mp2-round2-v1":
        raise ValueError("Unexpected round-two checkpoint format")
    if not bank:
        return np.empty((0, 4), np.float32)
    if config["family"] in ("average", "fusion"):
        scores = []
        for name in config["members"]:
            child = (root / name).resolve()
            if not child.is_relative_to(root):
                raise ValueError("Member outside checkpoint")
            scores.append(candidate_scores(bank, arrays, child))
        if config["family"] == "average":
            return np.mean(np.stack(scores), axis=0)
        _, valid, vote = groups(bank)
        if config["aggregation"] == "equal_probability":
            return np.mean(np.stack([relative_probabilities(s, valid) for s in scores]), axis=0)
        with np.load(root / "weights.npz", allow_pickle=False) as w:
            x = fusion_features(scores, valid, vote)
            return (x @ w["w"] + w["b"]).squeeze(-1)
    with np.load(root / "weights.npz", allow_pickle=False) as w:
        x = dense_matrix(bank, arrays["embeddings"], config, w)
        family = config["family"]
        if family == "torch":
            value = x @ w["w0"] + w["b0"]
            if "w1" in w:
                value = np.maximum(value, 0) @ w["w1"] + w["b1"]
            result = value.squeeze(-1)
        elif family == "group":
            membership, valid, _ = groups(bank)
            result = group_scores(x, membership, valid, w, config)
        elif family == "sparse":
            from scipy import sparse

            blocks = [sparse.csr_matrix(x.reshape(-1, x.shape[-1]))]
            if config.get("text"):
                blocks.append(text_matrix(bank, config, root, w))
            design = sparse.hstack(blocks, format="csr", dtype=np.float32)
            result = (np.asarray(design @ w["coef"]).ravel() + w["intercept"]).reshape(-1, 4)
        elif family == "xgb":
            import xgboost as xgb

            model = xgb.Booster()
            model.load_model(root / "model.ubj")
            model.set_param({"nthread": 1, "device": "cpu"})
            result = model.predict(xgb.DMatrix(x.reshape(-1, x.shape[-1]), nthread=1), output_margin=True).reshape(-1, 4)
        else:
            raise ValueError("Unknown round-two model family")
    if result.shape != (len(bank), 4) or not np.isfinite(result).all():
        raise ValueError("Malformed round-two candidate scores")
    return result.astype(np.float32)


def predict(bank, arrays, checkpoint):
    return select_choices(bank, candidate_scores(bank, arrays, checkpoint), "max") if bank else []
