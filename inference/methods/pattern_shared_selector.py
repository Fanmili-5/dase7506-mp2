"""Answer-group corrections with shared and observable-pattern-specific weights."""
import json
from pathlib import Path
import numpy as np
from selector import answers_for, select_choices
from quantity_features import extract
from quantity_shared_selector import _scores
from round2_features import raw_dense

FORMAT = 'mp2-pattern-shared-v1'
# Complete fixed partitions of 0..4 valid candidates. No label-dependent routing.
PATTERNS = [(), (1,), (2,), (1, 1), (3,), (2, 1), (1, 1, 1),
            (4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1)]


def grouped_features(bank, embeddings, quantities):
    e = np.asarray(embeddings, np.float64)
    q = np.asarray(quantities['full'], np.float64)
    if e.shape != (len(bank), 4, 1024) or q.shape != (len(bank), 4, 212):
        raise ValueError('Unexpected input shape')
    if not np.isfinite(e).all() or not np.isfinite(q).all():
        raise ValueError('Nonfinite input')
    x = np.zeros((len(bank), 4, 1237), np.float64)
    mask = np.zeros((len(bank), 4), bool)
    membership = np.full((len(bank), 4), -1, np.int32)
    tasks = np.zeros(len(bank), np.int32)
    for i, answers in enumerate(answers_for(bank)):
        groups = {}
        for j, answer in enumerate(answers):
            if answer is not None:
                groups.setdefault(answer, []).append(j)
        tasks[i] = PATTERNS.index(tuple(sorted(map(len, groups.values()), reverse=True)))
        for g, indices in enumerate(groups.values()):
            mask[i, g] = True
            membership[i, indices] = g
            x[i, g, :1024] = e[i, indices].mean(0)
            x[i, g, 1024:1236] = q[i, indices].mean(0)
            x[i, g, -1] = len(indices) / 4
    return x, mask, membership, tasks


def group_offsets(scores, membership):
    out = np.zeros_like(scores, dtype=np.float64)
    for i, row in enumerate(membership):
        for g in range(4):
            ix = np.flatnonzero(row == g)
            if len(ix):
                out[i, g] = np.max(scores[i, ix])
    return out


def correction(x, tasks, weights):
    z = np.clip((x - weights['mean']) / weights['scale'], -5, 5)
    return np.sum(z * (weights['shared'][None, None, :] + weights['deviations'][tasks, None, :]), axis=-1)


def candidate_scores(bank, arrays, checkpoint):
    if not bank:
        return np.empty((0, 4), np.float64)
    root = Path(checkpoint).resolve()
    config = json.loads((root / 'config.json').read_text())
    if config['format'] != FORMAT:
        raise ValueError('Unexpected format')
    base_root = (root / config['baseline']).resolve()
    if not base_root.is_relative_to(root):
        raise ValueError('Invalid baseline path')
    quantities = extract(bank)
    raw = raw_dense(bank, arrays['embeddings'], {'dense': 'v1_count'})
    base = _scores(base_root, raw, quantities)
    x, mask, membership, tasks = grouped_features(bank, arrays['embeddings'], quantities)
    with np.load(root / 'weights.npz', allow_pickle=False) as weights:
        delta = correction(x, tasks, weights)
    result = np.asarray(base, np.float64).copy()
    for i, row in enumerate(membership):
        for j, g in enumerate(row):
            if g >= 0:
                result[i, j] += delta[i, g]
    return result


def predict(bank, arrays, checkpoint):
    return select_choices(bank, candidate_scores(bank, arrays, checkpoint), 'max') if bank else []
