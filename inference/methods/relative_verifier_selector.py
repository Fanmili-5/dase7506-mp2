"""New standalone, train-fitted verifier; no old checkpoints or predictions."""
import json
from pathlib import Path
import numpy as np
from relative_verifier_inputs import inputs, group_scores


def candidate_scores(bank, arrays, checkpoint):
    root = Path(checkpoint).resolve()
    config = json.loads((root / 'config.json').read_text())
    if config['format'] != 'mp2-standalone-relative-v1':
        raise ValueError('Unexpected independent verifier')
    x, parsed = inputs(bank, arrays['embeddings'])
    with np.load(root / 'weights.npz', allow_pickle=False) as weights:
        w, mu, sd = (weights[k].astype(np.float64) for k in ('w', 'mu', 'sd'))
        raw = ((x - mu) / sd) @ w + float(weights['b'])
    return group_scores(raw, parsed, config['vote_bonus'])


def predict(bank, arrays, checkpoint):
    return [int(i) for i in candidate_scores(bank, arrays, checkpoint).argmax(axis=1)] if bank else []
