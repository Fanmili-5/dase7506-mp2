"""Frozen existing ensemble with one train-fitted soft rate penalty."""
import json
from pathlib import Path
import numpy as np
from gaussian_path_selector import candidate_scores as parent_scores
from selector import select_choices
from explicit_rate_consistency import feature_matrix


def candidate_scores(bank, arrays, checkpoint):
    root = Path(checkpoint).resolve()
    config = json.loads((root / 'rate_config.json').read_text())
    if config['format'] != 'mp2-explicit-rate-correction-v1':
        raise ValueError('Unexpected rate checkpoint format')
    theta = float(config['theta'])
    if not np.isfinite(theta) or theta < 0:
        raise ValueError('Invalid rate coefficient')
    old = parent_scores(bank, arrays, root / 'parent')
    return old - theta * feature_matrix(bank) if theta else old


def predict(bank, arrays, checkpoint):
    return select_choices(bank, candidate_scores(bank, arrays, checkpoint), 'max') if bank else []
