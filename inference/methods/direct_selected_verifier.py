"""Frozen validation-selected independent models, with optional old comparator.

Only current inputs are evaluated. No prediction caches or question-ID lookups.
"""
import json
from pathlib import Path
import numpy as np
from common import parse_answer


def standardize(score, valid):
    count=np.maximum(valid.sum(axis=1,keepdims=True),1)
    mean=np.where(valid,score,0).sum(axis=1,keepdims=True)/count
    centered=np.where(valid,score-mean,0)
    return centered/np.maximum(np.sqrt((centered**2).sum(axis=1,keepdims=True)/count),1e-6)


def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float64)
    root=Path(checkpoint).resolve();cfg=json.loads((root/'config.json').read_text())
    if cfg['format']!='mp2-direct-selected-verifier-v1':raise ValueError('Unexpected selected verifier')
    valid=np.asarray([[parse_answer(c['text']) is not None for c in row['candidates']] for row in bank],bool)
    result=np.zeros((len(bank),4),np.float64)
    for component in cfg['components']:
        path=(root/component['path']).resolve()
        if not path.is_relative_to(root):raise ValueError('Component path escapes checkpoint')
        if component['family']=='relative':
            from relative_verifier_selector import candidate_scores as score
        elif component['family']=='neural':
            from joint_verifier_selector import candidate_scores as score
        elif component['family']=='old_comparator':
            from gaussian_path_selector import candidate_scores as score
        else:raise ValueError('Unknown verifier component')
        value=score(bank,arrays,path)
        if cfg['normalize_components']:value=standardize(value,valid)
        result+=component['weight']*value
    if not np.isfinite(result).all():raise ValueError('Nonfinite verifier output')
    return np.where(valid,result,-1e6)


def predict(bank,arrays,checkpoint):
    return [int(i) for i in candidate_scores(bank,arrays,checkpoint).argmax(axis=1)] if bank else []
