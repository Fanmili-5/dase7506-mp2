"""Validation-selected objective extension, retaining the previous frozen model."""
import json
from pathlib import Path
import numpy as np
from common import parse_answer
from direct_selected_verifier import standardize


def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float64)
    root=Path(checkpoint).resolve();cfg=json.loads((root/'config.json').read_text())
    if cfg['format']!='mp2-objective-extension-selected-v1':raise ValueError('Unexpected objective extension')
    valid=np.asarray([[parse_answer(c['text']) is not None for c in row['candidates']] for row in bank],bool)
    result=np.zeros((len(bank),4),np.float64)
    for item in cfg['components']:
        path=(root/item['path']).resolve()
        if not path.is_relative_to(root):raise ValueError('Component escapes checkpoint')
        if item['family']=='incumbent':
            from direct_selected_verifier import candidate_scores as score
        elif item['family']=='relative':
            from relative_verifier_selector import candidate_scores as score
        elif item['family']=='objective':
            from group_objective_selector import candidate_scores as score
        else:raise ValueError('Unknown component family')
        values=score(bank,arrays,path)
        if cfg['normalize_components']:values=standardize(values,valid)
        result+=item['weight']*values
    if not np.isfinite(result).all():raise ValueError('Nonfinite selection scores')
    return np.where(valid,result,-1e6)


def predict(bank,arrays,checkpoint):
    return [int(i) for i in candidate_scores(bank,arrays,checkpoint).argmax(axis=1)] if bank else []
