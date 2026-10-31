"""Quantity CPU inference sharing reference-free features across ensemble members.

Only the current call's preprocessing is shared; no predictions or bank IDs are
retained. Per-member fitted normalization and clipping remain unchanged.
"""
import json
from pathlib import Path
import numpy as np
from quantity_role_features import extract
from quantity_selector import residual
from round2_features import raw_dense
from selector import select_choices

def _scores(root,raw,features):
    config=json.loads((root/'config.json').read_text())
    if config['format']!='mp2-role-v1':raise ValueError('Unexpected quantity format')
    if config.get('members'):
        scores=[]
        for name in config['members']:
            child=(root/name).resolve()
            if not child.is_relative_to(root):raise ValueError('Invalid ensemble member')
            scores.append(_scores(child,raw,features))
        return np.mean(np.stack(scores),axis=0)
    with np.load(root/'weights.npz',allow_pickle=False) as w:
        x=(raw-w['mean'])/w['scale']
        base=(x@w['base_w']+w['base_b']).squeeze(-1)
        return base+residual(features[config['features']],w,config)

def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float32)
    raw=raw_dense(bank,arrays['embeddings'],{'dense':'v1_count'})
    return _scores(Path(checkpoint).resolve(),raw,extract(bank))

def predict(bank,arrays,checkpoint):
    return select_choices(bank,candidate_scores(bank,arrays,checkpoint),'max') if bank else []
