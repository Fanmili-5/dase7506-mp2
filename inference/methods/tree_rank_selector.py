"""Numeric tree ranker: current-input features and NumPy-only CPU prediction."""
import json
from pathlib import Path
import numpy as np
from round2_features import raw_dense
from quantity_features import extract
from oblivious_trees import evaluate
from selector import select_choices


def features(bank,arrays,view):
    raw=raw_dense(bank,arrays['embeddings'],{'dense':'v1_count'})
    x=np.concatenate((raw,extract(bank)['full']),axis=-1)
    if view=='raw':return x
    if view=='relative':
        center=x.mean(axis=1,keepdims=True,dtype=np.float64).astype(np.float32)
        return np.concatenate((x,x-center),axis=-1)
    raise ValueError('Unknown tree feature view')


def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float64)
    root=Path(checkpoint).resolve();cfg=json.loads((root/'config.json').read_text())
    if cfg['format']!='mp2-numeric-trees-v1':raise ValueError('Unexpected tree checkpoint')
    views={};results=[]
    members=cfg.get('members') or ['.']
    for member in members:
        child=(root/member).resolve()
        if not child.is_relative_to(root):raise ValueError('Tree path outside checkpoint')
        conf=json.loads((child/'config.json').read_text())
        if conf['format']!='mp2-numeric-trees-v1' or conf.get('members'):
            raise ValueError('Expected a leaf tree model')
        view=conf['view']
        if view not in views:views[view]=features(bank,arrays,view)
        x=views[view]
        if x.shape[-1]!=conf['n_features']:raise ValueError('Wrong tree input dimension')
        with np.load(child/'weights.npz',allow_pickle=False) as weights:
            results.append(evaluate(x.reshape(-1,x.shape[-1]),weights).reshape(len(bank),4))
    return np.mean(np.stack(results),axis=0)


def predict(bank,arrays,checkpoint):
    return select_choices(bank,candidate_scores(bank,arrays,checkpoint),'max') if bank else []
