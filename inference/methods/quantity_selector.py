"""NumPy CPU inference: fixed linear verifier plus learned soft quantity signals."""
import json
from pathlib import Path
import numpy as np
from round2_features import raw_dense
from quantity_features import extract
from selector import select_choices

def residual(features,w,config):
    x=np.clip((features-w['feature_mean'])/w['feature_scale'],-5,5)
    if config['hidden']:x=np.maximum(x@w['hidden_w']+w['hidden_b'],0)
    return (x@w['out_w']+w['out_b']).squeeze(-1)

def candidate_scores(bank,arrays,checkpoint):
    root=Path(checkpoint);config=json.loads((root/'config.json').read_text())
    if config['format']!='mp2-quantity-v1':raise ValueError('Unexpected format')
    if not bank:return np.empty((0,4),np.float32)
    if config.get('members'):
        scores=[]
        for name in config['members']:
            child=(root/name).resolve()
            if not child.is_relative_to(root.resolve()):raise ValueError('Invalid member')
            scores.append(candidate_scores(bank,arrays,child))
        return np.mean(np.stack(scores),axis=0)
    with np.load(root/'weights.npz',allow_pickle=False) as w:
        raw=raw_dense(bank,arrays['embeddings'],{'dense':'v1_count'})
        x=(raw-w['mean'])/w['scale']
        base=(x@w['base_w']+w['base_b']).squeeze(-1)
        return base+residual(extract(bank)[config['features']],w,config)

def predict(bank,arrays,checkpoint):
    return select_choices(bank,candidate_scores(bank,arrays,checkpoint),'max')
