"""NumPy inference for a fitted binary Gaussian discriminant polynomial."""
import json
from pathlib import Path
import numpy as np
from selector import select_choices

FORMAT='mp2-shrunk-gaussian-v1'


def raw_features(embeddings,mode):
    x=np.asarray(embeddings,np.float64)
    if x.ndim!=3 or x.shape[1:]!=(4,1024) or not np.isfinite(x).all():
        raise ValueError('Expected finite [N,4,1024] embeddings')
    if mode=='raw':return x
    if mode=='relative':return x-x.mean(axis=1,keepdims=True)
    raise ValueError('Unknown Gaussian input mode')


def evaluate(x,weights):
    z=(x-weights['mean'])/weights['scale'];flat=z.reshape(-1,z.shape[-1])
    result=flat@weights['linear']+float(weights['bias'])
    if 'quadratic' in weights:
        result+=np.sum((flat@weights['quadratic'])*flat,axis=1)
    return result.reshape(x.shape[:-1])


def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float64)
    root=Path(checkpoint);config=json.loads((root/'config.json').read_text())
    if config['format']!=FORMAT:raise ValueError('Unexpected checkpoint format')
    x=raw_features(arrays['embeddings'],config['mode'])
    if len(x)!=len(bank):raise ValueError('Bank and array lengths differ')
    with np.load(root/'weights.npz',allow_pickle=False) as weights:result=evaluate(x,weights)
    if not np.isfinite(result).all():raise ValueError('Nonfinite discriminant scores')
    return result


def predict(bank,arrays,checkpoint):
    return select_choices(bank,candidate_scores(bank,arrays,checkpoint),'max') if bank else []
