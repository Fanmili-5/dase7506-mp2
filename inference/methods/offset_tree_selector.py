"""Current-question confidence/quantity correction of a frozen v2 baseline."""
import json
from pathlib import Path
import numpy as np
from round2_features import raw_dense,groups
from quantity_features import extract
from quantity_shared_selector import _scores as baseline_scores
from oblivious_trees import evaluate
from selector import select_choices

FORMAT='mp2-offset-tree-v1'


def feature_matrix(offsets,vote,valid,quantities,mode):
    b=np.asarray(offsets,np.float64)
    if b.ndim!=2 or b.shape[1]!=4 or not np.isfinite(b).all():raise ValueError('Expected finite four-candidate scores')
    center=b.mean(1,keepdims=True);ordered=np.sort(b,axis=1)
    p=np.exp(b-b.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    other=np.stack([np.max(np.delete(b,i,axis=1),axis=1) for i in range(4)],axis=1)
    common=np.stack((b.std(1),np.ptp(b,axis=1),ordered[:,-1]-ordered[:,-2],ordered[:,-2]-ordered[:,-3],
                     -(p*np.log(np.maximum(p,1e-30))).sum(1)),axis=-1)
    x=np.concatenate(((b-center)[...,None],(b-other)[...,None],p[...,None],np.asarray(vote,np.float64),
                      np.asarray(valid,np.float64)[...,None],np.repeat(common[:,None,:],4,axis=1)),axis=-1)
    if mode=='quantity':
        q=np.asarray(quantities['full'],np.float32)
        relative=q-q.mean(1,keepdims=True,dtype=np.float64).astype(np.float32)
        x=np.concatenate((x,q,relative),axis=-1)
    elif mode!='confidence':raise ValueError('Unknown offset-tree feature mode')
    if not np.isfinite(x).all():raise ValueError('Nonfinite correction input')
    return x.astype(np.float32)


def zero_weights():
    return dict(feature_indices=np.empty((0,0),np.int32),borders=np.empty((0,0),np.float32),
                depths=np.empty(0,np.int32),leaf_values=np.empty((0,1),np.float64),
                scale=np.asarray(1.,np.float64),bias=np.asarray(0.,np.float64))


def _child(root,name):
    p=(root/name).resolve()
    if not p.is_relative_to(root):raise ValueError('Checkpoint path outside root')
    return p


def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float64)
    root=Path(checkpoint).resolve();cfg=json.loads((root/'config.json').read_text())
    if cfg['format']!=FORMAT:raise ValueError('Unexpected offset tree checkpoint')
    raw=raw_dense(bank,arrays['embeddings'],{'dense':'v1_count'});q=extract(bank)
    b=baseline_scores(_child(root,cfg['baseline']),raw,q)
    _,valid,vote=groups(bank);views={};corrections=[]
    for name in cfg['heads']:
        child=_child(root,name);conf=json.loads((child/'config.json').read_text());mode=conf['mode']
        if mode not in views:views[mode]=feature_matrix(b,vote,valid,q,mode)
        x=views[mode]
        if x.shape[-1]!=conf['n_features']:raise ValueError('Unexpected correction dimension')
        with np.load(child/'weights.npz',allow_pickle=False) as w:
            corrections.append(evaluate(x.reshape(-1,x.shape[-1]),w).reshape(-1,4))
    if not corrections:raise ValueError('At least one correction head is required')
    return b.astype(np.float64)+np.mean(np.stack(corrections),axis=0)


def predict(bank,arrays,checkpoint):
    return select_choices(bank,candidate_scores(bank,arrays,checkpoint),'max') if bank else []
