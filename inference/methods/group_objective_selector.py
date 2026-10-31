"""NumPy-only inference for separately trained pairwise and answer-group models."""
import json
from pathlib import Path
import numpy as np
from relative_verifier_inputs import inputs,group_scores
from joint_verifier_selector import preprocess,forward,relation


def pool_values(scores,same,valid,weights,objective):
    if objective=='pairwise':return scores.mean(axis=1)
    best=np.where(same[:,None].astype(bool),scores[:,:,None,:],-1e4).max(axis=-1)
    beta=1/(1+np.exp(-float(weights['vote_logit'])))
    return (best+beta*np.log(np.maximum(same.sum(-1),1))[:,None]).mean(axis=1)


def candidate_scores(bank,arrays,checkpoint):
    root=Path(checkpoint).resolve();cfg=json.loads((root/'config.json').read_text())
    if cfg['format']!='mp2-objective-joint-v1':raise ValueError('Unexpected objective verifier')
    if not bank:return np.empty((0,4),np.float64)
    matrix,parsed=inputs(bank,arrays['embeddings']);same,valid=relation(parsed)
    with np.load(root/'preprocessing.npz',allow_pickle=False) as f:stats={k:f[k].astype(np.float64) for k in f.files}
    absolute,relative,scalar=preprocess(arrays['embeddings'],matrix[...,1024:],stats)
    prediction=np.zeros((len(bank),4),np.float64)
    for name in cfg['members']:
        path=(root/name).resolve()
        if not path.is_relative_to(root):raise ValueError('Checkpoint escapes root')
        with np.load(path,allow_pickle=False) as f:w={k:f[k] for k in f.files}
        for b in range(0,len(bank),32):
            raw=forward(absolute[b:b+32],relative[b:b+32],scalar[b:b+32],same[b:b+32],valid[b:b+32],w,'joint')
            prediction[b:b+32]+=pool_values(raw,same[b:b+32],valid[b:b+32],w,cfg['objective'])/len(cfg['members'])
    return group_scores(prediction,parsed,cfg.get('vote_bonus',0.))


def predict(bank,arrays,checkpoint):
    return [int(i) for i in candidate_scores(bank,arrays,checkpoint).argmax(axis=1)] if bank else []
