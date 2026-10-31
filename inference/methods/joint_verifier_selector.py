"""Independent multi-member four-candidate verifier; NumPy-only deployment.

Architecture uses BatchEnsemble-style shared matrices/member scalings (TabM,
ICLR2025) and permutation-equivariant set attention (Lee et al., ICML2019).
This is an original compact task adaptation, not official TabM reproduction.
"""
import json
from pathlib import Path
import numpy as np
from relative_verifier_inputs import inputs, group_scores


def preprocess(embeddings, scalar, stats):
    absolute = ((np.asarray(embeddings, np.float64) - stats['mean']) / stats['scale']) @ stats['zca']
    relative = absolute - absolute.mean(axis=1, keepdims=True)
    relative /= np.sqrt(np.mean(relative**2, axis=(1,2), keepdims=True)) + 1e-6
    relative /= stats['relative_scale']
    scalar = (scalar - stats['scalar_mean']) / stats['scalar_scale']
    return tuple(np.clip(v, -6, 6).astype(np.float32) for v in (absolute, relative, scalar))


def norm(x, gamma, bias):
    centered = x - x.mean(axis=-1, keepdims=True)
    return centered / np.sqrt(np.mean(centered**2, axis=-1, keepdims=True) + 1e-5) * gamma + bias


def forward(absolute, relative, scalar, same, valid, weights, mode):
    w = weights
    a, r, s = (x[:, None] for x in (absolute, relative, scalar))
    h = ((a * w['ar'][None,:,None]) @ w['aw']) * w['as'][None,:,None]
    h += ((r * w['rr'][None,:,None]) @ w['rw']) * w['rs'][None,:,None]
    h += s @ w['sw'] + w['ib'][None,:,None]
    h = np.maximum(h, 0)
    if mode == 'joint':
        q, k, v = np.split(h @ w['qkv_w'] + w['qkv_b'], 3, axis=-1)
        attention = (q @ np.swapaxes(k,-1,-2)) / np.sqrt(h.shape[-1])
        attention += w['same_bias'][None,:,None,None] * same[:,None]
        attention = np.where(valid[:,None,None,:], attention, -1e4)
        attention = np.exp(attention - attention.max(axis=-1,keepdims=True))
        attention /= np.maximum(attention.sum(axis=-1,keepdims=True),1e-12)
        h = norm(h + (attention @ v) @ w['attn_w'] + w['attn_b'], w['n1_g'], w['n1_b'])
        hidden = np.maximum(h @ w['ff_w'] + w['ff_b'], 0)
        h = norm(h + hidden @ w['fo_w'] + w['fo_b'], w['n2_g'], w['n2_b'])
    else:
        h = np.maximum(((h*w['hr'][None,:,None]) @ w['hw'])*w['hs'][None,:,None]+w['hb'][None,:,None],0)
    return np.sum(h*w['out'][None,:,None],axis=-1)+w['out_bias'][None,:,None]


def relation(parsed):
    valid = np.asarray([[v is not None for v in row] for row in parsed],bool)
    same = np.asarray([[[a is not None and a==b for b in row] for a in row] for row in parsed],np.float32)
    return same, valid


def candidate_scores(bank, arrays, checkpoint):
    root = Path(checkpoint).resolve()
    cfg = json.loads((root/'config.json').read_text())
    if cfg['format']!='mp2-joint-standalone-v1':raise ValueError('Unexpected standalone joint verifier')
    if not bank:return np.empty((0,4),np.float64)
    matrix, parsed = inputs(bank, arrays['embeddings'])
    same, valid = relation(parsed)
    with np.load(root/'preprocessing.npz',allow_pickle=False) as f:stats={k:f[k].astype(np.float64) for k in f.files}
    absolute,relative,scalar=preprocess(arrays['embeddings'],matrix[...,1024:],stats)
    prediction=np.zeros((len(bank),4),np.float64)
    for name in cfg['members']:
        path=(root/name).resolve()
        if not path.is_relative_to(root):raise ValueError('Checkpoint escapes root')
        with np.load(path,allow_pickle=False) as f:w={k:f[k] for k in f.files}
        for b in range(0,len(bank),32):
            member=forward(absolute[b:b+32],relative[b:b+32],scalar[b:b+32],same[b:b+32],valid[b:b+32],w,cfg['mode'])
            prediction[b:b+32]+=member.mean(axis=1)/len(cfg['members'])
    return group_scores(prediction,parsed,cfg.get('vote_bonus',0.))


def predict(bank, arrays, checkpoint):
    return [int(i) for i in candidate_scores(bank,arrays,checkpoint).argmax(axis=1)] if bank else []
