"""Odd features of candidate differences; pure mechanical prototype, no fit."""
import numpy as np

PAIRS=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))


def differences(embeddings,scale):
    e=np.asarray(embeddings,np.float32);assert e.ndim==3 and e.shape[1]==4
    d=np.stack([e[:,i]-e[:,j] for i,j in PAIRS],axis=1);d=np.clip(d/scale,-5,5);d/=np.maximum(np.linalg.norm(d,axis=-1,keepdims=True),1e-12);return d.astype(np.float32)


def design(d,frequencies):
    d=np.asarray(d,np.float32);f=np.asarray(frequencies,np.float32);values=np.sin(d@f)*np.float32(np.sqrt(2/f.shape[1]));return values.astype(np.float32)


def aggregate(pair_scores):
    p=np.asarray(pair_scores,np.float32);assert p.ndim==2 and p.shape[1]==6;s=np.zeros((len(p),4),np.float32)
    for k,(i,j) in enumerate(PAIRS):s[:,i]+=p[:,k]/3;s[:,j]-=p[:,k]/3
    return s


def scores(embeddings,scale,frequencies,weights):
    return aggregate(design(differences(embeddings,scale),frequencies)@weights)
