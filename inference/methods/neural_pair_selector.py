"""NumPy odd neural candidate comparisons; role inputs shared across members."""
import json
from pathlib import Path
import numpy as np
from odd_pair_features import PAIRS,aggregate
from quantity_role_features import extract
from selector import answers_for,select_choices


def block(raw,scale):
    d=np.stack([raw[:,i]-raw[:,j] for i,j in PAIRS],axis=1);d=np.clip(d/scale,-5,5);d/=np.maximum(np.linalg.norm(d,axis=-1,keepdims=True),1e-12);return d*np.float32(np.sqrt(raw.shape[-1]))


def inputs(embeddings,role_raw,embedding_scale,role_scale):
    e=block(np.asarray(embeddings,np.float32),embedding_scale)
    return np.concatenate((e,block(role_raw,role_scale)),axis=-1) if role_raw is not None else e


def score(root,embeddings,role_raw):
    config=json.loads((root/'config.json').read_text());assert config['format']=='mp2-neural-pair-v1'
    if config.get('members'):
        members=[]
        for name in config['members']:
            child=(root/name).resolve();assert child.is_relative_to(root);members.append(score(child,embeddings,role_raw))
        return np.mean(np.stack(members),axis=0)
    with np.load(root/'weights.npz',allow_pickle=False) as w:
        x=inputs(embeddings,role_raw,w['embedding_scale'],w['role_scale']);p=x@w['skip_w']+np.tanh(x@w['hidden_w'])@w['out_w'];return aggregate(p)


def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float32)
    root=Path(checkpoint).resolve();config=json.loads((root/'config.json').read_text());e=np.asarray(arrays['embeddings'],np.float32);assert e.shape==(len(bank),4,1024) and np.isfinite(e).all();parsed=answers_for(bank);ix=[i for i,a in enumerate(parsed) if len({v for v in a if v is not None})>=2];result=np.zeros((len(bank),4),np.float32)
    if ix:
        role_raw=extract([bank[i] for i in ix])['role_all'] if config['use_roles'] else None;result[ix]=score(root,e[ix],role_raw)
    return result


def predict(bank,arrays,checkpoint):return select_choices(bank,candidate_scores(bank,arrays,checkpoint),'max') if bank else []
