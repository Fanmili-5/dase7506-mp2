"""CPU-only local text/quantity residual plus fixed linear score."""
import json
import hashlib
from pathlib import Path
import numpy as np
from local_pair_model import token_pairs,encode,forward,Q_LENGTH,C_LENGTH
from quantity_role_features import extract
from quantity_selector import residual
from round2_features import raw_dense
from selector import select_choices

def _scores(root,raw,features,pairs,encoded):
    cfg=json.loads((root/'config.json').read_text())
    if cfg['format']!='mp2-local-pair-v1':raise ValueError('Unexpected local pair format')
    if cfg.get('members'):
        result=[]
        for member in cfg['members']:
            child=(root/member).resolve()
            if not child.is_relative_to(root):raise ValueError('Invalid member path')
            result.append(_scores(child,raw,features,pairs,encoded))
        return np.mean(np.stack(result),axis=0)
    text=(root/'vocabulary.json').read_bytes();key=hashlib.sha256(text).hexdigest()
    if key not in encoded:
        vocab=json.loads(text);encoded[key]=(encode(pairs[0],vocab,Q_LENGTH),encode(pairs[1],vocab,C_LENGTH))
    with np.load(root/'weights.npz',allow_pickle=False) as weights:
        x=(raw-weights['mean'])/weights['scale']
        base=(x@weights['base_w']+weights['base_b']).squeeze(-1)
        number=residual(features[cfg['features']],weights,cfg)
        learned={k:weights['text_'+k] for k in ('embedding','conv_w','conv_b','compare_w','compare_b','aggregate_w','aggregate_b','out_w','out_b')}
        text_score=forward(*encoded[key],learned,interact=cfg['interact']).reshape(-1,4)
        return base+number+text_score

def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float32)
    raw=raw_dense(bank,arrays['embeddings'],{'dense':'v1_count'})
    return _scores(Path(checkpoint).resolve(),raw,extract(bank),token_pairs(bank),{})

def predict(bank,arrays,checkpoint):
    return select_choices(bank,candidate_scores(bank,arrays,checkpoint),'max') if bank else []
