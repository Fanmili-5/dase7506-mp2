"""Independent text scores: no fixed-vector or quantity offset in this head."""
import json
import hashlib
from pathlib import Path
import numpy as np
from local_pair_model import token_pairs,encode,forward,Q_LENGTH,C_LENGTH
from selector import select_choices

def _scores(root,pairs,encoded):
    cfg=json.loads((root/'config.json').read_text())
    if cfg['format']!='mp2-standalone-text-v1':raise ValueError('Unexpected text format')
    if cfg.get('members'):
        result=[]
        for name in cfg['members']:
            child=(root/name).resolve()
            if not child.is_relative_to(root):raise ValueError('Invalid member path')
            result.append(_scores(child,pairs,encoded))
        return np.mean(np.stack(result),axis=0)
    raw=(root/'vocabulary.json').read_bytes();key=hashlib.sha256(raw).hexdigest()
    if key not in encoded:
        vocab=json.loads(raw);encoded[key]=(encode(pairs[0],vocab,Q_LENGTH),encode(pairs[1],vocab,C_LENGTH))
    with np.load(root/'weights.npz',allow_pickle=False) as weights:
        w={k:weights[k] for k in weights.files}
        return forward(*encoded[key],w,interact=cfg['interact']).reshape(-1,4)

def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float32)
    return _scores(Path(checkpoint).resolve(),token_pairs(bank),{})

def predict(bank,arrays,checkpoint):
    return select_choices(bank,candidate_scores(bank,arrays,checkpoint),'max') if bank else []
