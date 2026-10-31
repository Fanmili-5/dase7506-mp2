"""NumPy-only scoring from frozen final-calculation path features."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from final_program_paths import rows
from selector import answers_for,select_choices


def encode_row(tokens,vocab,idf):
    selected=sorted((vocab[t],n) for t,n in Counter(tokens).items() if t in vocab)
    ix=np.asarray([i for i,n in selected],np.int32);values=np.asarray([(1+np.log(n))*idf[i] for i,n in selected],np.float32)
    values/=max(float(np.linalg.norm(values)),1e-12)
    return ix,values


def score_rows(raw,vocab,idf,weights):
    return np.asarray([np.dot(values,weights[ix]) for ix,values in (encode_row(row,vocab,idf) for row in raw)],np.float32).reshape(-1,4)


def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float32)
    root=Path(checkpoint).resolve();cfg=json.loads((root/'config.json').read_text())
    if cfg['format']!='mp2-final-paths-v1':raise ValueError('Unexpected final paths format')
    parsed=answers_for(bank);ix=[i for i,v in enumerate(parsed) if len({a for a in v if a is not None})>=2];result=np.zeros((len(bank),4),np.float32)
    if ix:
        vocab=json.loads((root/'vocabulary.json').read_text())
        with np.load(root/'weights.npz',allow_pickle=False) as w:result[ix]=score_rows(rows([bank[i] for i in ix],cfg['roles']),vocab,w['idf'],w['weights'])
    return result


def predict(bank,arrays,checkpoint):return select_choices(bank,candidate_scores(bank,arrays,checkpoint),'max') if bank else []
