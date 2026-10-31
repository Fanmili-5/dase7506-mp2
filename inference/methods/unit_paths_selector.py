"""Portable sparse path plus dense unit-feature ranking, NumPy only."""
import json
from pathlib import Path
import numpy as np
from selector import answers_for,select_choices
from final_program_paths import trace,features
from final_paths_selector import score_rows
from unit_propagation import prepare,analyze,FEATURE_NAMES


def transform(raw,mean,scale):
    return (np.clip((raw-mean)/scale,-5,5)/np.float32(np.sqrt(len(FEATURE_NAMES)))).astype(np.float32)


def score_cached(raw,units,vocab,w,use_units):
    s=score_rows(raw,vocab,w['idf'],w['weights'][:len(vocab)])
    if use_units:s=s+(transform(units,w['unit_mean'],w['unit_scale'])@w['weights'][len(vocab):]).astype(np.float32)
    return s.astype(np.float32)


def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float32)
    root=Path(checkpoint).resolve();cfg=json.loads((root/'config.json').read_text())
    if cfg['format']!='mp2-unit-paths-v1':raise ValueError('Unexpected unit path checkpoint')
    parsed=answers_for(bank);ix=[i for i,a in enumerate(parsed) if len({v for v in a if v is not None})>=2];out=np.zeros((len(bank),4),np.float32)
    if not ix:return out
    vocab=json.loads((root/'vocabulary.json').read_text());raw=[];units=[]
    for i in ix:
        row=bank[i];prepared=prepare(row['question']) if cfg['use_units'] else None;ur=[]
        for c in row['candidates']:
            p=trace(row['question'],c['text']);raw.append(features(row['question'],c['text'],False,p)[0])
            if cfg['use_units']:ur.append(analyze(row['question'],c['text'],p,prepared)[0])
        if cfg['use_units']:units.append(ur)
    with np.load(root/'weights.npz',allow_pickle=False) as w:out[ix]=score_cached(raw,np.asarray(units,np.float32),vocab,w,cfg['use_units'])
    return out


def predict(bank,arrays,checkpoint):return select_choices(bank,candidate_scores(bank,arrays,checkpoint),'max') if bank else []
