"""NumPy-only nonlinear comparisons of the six candidate differences."""
import json
from pathlib import Path
import numpy as np
from odd_pair_features import differences,design,aggregate
from selector import answers_for,select_choices


def pair_design(embeddings,scale,frequencies):
    d=differences(embeddings,scale)
    if frequencies.shape[1]:return np.concatenate((d,design(d,frequencies)),axis=-1)*np.float32(1/np.sqrt(2))
    return d


def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float32)
    root=Path(checkpoint).resolve();config=json.loads((root/'config.json').read_text());assert config['format']=='mp2-odd-pair-v1';e=np.asarray(arrays['embeddings'],np.float32);assert e.shape==(len(bank),4,1024) and np.isfinite(e).all();parsed=answers_for(bank);ix=[i for i,a in enumerate(parsed) if len({x for x in a if x is not None})>=2];scores=np.zeros((len(bank),4),np.float32)
    if ix:
        with np.load(root/'weights.npz',allow_pickle=False) as z:scores[ix]=aggregate(pair_design(e[ix],z['scale'],z['frequencies'])@z['weights'])
    return scores


def predict(bank,arrays,checkpoint):return select_choices(bank,candidate_scores(bank,arrays,checkpoint),'max') if bank else []
