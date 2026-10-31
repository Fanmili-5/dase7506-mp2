"""Fixed equal-weight combination, standardized inside the current question."""
import json
from pathlib import Path
import numpy as np
from gaussian_selector import candidate_scores as gaussian
from quantity_shared_selector import candidate_scores as quantity
from selector import answers_for,select_choices

FORMAT='mp2-gaussian-fixed-combination-v1'


def combine(a,b,valid):
    def standardize(s):
        s=np.asarray(s,np.float64);v=np.asarray(valid,bool);count=np.maximum(v.sum(1,keepdims=True),1)
        mean=(s*v).sum(1,keepdims=True)/count
        sd=np.sqrt((((s-mean)**2)*v).sum(1,keepdims=True)/count)
        return (s-mean)/np.maximum(sd,1e-6)
    return standardize(a)+standardize(b)


def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float64)
    root=Path(checkpoint).resolve();cfg=json.loads((root/'config.json').read_text())
    if cfg['format']!=FORMAT:raise ValueError('Unexpected combination format')
    q=(root/'gaussian').resolve();b=(root/'baseline').resolve()
    if not q.is_relative_to(root) or not b.is_relative_to(root):raise ValueError('Invalid checkpoint path')
    valid=np.asarray([[v is not None for v in a] for a in answers_for(bank)])
    return combine(gaussian(bank,arrays,q),quantity(bank,arrays,b),valid)


def predict(bank,arrays,checkpoint):
    return select_choices(bank,candidate_scores(bank,arrays,checkpoint),'max') if bank else []
