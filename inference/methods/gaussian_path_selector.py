"""Frozen path/text gate and Gaussian combination; NumPy CPU inference only."""
import json
from pathlib import Path
import numpy as np
from selector import answers_for,select_choices
from quantity_shared_selector import candidate_scores as quantity
from local_pair_selector import candidate_scores as local_pair
from standalone_text_selector import candidate_scores as text
from final_paths_selector import candidate_scores as paths
from gaussian_selector import candidate_scores as gaussian
from gaussian_combined_selector import combine
from conditional_gate import prepare,score
from round4_selector import question_standardize

FORMAT='mp2-gaussian-path-fixed-v1'


def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float64)
    root=Path(checkpoint).resolve();cfg=json.loads((root/'config.json').read_text())
    if cfg['format']!=FORMAT:raise ValueError('Unexpected Gaussian/path format')
    embedding=np.asarray(arrays['embeddings'])
    if embedding.shape!=(len(bank),4,1024) or not np.isfinite(embedding).all():raise ValueError('Expected finite [N,4,1024] embeddings')
    aa=answers_for(bank);ix=np.asarray([i for i,a in enumerate(aa) if len(set(v for v in a if v is not None))>1],int)
    result=np.zeros((len(bank),4),np.float64)
    if not len(ix):return result
    rows=[bank[i] for i in ix];arr={'embeddings':embedding[ix]};valid=np.asarray([[v is not None for v in aa[i]] for i in ix],bool)
    def child(name):
        p=(root/name).resolve()
        if not p.is_relative_to(root):raise ValueError('Checkpoint escapes root')
        return p
    base=quantity(rows,arr,child('baseline'))
    local=(question_standardize(local_pair(rows,arr,child('pair')),valid)+question_standardize(base,valid))/2
    independent=question_standardize(base,valid)+.5*question_standardize(text(rows,arr,child('text')),valid)
    z,f=prepare(rows,np.stack([base,local,independent],axis=-1))
    with np.load(child('gate')/'weights.npz',allow_pickle=False) as w:gated=score(z,f,w)
    pathmix=(question_standardize(gated,valid)+cfg['path_beta']*question_standardize(paths(rows,arr,child('paths')),valid)).astype(np.float32)
    gaussianmix=combine(gaussian(rows,arr,child('gaussian')),base,valid)
    result[ix]=combine(gaussianmix,pathmix,valid)
    return result


def predict(bank,arrays,checkpoint):
    return select_choices(bank,candidate_scores(bank,arrays,checkpoint),'max') if bank else []
