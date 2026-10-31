"""Remove each question's common vector before an existing Nyström map."""
import json
from pathlib import Path
import numpy as np
from round4b_selector import candidate_scores as kernel_scores
from selector import select_choices


def preprocess(embeddings,mode):
    e=np.asarray(embeddings,np.float32)
    if mode=='raw':return e
    if mode not in ('question_centered','question_centered_unit'):raise ValueError('Unknown precentering')
    e=e-e.mean(axis=1,keepdims=True)
    if mode=='question_centered_unit':e=e/np.maximum(np.linalg.norm(e,axis=-1,keepdims=True),1e-6)
    return e


def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float32)
    root=Path(checkpoint);cfg=json.loads((root/'config.json').read_text())
    if cfg.get('format')=='mp2-precenter-v1' and cfg.get('family')=='average':
        return np.mean(np.stack([candidate_scores(bank,arrays,root/m) for m in cfg['members']]),axis=0)
    assert cfg['format']=='mp2-round4b-v1' and cfg['family']=='nystrom'
    return kernel_scores(bank,{'embeddings':preprocess(arrays['embeddings'],cfg['embedding_preprocess'])},root)


def predict(bank,arrays,checkpoint):return select_choices(bank,candidate_scores(bank,arrays,checkpoint),'max') if bank else []
