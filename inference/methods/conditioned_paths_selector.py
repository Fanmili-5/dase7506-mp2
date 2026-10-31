"""Native NumPy semantic-context by program-path compatibility scores."""
import json
from pathlib import Path
import numpy as np
from final_program_paths import rows
from final_paths_selector import encode_row
from selector import answers_for,select_choices


def pooled(embeddings):
    return np.asarray(embeddings,dtype=np.float64).mean(axis=1).astype(np.float32)


def contexts(embeddings,mean,components,scale):
    x=(pooled(embeddings)-mean)@components;x=x/scale
    x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    return np.concatenate((np.ones((len(x),1),np.float32),x),axis=1).astype(np.float32)


def score_rows(raw,vocab,idf,weights,context):
    assert len(raw)==4*len(context)
    result=[]
    for i,row in enumerate(raw):
        ix,v=encode_row(row,vocab,idf);result.append(float((v@weights[ix])@context[i//4]))
    return np.asarray(result,np.float32).reshape(-1,4)


def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float32)
    root=Path(checkpoint).resolve();cfg=json.loads((root/'config.json').read_text());assert cfg['format']=='mp2-conditioned-paths-v1';parsed=answers_for(bank);ix=[i for i,a in enumerate(parsed) if len({v for v in a if v is not None})>=2];scores=np.zeros((len(bank),4),np.float32)
    if ix:
        vocab=json.loads((root/'vocabulary.json').read_text())
        with np.load(root/'weights.npz',allow_pickle=False) as z:
            context=contexts(np.asarray(arrays['embeddings'])[ix],z['context_mean'],z['context_components'],z['context_scale']);scores[ix]=score_rows(rows([bank[i] for i in ix],True),vocab,z['idf'],z['weights'],context)
    return scores


def predict(bank,arrays,checkpoint):return select_choices(bank,candidate_scores(bank,arrays,checkpoint),'max') if bank else []
