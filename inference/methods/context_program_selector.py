"""Existing-program scoring conditioned on the supplied four-solution mean."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from program_sequence_selector import (ProgramModel,OUTPUT,input_rows,vocabulary,question_batch as text_batch,grouped_scores,supported_residual)


def raw_context(embeddings):
    e=np.asarray(embeddings);assert e.ndim==3 and e.shape[1:]==(4,1024) and np.isfinite(e).all();return e.astype(np.float64).mean(axis=1).astype(np.float32)


def context_statistics(raw,fit):
    x=raw[np.asarray(fit,int)].astype(np.float64);return x.mean(axis=0).astype(np.float32),np.maximum(x.std(axis=0),.05).astype(np.float32)


def attach_context(rows,raw,mean,scale,mode):
    assert mode in ('zero','provided_mean');x=np.clip((raw-mean)/scale,-5,5).astype(np.float32)
    if mode=='zero':x=np.zeros_like(x)
    return [dict(r,frozen_condition=c) for r,c in zip(rows,x)]


def question_batch(rows,vocab,device):
    return (*text_batch(rows,vocab,device),torch.as_tensor(np.stack([r['frozen_condition'] for r in rows]),device=device))


class ContextProgram(ProgramModel):
    def __init__(self,nwords,mode):
        super().__init__(nwords);assert mode in ('zero','provided_mean');self.mode=mode;self.context=nn.Linear(1024,96,bias=False);nn.init.zeros_(self.context.weight);self.register_buffer('context_mean',torch.zeros(1024));self.register_buffer('context_scale',torch.ones(1024))
    def encode(self,q,lengths,slots,condition):
        enc,mask,initial,slots=super().encode(q,lengths,slots)
        if self.mode=='zero':condition=condition*0
        shift=torch.tanh(self.context(condition));return enc+shift[:,None,:]*mask[:,:,None],mask,initial+shift,slots


def export(model,vocab,bundle,extra=None):
    root=Path(bundle);root.mkdir(parents=True,exist_ok=False);cfg=dict(format='mp2-context-program-v1',context_mode=model.mode,vocabulary=vocab,output_tokens=OUTPUT);cfg.update(extra or {});assert cfg['context_mode']==model.mode;(root/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n');np.savez(root/'weights.npz',**{k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()})


def load(root,device='cpu'):
    root=Path(root);cfg=json.loads((root/'config.json').read_text());assert cfg['format']=='mp2-context-program-v1' and cfg['output_tokens']==OUTPUT;model=ContextProgram(len(cfg['vocabulary']),cfg['context_mode']).to(device)
    with np.load(root/'weights.npz',allow_pickle=False) as z:model.load_state_dict({k:torch.tensor(z[k].copy(),device=device) for k in z.files})
    model.eval();return model,cfg['vocabulary']


def score_rows(model,vocab,rows,device='cpu'):
    result=[]
    with torch.inference_mode():
        for start in range(0,len(rows),32):
            batch=rows[start:start+32];encoded=model.encode(*question_batch(batch,vocab,device));s,m,_=grouped_scores(model,encoded,batch);result.append(supported_residual(s.cpu().numpy(),m.cpu().numpy()))
    return np.concatenate(result) if result else np.empty((0,4),np.float32)


def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float32)
    torch.set_num_threads(1);root=Path(checkpoint).resolve();cfg=json.loads((root/'config.json').read_text());rows=input_rows(bank);raw=raw_context(arrays['embeddings']);scores=[]
    for name in cfg.get('members',['.']):
        child=(root/name).resolve();assert child==root or child.is_relative_to(root);model,vocab=load(child);prepared=attach_context(rows,raw,model.context_mean.numpy(),model.context_scale.numpy(),model.mode);scores.append(score_rows(model,vocab,prepared))
    return np.mean(np.stack(scores),axis=0)


def predict(bank,arrays,checkpoint):
    from selector import select_choices
    return select_choices(bank,candidate_scores(bank,arrays,checkpoint),'max') if bank else []
