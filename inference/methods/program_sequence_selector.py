"""From-scratch conditional likelihood of existing candidate programs; no decoding."""
import json,math
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from program_sequence_data import question_tokens,question_values,candidate_programs
from gold_programs import CONSTANTS

OUTPUT=['PAD','BOS','EOS','ADD','SUB','MUL','DIV']+['Q'+str(i) for i in range(12)]+['C'+str(i) for i in sorted(CONSTANTS)]
OUTPUT_ID={s:i for i,s in enumerate(OUTPUT)}
RESERVED=['PAD','UNK']+['qslot_'+str(i) for i in range(12)]
MAX_QUESTION=192


def input_rows(bank):
    return [dict(question_tokens=question_tokens(r['question']),question_slots=len(question_values(r['question'])[0]),candidate_prefixes=[candidate_programs(r['question'],c['text'])['prefixes'] for c in r['candidates']]) for r in bank]


def vocabulary(rows,fit):
    counts=Counter(t for i in fit for t in rows[int(i)]['question_tokens'] if t not in RESERVED)
    return RESERVED+[t for t,n in sorted(counts.items(),key=lambda x:(-x[1],x[0]))[:6000-len(RESERVED)]]


def question_batch(rows,vocab,device):
    table={s:i for i,s in enumerate(vocab)};tokens=[r['question_tokens'][:MAX_QUESTION] or ['UNK'] for r in rows];lengths=torch.tensor([len(t) for t in tokens],dtype=torch.long);q=torch.zeros((len(rows),int(lengths.max())),dtype=torch.long,device=device)
    slots=torch.full((len(rows),12),-1,dtype=torch.long,device=device)
    for i,tt in enumerate(tokens):
        q[i,:len(tt)]=torch.tensor([table.get(t,1) for t in tt],device=device)
        for k,t in enumerate(tt):
            if t in RESERVED[2:]:slots[i,int(t.split('_')[1])]=k
    return q,lengths,slots


class ProgramModel(nn.Module):
    def __init__(self,nwords):
        super().__init__();self.words=nn.Embedding(nwords,48,padding_idx=0);self.encoder=nn.GRU(48,48,batch_first=True,bidirectional=True);self.program=nn.Embedding(len(OUTPUT),32,padding_idx=0);self.decoder=nn.GRU(32,96,batch_first=True);self.initial=nn.Linear(96,96);self.out=nn.Linear(192,len(OUTPUT));self.pointer=nn.Linear(192,96,bias=False);self.dropout=nn.Dropout(.15)
    def encode(self,q,lengths,slots):
        packed=nn.utils.rnn.pack_padded_sequence(self.dropout(self.words(q)),lengths.cpu(),batch_first=True,enforce_sorted=False);enc,_=self.encoder(packed);enc,_=nn.utils.rnn.pad_packed_sequence(enc,batch_first=True,total_length=q.shape[1]);mask=q!=0;mean=(enc*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True);return enc,mask,torch.tanh(self.initial(mean)),slots
    def loglikelihood(self,encoded,owners,programs):
        enc,mask,initial,slots=encoded;device=enc.device;owner=torch.tensor(owners,dtype=torch.long,device=device);seq=[[OUTPUT_ID[t] for t in p]+[OUTPUT_ID['EOS']] for p in programs];length=torch.tensor([len(p) for p in seq],device=device);size=int(length.max());target=torch.zeros((len(seq),size),dtype=torch.long,device=device)
        for i,p in enumerate(seq):target[i,:len(p)]=torch.tensor(p,device=device)
        previous=torch.cat([torch.full((len(seq),1),OUTPUT_ID['BOS'],device=device,dtype=torch.long),target[:,:-1]],dim=1);dec,_=self.decoder(self.dropout(self.program(previous)),initial[owner][None]);qenc=enc[owner];attention=torch.bmm(dec,qenc.transpose(1,2))/math.sqrt(96);attention=attention.masked_fill(~mask[owner,None,:],-1e4).softmax(-1);context=torch.bmm(attention,qenc);joint=torch.cat([dec,context],dim=-1);logits=self.out(self.dropout(joint));slot=slots[owner];slot_enc=qenc.gather(1,slot.clamp(min=0)[:,:,None].expand(-1,-1,96));p=torch.bmm(self.pointer(joint),slot_enc.transpose(1,2))/math.sqrt(96);logits[:,:,7:19]=logits[:,:,7:19]+p;logits[:,:,7:19]=logits[:,:,7:19].masked_fill(slot[:,None,:]<0,-1e4);logits[:,:,:2]=-1e4;token=logits.log_softmax(-1).gather(-1,target[:,:,None]).squeeze(-1);total=(token*(target!=0)).sum(1);return total,length


def grouped_scores(model,encoded,rows,refs=None):
    owners=[];programs=[];groups=[];reference_groups=[];seen={}
    def indices(i,variants):
        out=[]
        for p in variants:
            key=(i,tuple(p))
            if key not in seen:seen[key]=len(programs);owners.append(i);programs.append(p)
            out.append(seen[key])
        return out
    for i,r in enumerate(rows):
        groups.append([indices(i,variants) for variants in r['candidate_prefixes']]);reference_groups.append(indices(i,[] if refs is None else refs[i]))
    zero=encoded[0].sum()*0
    if programs:total,length=model.loglikelihood(encoded,owners,programs);means=total/length
    scores=[];support=[];losses=[]
    for cs,rr in zip(groups,reference_groups):
        scores.append(torch.stack([torch.logsumexp(means[ix],0)-math.log(len(ix)) if ix else zero for ix in cs]));support.append([bool(ix) for ix in cs])
        if rr:losses.append(-torch.logsumexp(total[rr],0)/length[rr].float().mean())
    return torch.stack(scores),torch.tensor(support,device=encoded[0].device),torch.stack(losses).mean() if losses else zero


def supported_residual(scores,support):
    # Missing programs get a neutral residual, never a hard exclusion.
    n=support.sum(1,keepdims=True);mu=(scores*support).sum(1,keepdims=True)/np.maximum(n,1);d=(scores-mu)*support;scale=np.sqrt((d*d).sum(1,keepdims=True)/np.maximum(n,1));return (d/np.maximum(scale,.2)*(n>=2)).astype(np.float32)


def export(model,vocab,bundle,extra=None):
    root=Path(bundle);root.mkdir(parents=True,exist_ok=False);conf=dict(format='mp2-program-sequence-v1',vocabulary=vocab,output_tokens=OUTPUT,**(extra or {}));(root/'config.json').write_text(json.dumps(conf,ensure_ascii=False,indent=2)+'\n');np.savez(root/'weights.npz',**{k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()})


def load(root,device='cpu'):
    root=Path(root);conf=json.loads((root/'config.json').read_text());assert conf['format']=='mp2-program-sequence-v1' and conf['output_tokens']==OUTPUT;model=ProgramModel(len(conf['vocabulary'])).to(device)
    with np.load(root/'weights.npz',allow_pickle=False) as z:model.load_state_dict({k:torch.tensor(z[k].copy(),device=device) for k in z.files})
    model.eval();return model,conf['vocabulary']


def score_rows(model,vocab,rows,device='cpu'):
    result=[]
    with torch.inference_mode():
        for start in range(0,len(rows),32):
            batch=rows[start:start+32];encoded=model.encode(*question_batch(batch,vocab,device));s,m,_=grouped_scores(model,encoded,batch);result.append(supported_residual(s.cpu().numpy(),m.cpu().numpy()))
    return np.concatenate(result) if result else np.empty((0,4),np.float32)


def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float32)
    torch.set_num_threads(1);root=Path(checkpoint).resolve();conf=json.loads((root/'config.json').read_text());rows=input_rows(bank)
    if conf.get('members'):
        scores=[]
        for name in conf['members']:
            child=(root/name).resolve();assert child.is_relative_to(root);model,vocab=load(child);scores.append(score_rows(model,vocab,rows))
        return np.mean(np.stack(scores),axis=0)
    model,vocab=load(root);return score_rows(model,vocab,rows)


def predict(bank,arrays,checkpoint):
    from selector import select_choices
    return select_choices(bank,candidate_scores(bank,arrays,checkpoint),'max') if bank else []
