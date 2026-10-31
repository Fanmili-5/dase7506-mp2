"""NumPy-only bidirectional text cross-encoder with no pretrained state."""
from collections import Counter
import json
import hashlib
from pathlib import Path
import numpy as np
from local_pair_model import token_pairs
from selector import select_choices,answers_for

PAD,UNK,CLS,SEP,MASK=range(5)
MAX_LENGTH=256


def vocabulary(fit_pairs,max_words=6000):
    count=Counter();frequency=Counter()
    for pairs in fit_pairs:
        words=[t for q,c in pairs for t in q+c];count.update(words);frequency.update(set(words))
    words=sorted((t for t in count if frequency[t]>=2),key=lambda t:(-count[t],t))[:max_words-5]
    return {'<pad>':PAD,'<unknown>':UNK,'<cls>':CLS,'<sep>':SEP,'<mask>':MASK,**{t:i+5 for i,t in enumerate(sorted(words))}}


def encode_pairs(questions,candidates,vocab,max_length=MAX_LENGTH):
    ids=np.zeros((len(questions),max_length),np.int32);segments=np.zeros_like(ids)
    for i,(q,c) in enumerate(zip(questions,candidates)):
        if len(q)>60:q=q[:24]+q[-36:]
        available=max_length-len(q)-3
        if len(c)>available:c=c[:min(32,available)]+c[-max(0,available-32):] if available>32 else c[-available:]
        tokens=[CLS]+[vocab.get(t,UNK) for t in q]+[SEP]+[vocab.get(t,UNK) for t in c]+[SEP]
        ids[i,:len(tokens)]=tokens;segments[i,len(q)+2:len(tokens)]=1
    return ids,segments


def layer_norm(x,g,b):
    mean=x.mean(axis=-1,keepdims=True);variance=np.mean((x-mean)**2,axis=-1,keepdims=True)
    return (x-mean)/np.sqrt(variance+np.float32(1e-5))*g+b


def forward(ids,segments,w,config,batch_size=32):
    results=[];heads=config['heads'];dimension=w['embedding'].shape[1];head_dim=dimension//heads
    for begin in range(0,len(ids),batch_size):
        ii=ids[begin:begin+batch_size];ss=segments[begin:begin+batch_size];length=max(1,int(np.count_nonzero(ii,axis=1).max()));ii=ii[:,:length];ss=ss[:,:length];valid=ii!=PAD
        x=(w['embedding'][ii]+w['position'][:length]+w['segment'][ss])*valid[...,None]
        for k in range(config['layers']):
            prefix=f'l{k}_';z=layer_norm(x,w[prefix+'ln1_g'],w[prefix+'ln1_b']);qkv=z@w[prefix+'qkv_w']+w[prefix+'qkv_b'];qkv=qkv.reshape(len(ii),length,3,heads,head_dim).transpose(2,0,3,1,4);q,key,value=qkv
            logits=(q@key.swapaxes(-1,-2))/np.float32(np.sqrt(head_dim));logits=np.where(valid[:,None,None,:],logits,np.float32(-1e4));p=np.exp(logits-logits.max(axis=-1,keepdims=True))*valid[:,None,None,:];p/=np.maximum(p.sum(axis=-1,keepdims=True),np.float32(1e-12))
            attention=(p@value).transpose(0,2,1,3).reshape(len(ii),length,dimension);x=(x+attention@w[prefix+'proj_w']+w[prefix+'proj_b'])*valid[...,None]
            z=layer_norm(x,w[prefix+'ln2_g'],w[prefix+'ln2_b']);feed=np.maximum(z@w[prefix+'ff1_w']+w[prefix+'ff1_b'],0)@w[prefix+'ff2_w']+w[prefix+'ff2_b'];x=(x+feed)*valid[...,None]
        x=layer_norm(x,w['final_g'],w['final_b']);results.append((x[:,0]@w['out_w']+w['out_b']).ravel())
    return np.concatenate(results).astype(np.float32) if results else np.empty(0,np.float32)


def _scores(root,pairs,encoded):
    cfg=json.loads((root/'config.json').read_text());assert cfg['format']=='mp2-tiny-cross-v1'
    if cfg.get('family')=='average':
        children=[(root/m).resolve() for m in cfg['members']]
        if any(not child.is_relative_to(root.resolve()) for child in children):raise ValueError('Invalid encoder member path')
        return np.mean(np.stack([_scores(child,pairs,encoded) for child in children]),axis=0)
    text=(root/'vocabulary.json').read_bytes();key=(hashlib.sha256(text).hexdigest(),cfg['max_length'])
    if key not in encoded:encoded[key]=encode_pairs(*pairs,json.loads(text),cfg['max_length'])
    ids,segments=encoded[key]
    with np.load(root/'weights.npz',allow_pickle=False) as w:scores=forward(ids,segments,w,cfg)
    return scores.reshape(-1,4)


def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float32)
    parsed=answers_for(bank);indices=[i for i,values in enumerate(parsed) if len({v for v in values if v is not None})>=2];out=np.zeros((len(bank),4),np.float32)
    if indices:out[indices]=_scores(Path(checkpoint).resolve(),token_pairs([bank[i] for i in indices]),{})
    return out


def predict(bank,arrays,checkpoint):return select_choices(bank,candidate_scores(bank,arrays,checkpoint),'max') if bank else []
