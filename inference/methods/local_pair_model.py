"""NumPy local text-pair computation; weights must be fitted inside the training scope.

No built-in vocabulary or weights. This module only operates on passed inputs;
the prospective fitting pipeline must learn vocabulary solely within its fit.
"""
from collections import Counter
from fractions import Fraction
import re
import numpy as np
from quantity_role_features import normalize
from quantity_typed_text import number_value
from round2_features import NUMBER

Q_LENGTH=64
C_LENGTH=192


def split_words(text,table):
    def replace(m):
        value=number_value(m.group())
        if value in table:return ' '+table[value]+' '
        if value is None:return ' unknown_number '
        if value in (0,1,2,10,12,60,100):return ' constant_'+str(int(value))+' '
        magnitude=min(12,int(np.log10(max(1.,abs(float(value))))))
        return ' '+('negative_' if value<0 else 'positive_')+'number_'+str(magnitude)+' '
    text=re.sub(r'(?<=[\d)])\s*-\s*(?=\d)', ' op_sub ', text)
    text=NUMBER.sub(replace,text)
    return re.findall(r'[a-z_][a-z_0-9]*|[+*/=()-]',text.lower())


def token_pairs(bank):
    questions=[];candidates=[]
    for row in bank:
        question,_=normalize(row['question']);table={}
        for m in list(NUMBER.finditer(question))[:32]:
            value=number_value(m.group())
            if value is not None and value not in table:table[value]='question_slot_'+str(len(table))
        q=split_words(question,table)
        q=q if len(q)<=Q_LENGTH else q[:32]+q[-32:]
        for candidate in row['candidates']:
            text,_=normalize(candidate['text']);c=split_words(text,table)
            c=c if len(c)<=C_LENGTH else c[:32]+c[-160:]
            questions.append(q);candidates.append(c)
    return questions,candidates


def vocabulary(questions,candidates,max_words=8000):
    if len(questions)!=len(candidates) or len(questions)%4:raise ValueError('Expected four pairs per question')
    counts=Counter(t for row in questions+candidates for t in row);df=Counter()
    for begin in range(0,len(questions),4):
        df.update(set(t for row in questions[begin:begin+4]+candidates[begin:begin+4] for t in row))
    words=sorted((t for t in counts if df[t]>=2),key=lambda t:(-counts[t],t))[:max_words-2]
    return {'<pad>':0,'<unknown>':1,**{t:i+2 for i,t in enumerate(sorted(words))}}


def encode(rows,vocab,length):
    result=np.zeros((len(rows),length),np.int32)
    for i,row in enumerate(rows):result[i,:min(length,len(row))]=[vocab.get(t,1) for t in row[:length]]
    return result


def random_weights(size,dimension=32,seed=17):
    rng=np.random.default_rng(seed)
    def matrix(a,b):return (rng.normal(size=(a,b))/np.sqrt(a)).astype(np.float32)
    embedding=matrix(size,dimension);embedding[0]=0
    return {'embedding':embedding,'conv_w':matrix(3*dimension,dimension),'conv_b':np.zeros(dimension,np.float32),
            'compare_w':matrix(4*dimension,dimension),'compare_b':np.zeros(dimension,np.float32),
            'aggregate_w':matrix(4*dimension,dimension),'aggregate_b':np.zeros(dimension,np.float32),
            'out_w':matrix(dimension,1),'out_b':np.zeros(1,np.float32)}


def softmax(x,valid):
    x=np.where(valid,x,-1e4);value=np.exp(x-x.max(axis=-1,keepdims=True))*valid
    return value/np.maximum(value.sum(axis=-1,keepdims=True),1e-12)


def convolution(ids,w):
    x=w['embedding'][ids];n,length,d=x.shape
    padded=np.pad(x,((0,0),(1,1),(0,0)))
    windows=np.concatenate((padded[:,:length],padded[:,1:length+1],padded[:,2:length+2]),axis=-1)
    return np.maximum(windows@w['conv_w']+w['conv_b'],0)*(ids!=0)[...,None]


def forward(qids,cids,w,interact=True,batch_size=64):
    if len(qids)!=len(cids):raise ValueError('Mismatched pairs')
    results=[]
    for begin in range(0,len(qids),batch_size):
        qi=qids[begin:begin+batch_size];ci=cids[begin:begin+batch_size];qm=qi!=0;cm=ci!=0
        q=convolution(qi,w);c=convolution(ci,w)
        if interact:
            qn=q/np.maximum(np.sqrt((q*q).sum(axis=-1,keepdims=True)),1e-6)
            cn=c/np.maximum(np.sqrt((c*c).sum(axis=-1,keepdims=True)),1e-6)
            e=(qn@cn.transpose(0,2,1))*np.float32(4.)
            qc=softmax(e,cm[:,None,:])@c
            cq=softmax(e.transpose(0,2,1),qm[:,None,:])@q
        else:
            qc=np.broadcast_to(c.sum(axis=1,keepdims=True)/np.maximum(cm.sum(axis=1,keepdims=True)[...,None].astype(np.float32),1),q.shape)
            cq=np.broadcast_to(q.sum(axis=1,keepdims=True)/np.maximum(qm.sum(axis=1,keepdims=True)[...,None].astype(np.float32),1),c.shape)
        compared=[]
        for x,aligned,mask in ((q,qc,qm),(c,cq,cm)):
            f=np.concatenate((x,aligned,np.abs(x-aligned),x*aligned),axis=-1)
            f=np.maximum(f@w['compare_w']+w['compare_b'],0)*mask[...,None]
            mean=f.sum(axis=1)/np.maximum(mask.sum(axis=1,keepdims=True).astype(np.float32),1)
            compared.extend((mean,f.max(axis=1)))
        pooled=np.concatenate(compared,axis=-1)
        h=np.maximum(pooled@w['aggregate_w']+w['aggregate_b'],0)
        results.append((h@w['out_w']+w['out_b']).ravel().astype(np.float32))
    return np.concatenate(results) if results else np.empty(0,np.float32)
