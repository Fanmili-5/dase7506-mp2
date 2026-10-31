"""CPU arithmetic-template scoring, without reference access or generation."""
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import re
import numpy as np
from common import parse_answer
from gold_programs import question_values, loads
from quantity_features import requested
from round2_features import NUMBER
from quantity_typed_text import word
from selector import select_choices


def features(question):
    values,meta=question_values(question);text=meta['normalized_question']
    table={v:'qslot_'+str(i) for i,v in enumerate(values)}
    def replace(m):
        try:return ' '+table.get(Fraction(m.group().replace(',','').rstrip('%')),'number')+' '
        except (ValueError,ZeroDivisionError):return ' number '
    text=NUMBER.sub(replace,text)
    tokens=[word(t) for t in re.findall(r'qslot_\d+|[a-zA-Z]+|[$%]',text.lower())][:256]
    result=['w:'+t for t in tokens]+['b:'+a+'_'+b for a,b in zip(tokens,tokens[1:])]
    result+=['ask:'+word(t) for t in re.findall(r'[a-zA-Z]+',requested(text).lower())[-24:]]
    for i,t in enumerate(tokens):
        if t.startswith('qslot_'):
            result.extend('near:'+t+'_'+v for v in tokens[max(0,i-3):i]+tokens[i+1:i+4])
    result.append('slot_count:'+str(len(values)))
    return result


def fit_design(questions,max_features=6000):
    rows=[features(q) for q in questions];df=Counter(t for r in rows for t in set(r))
    selected=sorted((t for t in df if df[t]>=2),key=lambda t:(-df[t],t))[:max_features]
    vocab={t:i for i,t in enumerate(sorted(selected))}
    idf=np.asarray([np.log((1+len(rows))/(1+df[t]))+1 for t in vocab],np.float32)
    return vocab,idf


def design(questions,vocab,idf):
    result=np.zeros((len(questions),len(vocab)),np.float32)
    for i,q in enumerate(questions):
        for t,count in Counter(features(q)).items():
            if t in vocab:result[i,vocab[t]]=(1+np.log(count))*idf[vocab[t]]
    result/=np.maximum(np.linalg.norm(result,axis=1,keepdims=True),1e-12)
    return result


def program_values(questions,programs):
    values=[question_values(q)[0] for q in questions]
    matrix=np.full((len(values),12),np.nan,np.float64)
    for i,row in enumerate(values):matrix[i,:min(12,len(row))]=list(map(float,row[:12]))
    cache={}
    def compute(p):
        if p in cache:return cache[p]
        if p[0]=='q':v=matrix[:,p[1]]
        elif p[0]=='c':v=np.full(len(values),float(p[1]))
        else:
            a,b=compute(p[1]),compute(p[2])
            with np.errstate(all='ignore'):
                if p[0]=='+':v=a+b
                elif p[0]=='-':v=a-b
                elif p[0]=='*':v=a*b
                elif p[0]=='/':v=np.divide(a,b,out=np.full_like(a,np.nan),where=b!=0)
                else:raise ValueError('Unknown program operation')
            v=np.where(np.isfinite(v)&(np.abs(v)<=1e12),v,np.nan)
        cache[p]=v;return v
    return np.stack([compute(p) for p in programs],axis=1)


def matches(values,answers):
    # This is a fallible internal feature, never a replacement official scorer.
    return np.isfinite(values)&np.isfinite(answers)&(np.abs(values-answers)<=1e-7+1e-9*np.abs(answers))


def from_weights(bank,vocab,programs,w):
    x=design([r['question'] for r in bank],vocab,w['idf'])
    logits=np.maximum(x@w['w0']+w['b0'],0)@w['w1']+w['b1']
    values=program_values([r['question'] for r in bank],programs);valid=np.isfinite(values)
    logits=np.where(valid,logits,-1e6);p=np.exp(np.clip(logits-logits.max(axis=1,keepdims=True),-60,0))*valid
    p/=np.maximum(p.sum(axis=1,keepdims=True),1e-12)
    answers=np.asarray([[float(Fraction(a)) if (a:=parse_answer(c['text'])) is not None else np.nan for c in row['candidates']] for row in bank])
    mass=np.sum(p[:,None,:]*matches(values[:,None,:],answers[:,:,None]),axis=-1)
    score=np.log(.01+mass);return (score-score.mean(axis=1,keepdims=True)).astype(np.float32)


def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float32)
    root=Path(checkpoint);cfg=json.loads((root/'config.json').read_text())
    assert cfg['format']=='mp2-gold-program-v1'
    if cfg.get('family')=='average':
        return np.mean(np.stack([candidate_scores(bank,arrays,root/m) for m in cfg['members']]),axis=0)
    vocab=json.loads((root/'vocabulary.json').read_text());programs=[loads(p) for p in json.loads((root/'programs.json').read_text())]
    with np.load(root/'weights.npz',allow_pickle=False) as w:return from_weights(bank,vocab,programs,w)


def predict(bank,arrays,checkpoint):
    return select_choices(bank,candidate_scores(bank,arrays,checkpoint),'max') if bank else []
