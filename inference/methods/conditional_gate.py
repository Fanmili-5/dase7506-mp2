"""Small question-conditioned convex gate over frozen verifier views."""
import numpy as np
from selector import answers_for,select_choices
from round4_selector import question_standardize


def prepare(bank,scores):
    # scores: [questions, candidates, views], without labels or question IDs.
    if not bank:return np.empty((0,4,3),np.float32),np.empty((0,20),np.float32)
    answers=answers_for(bank);valid=np.asarray([[a is not None for a in row] for row in answers],bool)
    z=np.stack([question_standardize(scores[:,:,j],valid) for j in range(scores.shape[2])],axis=-1).astype(np.float32)
    features=[]
    for i,aa in enumerate(answers):
        groups=list(dict.fromkeys(a for a in aa if a is not None));counts=[aa.count(a) for a in groups]
        chosen=[select_choices([bank[i]],z[i:i+1,:,j],'max')[0] for j in range(scores.shape[2])]
        selected=[aa[j] for j in chosen];row=[len(groups)/4,max(counts,default=0)/4,min(counts,default=0)/4,sum(a is None for a in aa)/4]
        for j in range(scores.shape[2]):
            maxima=np.asarray([max(z[i,k,j] for k,a in enumerate(aa) if a==g) for g in groups],np.float32)
            top=np.sort(maxima);gap=float(top[-1]-top[-2]) if len(top)>1 else 0.
            if len(maxima):
                prob=np.exp(maxima-maxima.max());prob/=prob.sum();entropy=-float((prob*np.log(np.maximum(prob,1e-12))).sum())
            else:entropy=0.
            row.extend((gap,entropy,aa.count(selected[j])/4,float(selected[j]==selected[0]),float(z[i,chosen[j],j]-z[i,chosen[0],j])))
        row.append(float(selected[1]==selected[2]));features.append(row)
    return z,np.asarray(features,np.float32).reshape(len(bank),20)


def score(z,features,weights):
    x=(features-weights['mean'])/weights['scale'];logits=x@weights['w']+weights['b'];p=np.exp(logits-logits.max(axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
    return np.sum(z*p[:,None,:],axis=-1).astype(np.float32)
