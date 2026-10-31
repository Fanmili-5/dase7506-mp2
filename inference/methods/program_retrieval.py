"""Transfer fit-only arithmetic templates by lexical/quantity-role similarity."""
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import numpy as np
from common import parse_answer
from gold_programs import extract, dumps, loads, question_values
from program_selector import features, program_values, matches
from selector import select_choices, answers_for


def tokens(question, recipe):
    row=features(question)
    if recipe=='lexical':return [t for t in row if t.startswith(('w:','b:'))]
    if recipe=='role':return row
    raise ValueError('Unknown retrieval recipe')


def vector(question,vocab,idf,recipe):
    counts=Counter(tokens(question,recipe));items=sorted((vocab[t],n) for t,n in counts.items() if t in vocab)
    indices=np.asarray([i for i,n in items],np.int16)
    values=np.asarray([(1+np.log(n))*idf[i] for i,n in items],np.float32)
    values/=max(float(np.linalg.norm(values)),1e-12)
    return indices,values


def fit_index(bank,references,recipe,checkpoint):
    """The caller supplies only the current fit partition, never hold rows."""
    assert len(bank)==len(references) and bank
    questions=[row['question'] for row in bank];df=Counter(t for q in questions for t in set(tokens(q,recipe)))
    selected=sorted((t for t in df if df[t]>=2),key=lambda t:(-df[t],t))[:6000]
    vocab={t:i for i,t in enumerate(sorted(selected))}
    idf=np.asarray([np.log((1+len(bank))/(1+df[t]))+1 for t in vocab],np.float32)
    ptr=[0];columns=[];values=[];slotcounts=[];programs=[];program_index={};row_programs=[];owners=[]
    for i,(row,reference) in enumerate(zip(bank,references)):
        if 'question' in reference:assert reference['question']==row['question']
        result=extract(row['question'],reference['answer'])
        if not result['programs']:continue
        indices,data=vector(row['question'],vocab,idf,recipe)
        if not len(indices):continue
        ids=[]
        for program in result['programs']:
            text=dumps(program)
            if text not in program_index:program_index[text]=len(programs);programs.append(text)
            ids.append(program_index[text])
        owners.append(i);row_programs.append(ids);columns.extend(indices.tolist());values.extend(data.tolist());ptr.append(len(columns));slotcounts.append(result['question_slots'])
    if not owners:raise ValueError('No supported fit programs')
    root=Path(checkpoint);root.mkdir(parents=True,exist_ok=False)
    (root/'config.json').write_text(json.dumps({'format':'mp2-program-retrieval-v1','recipe':recipe,'neighbors':8,'min_similarity':.15,'similarity_power':4,'probability_floor':.05},sort_keys=True))
    (root/'vocabulary.json').write_text(json.dumps(vocab,sort_keys=True))
    (root/'programs.json').write_text(json.dumps({'programs':programs,'rows':row_programs},separators=(',',':')))
    np.savez(root/'index.npz',idf=idf,ptr=np.asarray(ptr,np.int32),columns=np.asarray(columns,np.int16),values=np.asarray(values,np.float32),slotcounts=np.asarray(slotcounts,np.int16))
    return {'indexed_fit_row_offsets':owners,'fitted_input_count':len(bank),'indexed_rows':len(owners),'program_count':len(programs),'vocabulary_size':len(vocab),'nonzero_features':len(columns),'no_ids_or_reference_answers_serialized':True,'asset_bytes':sum(p.stat().st_size for p in root.iterdir())}


def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float32)
    root=Path(checkpoint).resolve();cfg=json.loads((root/'config.json').read_text())
    if cfg['format']!='mp2-program-retrieval-v1':raise ValueError('Unexpected retrieval checkpoint')
    vocab=json.loads((root/'vocabulary.json').read_text());packed=json.loads((root/'programs.json').read_text());programs=[loads(p) for p in packed['programs']]
    with np.load(root/'index.npz',allow_pickle=False) as f:w={k:f[k].copy() for k in f.files}
    row_indices=np.repeat(np.arange(len(w['ptr'])-1),np.diff(w['ptr']));out=np.zeros((len(bank),4),np.float32)
    for i,(row,parsed) in enumerate(zip(bank,answers_for(bank))):
        if len({v for v in parsed if v is not None})<2:continue
        qvalues,_=question_values(row['question'])
        if not 0<len(qvalues)<=12:continue
        cols,vals=vector(row['question'],vocab,w['idf'],cfg['recipe']);query=np.zeros(len(vocab),np.float32);query[cols]=vals
        sims=np.bincount(row_indices,weights=query[w['columns']]*w['values'],minlength=len(w['ptr'])-1)
        eligible=np.flatnonzero((w['slotcounts']==len(qvalues))&(sims>=cfg['min_similarity']))
        if not len(eligible):continue
        neighbors=eligible[np.argsort(-sims[eligible],kind='stable')[:cfg['neighbors']]];weights=sims[neighbors]**cfg['similarity_power'];mass=np.zeros(4,np.float64)
        answers=np.asarray([float(Fraction(v)) if v is not None else np.nan for v in parsed])
        for neighbor,weight in zip(neighbors,weights):
            selected=[programs[j] for j in packed['rows'][int(neighbor)]];values=program_values([row['question']],selected)[0]
            mass+=weight*matches(values[None,:],answers[:,None]).mean(axis=1)
        mass/=max(float(weights.sum()),1e-12);score=np.log(cfg['probability_floor']+mass);score-=score.mean();out[i]=(score*min(1,float(sims[neighbors[0]]))**2).astype(np.float32)
    return out


def predict(bank,arrays,checkpoint):
    return select_choices(bank,candidate_scores(bank,arrays,checkpoint),'max') if bank else []
