"""Select the completed independent models using official validation only."""
import os
for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','GOTO_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'runs/tj_rate_final_20261031_v1/snapshot_v2'
sys.path[:0]=[str(ROOT/'scripts'),str(OLD/'methods'),str(ROOT/'methods'),str(ROOT/'code')]
import round2_guard
round2_guard.install(allow_validation=True,offline=False)
import numpy as np
from common import load_inputs,read_jsonl,parse_answer,DATA,PROTOCOL
from relative_verifier_inputs import group_scores
from direct_selected_verifier import standardize
from gaussian_path_selector import candidate_scores as old_scores
from freeze_inference import inventory,verify_snapshot

OUT=ROOT/'runs/tj_direct_selection_20261031_v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def size(p):return sum(q.stat().st_size for q in Path(p).rglob('*') if q.is_file())


def main():
    started=time.perf_counter();OUT.mkdir(exist_ok=False)
    save(OUT/'registration.json',{'candidate_families':['relative','independent','joint','old_comparator'],
        'pair_weights':[.25,.5,.75],'triples':[['relative','independent','joint'],['relative','old_comparator','independent'],['relative','old_comparator','joint']],
        'selection':'validation hard correct descending, asset bytes ascending, fixed name',
        'no_test_based_selection':True,'maximum_candidates':25,'source_sha256':sha(Path(__file__))})
    linear=ROOT/'runs/tj_standalone_relative_20261031_v1'
    stages={'relative':linear,'independent':ROOT/'runs/tj_joint_direct_20261031_independent_v1',
            'joint':ROOT/'runs/tj_joint_direct_20261031_joint_v1'}
    summaries={k:json.loads((p/'summary.json').read_text()) for k,p in stages.items()}
    assert all(x['state']=='complete' for x in summaries.values())
    bank,arrays=load_inputs('validation');parsed=[[parse_answer(c['text']) for c in row['candidates']] for row in bank]
    valid=np.asarray([[v is not None for v in row] for row in parsed],bool)
    hard=np.asarray([len({v for v in row if v is not None})>1 for row in parsed])
    with np.load(linear/'private_validation_scores.npz',allow_pickle=False) as z:
        labels=z['labels'];ss=z['raw'][summaries['relative']['winner']['fit_index']]
    scores={'relative':group_scores(ss,parsed,summaries['relative']['winner']['vote_bonus'])}
    for name in ('independent','joint'):
        winner=summaries[name]['winner'];members=[f'{name}_{seed}.npz' for seed in (17,29,43)]
        with np.load(stages[name]/'private_validation_scores.npz',allow_pickle=False) as z:
            assert np.array_equal(z['labels'],labels)
            values=z['scores'][[members.index(m) for m in winner['members']]].mean(axis=0)
        scores[name]=group_scores(values,parsed,winner['vote_bonus'])
    verify_snapshot(OLD)
    scores['old_comparator']=old_scores(bank,arrays,OLD/'checkpoint/parent')
    old_good=labels[np.arange(len(bank)),np.where(valid,scores['old_comparator'],-1e6).argmax(1)]
    assert int(old_good[hard].sum())==303
    paths={k:p/'checkpoint' for k,p in stages.items()};paths['old_comparator']=OLD/'checkpoint/parent'
    assets={k:size(p) for k,p in paths.items()}
    normalized={k:standardize(s,valid) for k,s in scores.items()}
    candidates=[]
    def add(name,components):
        n=len(components);score=sum((normalized[k] if n>1 else scores[k])*weight for k,weight in components)
        choice=np.where(valid,score,-1e6).argmax(1);good=labels[np.arange(len(bank)),choice]
        footprint=sum(assets[k] for k,_ in components)+2048
        candidates.append({'name':name,'components':[[k,float(w)] for k,w in components],
            'validation_hard_correct':int(good[hard].sum()),'validation_hard_n':int(hard.sum()),
            'validation_all_correct':int(good.sum()),'validation_all_n':len(bank),
            'rescued_vs_old':int((~old_good & good & hard).sum()),'lost_vs_old':int((old_good & ~good & hard).sum()),
            'conservative_asset_bytes':footprint,'within_asset_limit':footprint<=16*1024**2})
    keys=list(scores)
    for key in keys:add(key,[(key,1.)])
    for a,b in itertools.combinations(keys,2):
        for weight in (.25,.5,.75):add(f'{a}_{weight}_{b}',[(a,weight),(b,1-weight)])
    for triple in [('relative','independent','joint'),('relative','old_comparator','independent'),('relative','old_comparator','joint')]:
        add('_'.join(triple),[(k,1/3) for k in triple])
    assert len(candidates)==25
    ranking=sorted([c for c in candidates if c['within_asset_limit']],key=lambda c:(-c['validation_hard_correct'],c['conservative_asset_bytes'],c['name']))
    save(OUT/'validation_selection.json',{'candidates':candidates,'ranking':ranking,'model_summaries':summaries,'test_opened':False})
    # Common real-training preflight bank. It is never used to score generalization.
    trainbank,ta=load_inputs('train');trainrefs=read_jsonl(DATA/'gsm8k_train.jsonl')
    indices=[i for i,row in enumerate(trainbank) if len({parse_answer(c['text']) for c in row['candidates'] if parse_answer(c['text']) is not None})>1][:467]
    audit=OUT/'training_resource_bank';(audit/'candidates').mkdir(parents=True)
    for name,rows in [('candidates/gsm8k_audit.jsonl',[trainbank[i] for i in indices]),('gsm8k_audit.jsonl',[trainrefs[i] for i in indices])]:
        (audit/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    np.savez(audit/'candidates/gsm8k_audit.npz',embeddings=ta['embeddings'][indices].reshape(-1,1024))
    files=['candidates/gsm8k_audit.jsonl','candidates/gsm8k_audit.npz','gsm8k_audit.jsonl']
    save(audit/'manifest.json',{'protocol':PROTOCOL,'splits':{'audit':len(indices)},'sha256':{n:sha(audit/n) for n in files}})
    accepted=None
    for rank,candidate in enumerate(ranking[:3]):
        snap=OUT/f'candidate_{rank}_snapshot';snap.mkdir()
        for folder in ('code','methods'):shutil.copytree(OLD/folder,snap/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in ('relative_verifier_inputs.py','relative_verifier_selector.py','joint_verifier_selector.py','direct_selected_verifier.py'):
            shutil.copyfile(ROOT/'methods'/name,snap/'methods'/name)
        checkpoint=snap/'checkpoint';checkpoint.mkdir();components=[]
        for name,weight in candidate['components']:
            shutil.copytree(paths[name],checkpoint/name)
            family='relative' if name=='relative' else ('old_comparator' if name=='old_comparator' else 'neural')
            components.append({'family':family,'path':name,'weight':weight})
        save(checkpoint/'config.json',{'format':'mp2-direct-selected-verifier-v1','components':components,'normalize_components':len(components)>1})
        asset_bytes=size(checkpoint);assert asset_bytes<=16*1024**2
        save(snap/'freeze.json',{'format_version':1,'method':'methods/direct_selected_verifier.py','checkpoint':'checkpoint',
            'selection_frozen_before_test':True,'declared_embedded_asset_bytes':0,'declared_inference_asset_bytes':asset_bytes,
            'asset_limit_bytes':16*1024**2,'files':inventory(snap,ignore_python_cache=True),'selection_record_sha256':sha(OUT/'validation_selection.json')})
        verify_snapshot(snap)
        output=OUT/f'candidate_{rank}_preflight'
        command=[sys.executable,str(ROOT/'scripts/evaluate_direct_selected_20261031.py'),'--snapshot',str(snap),'--split','audit','--data-dir',str(audit),'--output',str(output)]
        with (OUT/f'candidate_{rank}_preflight.log').open('w') as f:proc=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=60)
        if proc.returncode:continue
        resource=json.loads((output/'resources.json').read_text())
        if resource['local_resource_pass']:
            accepted={'candidate':candidate,'snapshot':str(snap.relative_to(ROOT)),'freeze_sha256':sha(snap/'freeze.json'),
                'resource_preflight':resource,'asset_bytes':asset_bytes};break
    if accepted is None:raise RuntimeError('No validation-ranked candidate passes complete CPU preflight')
    summary={'state':'selected_frozen_pending_final_test','selected':accepted,'seconds':time.perf_counter()-started,
        'all_test_read_attempts':round2_guard.EVENTS['test_read_attempts'],'test_opened':False,'target_achieved':False}
    save(OUT/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
