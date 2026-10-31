"""One bounded validation comparison. Test remains inaccessible in this process."""
import os
for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','GOTO_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
INC=ROOT/'runs/tj_direct_selection_20261031_v1/candidate_0_snapshot'
OUT=ROOT/'runs/tj_objective_selection_20261031_v1'
sys.path[:0]=[str(ROOT/'scripts'),str(INC/'methods'),str(ROOT/'methods'),str(ROOT/'code')]
import round2_guard
round2_guard.install(allow_validation=True,offline=False)
import numpy as np
from common import load_inputs,parse_answer,DATA
from relative_verifier_inputs import group_scores
from direct_selected_verifier import standardize,candidate_scores as incumbent_scores
from freeze_inference import verify_snapshot,inventory


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def size(path):return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def main():
    began=time.perf_counter();OUT.mkdir(exist_ok=False)
    save(OUT/'registration.json',{'minimum_validation_hard_correct_to_score_new_test':310,
        'candidate_families':['relative','incumbent','pairwise','answer_group'],'pair_weights':[.25,.5,.75],
        'max_candidates':25,'test_access':False,'source_sha256':sha(__file__)})
    stages={name:ROOT/f'runs/tj_objective_direct_20261031_{name}_v1' for name in ('pairwise','answer_group')}
    summaries={name:json.loads((path/'summary.json').read_text()) for name,path in stages.items()}
    assert all(s['state']=='complete' for s in summaries.values())
    linear=ROOT/'runs/tj_standalone_relative_20261031_v1';ls=json.loads((linear/'summary.json').read_text())
    bank,arrays=load_inputs('validation');parsed=[[parse_answer(c['text']) for c in row['candidates']] for row in bank]
    valid=np.asarray([[a is not None for a in row] for row in parsed],bool)
    hard=np.asarray([len({a for a in row if a is not None})>1 for row in parsed])
    with np.load(linear/'private_validation_scores.npz',allow_pickle=False) as z:
        labels=z['labels'].copy();raw=z['raw'][ls['winner']['fit_index']]
    scores={'relative':group_scores(raw,parsed,ls['winner']['vote_bonus'])}
    verify_snapshot(INC);scores['incumbent']=incumbent_scores(bank,arrays,INC/'checkpoint')
    for name,path in stages.items():
        winner=summaries[name]['winner'];all_members=[f'joint_{s}.npz' for s in (17,29,43)]
        with np.load(path/'private_validation_scores.npz',allow_pickle=False) as z:
            assert np.array_equal(z['labels'],labels)
            raw=z['scores'][[all_members.index(m) for m in winner['members']]].mean(axis=0)
        scores[name]=group_scores(raw,parsed,winner['vote_bonus'])
    current_good=labels[np.arange(len(bank)),scores['incumbent'].argmax(1)]
    assert current_good[hard].sum()==309
    normalized={name:standardize(s,valid) for name,s in scores.items()}
    paths={'relative':linear/'checkpoint','incumbent':INC/'checkpoint',**{k:p/'checkpoint' for k,p in stages.items()}}
    assets={k:size(p) for k,p in paths.items()};candidates=[]
    def add(name,components):
        values=sum((normalized[k] if len(components)>1 else scores[k])*w for k,w in components)
        choice=np.where(valid,values,-1e6).argmax(1);good=labels[np.arange(len(bank)),choice]
        footprint=sum(assets[k] for k,_ in components)+2048
        candidates.append({'name':name,'components':[[k,float(w)] for k,w in components],
            'validation_hard_correct':int(good[hard].sum()),'validation_hard_n':int(hard.sum()),
            'validation_all_correct':int(good.sum()),'validation_all_n':len(bank),
            'rescued_vs_incumbent':int((~current_good & good & hard).sum()),'lost_vs_incumbent':int((current_good & ~good & hard).sum()),
            'conservative_asset_bytes':footprint,'within_asset_limit':footprint<=16*1024**2})
    for name in scores:add(name,[(name,1)])
    for a,b in itertools.combinations(scores,2):
        for weight in (.25,.5,.75):add(f'{a}_{weight}_{b}',[(a,weight),(b,1-weight)])
    for triple in [('relative','pairwise','answer_group'),('relative','incumbent','pairwise'),('relative','incumbent','answer_group')]:
        add('_'.join(triple),[(k,1/3) for k in triple])
    assert len(candidates)==25
    ranking=sorted((c for c in candidates if c['within_asset_limit']),key=lambda c:(-c['validation_hard_correct'],c['conservative_asset_bytes'],c['name']))
    save(OUT/'validation_selection.json',{'candidates':candidates,'ranking':ranking,'model_summaries':summaries,'test_opened':False})
    if ranking[0]['validation_hard_correct']<=309:
        summary={'state':'complete_no_validation_gain','best_candidate':ranking[0],'new_fits':6,'test_opened':False,
                 'seconds':time.perf_counter()-began,'data_access_audit':dict(round2_guard.EVENTS)}
        save(OUT/'summary.json',summary);print(json.dumps(summary),flush=True);return
    chosen=None
    for rank,candidate in enumerate([r for r in ranking if r['validation_hard_correct']>309][:3]):
        snap=OUT/f'candidate_{rank}_snapshot';snap.mkdir()
        for name in ('code','methods'):shutil.copytree(INC/name,snap/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in ('group_objective_selector.py','improvement_selected_verifier.py'):
            shutil.copyfile(ROOT/'methods'/name,snap/'methods'/name)
        checkpoint=snap/'checkpoint';checkpoint.mkdir();components=[]
        for name,weight in candidate['components']:
            shutil.copytree(paths[name],checkpoint/name)
            family=name if name in ('relative','incumbent') else 'objective'
            components.append({'family':family,'path':name,'weight':weight})
        save(checkpoint/'config.json',{'format':'mp2-objective-extension-selected-v1','components':components,'normalize_components':len(components)>1})
        asset_bytes=size(checkpoint);assert asset_bytes<=16*1024**2
        save(snap/'freeze.json',{'format_version':1,'method':'methods/improvement_selected_verifier.py','checkpoint':'checkpoint',
            'selection_frozen_before_test':True,'declared_embedded_asset_bytes':0,'declared_inference_asset_bytes':asset_bytes,
            'asset_limit_bytes':16*1024**2,'files':inventory(snap,ignore_python_cache=True),'selection_record_sha256':sha(OUT/'validation_selection.json')})
        verify_snapshot(snap)
        audit=ROOT/'runs/tj_direct_selection_20261031_v1/training_resource_bank';output=OUT/f'candidate_{rank}_preflight'
        args=[sys.executable,str(ROOT/'scripts/evaluate_direct_selected_20261031.py'),'--snapshot',str(snap),'--split','audit','--data-dir',str(audit),'--output',str(output)]
        with (OUT/f'candidate_{rank}_preflight.log').open('w') as f:p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=60)
        if p.returncode:continue
        resource=json.loads((output/'resources.json').read_text())
        if not resource['local_resource_pass']:continue
        validation=OUT/'final_validation_replay'
        args=[sys.executable,str(ROOT/'scripts/evaluate_direct_selected_20261031.py'),'--snapshot',str(snap),'--split','validation','--data-dir',str(DATA),'--output',str(validation)]
        with (OUT/'validation_replay.log').open('w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=60)
        official=json.loads((validation/'validation-score.json').read_text())
        assert official['correct']==candidate['validation_all_correct']
        chosen={'candidate':candidate,'snapshot':str(snap.relative_to(ROOT)),'freeze_sha256':sha(snap/'freeze.json'),
            'asset_bytes':asset_bytes,'training_resource_preflight_seconds':resource['complete_cpu_seconds'],'validation_cpu_match':True};break
    if chosen is None:raise RuntimeError('Improved validation candidates did not pass actual CPU verification')
    summary={'state':'selected_frozen_pending_final_test','selected':chosen,'new_fits':6,
        'seconds':time.perf_counter()-began,'test_opened':False,'data_access_audit':dict(round2_guard.EVENTS)}
    save(OUT/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
