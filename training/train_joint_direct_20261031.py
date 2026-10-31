"""New standalone candidate networks on GPU1, real data and official validation."""
import os
GPU=os.environ.get('CUDA_VISIBLE_DEVICES','')
for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','GOTO_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import hashlib
import json
from pathlib import Path
import signal
import socket
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/p) for p in ('scripts','methods','code')]
import round2_guard
round2_guard.install(allow_validation=True,offline=True)
import numpy as np
from screen_methods import load_training
os.environ['CUDA_VISIBLE_DEVICES']=GPU
from common import load_inputs,read_jsonl,DATA,parse_answer
from relative_verifier_inputs import inputs,group_scores
from joint_verifier_selector import preprocess,forward,relation
OUT=ROOT/'runs/tj_joint_direct_20261031_v1'


def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('independent','joint'),required=True)
    args=parser.parse_args()
    expected={'independent':'GPU-26a4129f-be11-f46b-ecfa-ee7ced76267d','joint':'GPU-87089f79-acfb-f457-1003-787e55013d1b'}[args.mode]
    assert socket.gethostname()=='muyunfei518' and GPU==expected
    OUT=ROOT/f'runs/tj_joint_direct_20261031_{args.mode}_v1'
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('New model bounded3600s training')))
    signal.alarm(3600)
    OUT.mkdir(exist_ok=False);started=time.perf_counter()
    save(OUT/'registration.json',{'modes':[args.mode],'seeds':[17,29,43],'members':8,'width':128,
        'loss':'per-member correct-set mass + 0.1 valid-candidate BCE','max_epochs':150,'patience':20,'batch_questions':128,
        'optimizer':'AdamW lr0.001 weight_decay0.03','dropout':.25,'zca_ridge':.1,
        'preprocessing_fit_scope':'all official training disagreement candidates only',
        'early_stopping':'official validation disagreement accuracy, mass-loss tiebreak','vote_bonuses_after_fit':[0,.5,1.],
        'all_parameters_from_scratch':True,'no_original_model_scores_weights':True,
        'gpu_uuid':GPU,'user_authorized_two_independent_single_GPU_jobs':True,
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ('scripts/train_joint_direct_20261031.py','scripts/joint_verifier_network.py','methods/joint_verifier_selector.py','methods/relative_verifier_inputs.py')}})
    bank,emb,labels,hard=load_training();labels=labels.astype(bool)
    vb,va=load_inputs('validation');refs=read_jsonl(DATA/'gsm8k_validation.jsonl')
    vy=[]
    for row,ref in zip(vb,refs):
        assert row['id']==ref['id'] and row['question']==ref['question']
        gold=parse_answer(ref['answer']);vy.append([parse_answer(c['text'])==gold for c in row['candidates']])
    vy=np.asarray(vy,bool)
    cache=ROOT/'runs/tj_standalone_relative_20261031_v1/private_inputs.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as z:raw=z['train'].copy();vraw=z['validation'].copy();assert np.array_equal(labels,z['train_labels'].astype(bool)) and np.array_equal(vy,z['validation_labels'])
        parsed=[[parse_answer(c['text']) for c in row['candidates']] for row in bank]
        vp=[[parse_answer(c['text']) for c in row['candidates']] for row in vb]
    else:
        raw,parsed=inputs(bank,emb);vraw,vp=inputs(vb,va['embeddings'])
    same,valid=relation(parsed);vsame,vvalid=relation(vp)
    vh=np.asarray([len({a for a in row if a is not None})>1 for row in vp],bool)
    fit=np.flatnonzero(hard)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert torch.cuda.device_count()==1 and str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix('gpu-')==GPU[4:]
    from joint_verifier_network import Network,objective
    flat=np.asarray(emb[fit],np.float64).reshape(-1,1024)
    mean=flat.mean(0);scale=np.maximum(flat.std(0),1e-6)
    tf=torch.as_tensor((flat-mean)/scale,device='cuda',dtype=torch.float64)
    covariance=tf.T@tf/len(tf);eig,vec=torch.linalg.eigh((covariance+covariance.T)*.5)
    zca=((vec/(eig.clamp(min=0)+.1).sqrt()[None])@vec.T).cpu().numpy().astype(np.float32)
    stats={'mean':mean.astype(np.float32),'scale':scale.astype(np.float32),'zca':zca,
           'relative_scale':np.ones(1024,np.float32),'scalar_mean':raw[fit,:,1024:].reshape(-1,32).mean(0).astype(np.float32),
           'scalar_scale':np.maximum(raw[fit,:,1024:].reshape(-1,32).std(0),1e-3).astype(np.float32)}
    absolute,relative,scalar=preprocess(emb,raw[...,1024:],stats)
    stats['relative_scale']=np.maximum(relative[fit].reshape(-1,1024).std(0),.05).astype(np.float32)
    # Recompute with final fit-only scales, shared exactly by export and fitting.
    absolute,relative,scalar=preprocess(emb,raw[...,1024:],stats)
    vabs,vrel,vscalar=preprocess(va['embeddings'],vraw[...,1024:],stats)
    np.savez(OUT/'preprocessing.npz',**stats)
    train=[torch.as_tensor(x[fit],device='cuda') for x in (absolute,relative,scalar,same,valid)]
    target=torch.as_tensor(labels[fit].astype(np.float32),device='cuda')
    val=[torch.as_tensor(x,device='cuda') for x in (vabs,vrel,vscalar,vsame,vvalid)]
    tvy=torch.as_tensor(vy.astype(np.float32),device='cuda');tvh=torch.as_tensor(vh,device='cuda')
    all_records=[];all_scores=[];all_members=[]
    for mode in (args.mode,):
        for seed in (17,29,43):
            tick=time.perf_counter();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
            model=Network(mode).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.03)
            best=(-1,-float('inf'));best_state=None;best_epoch=0;stale=updates=0;history=[]
            for epoch in range(1,151):
                model.train();order=rng.permutation(len(fit));loss_sum=0.
                for b in range(0,len(order),128):
                    ix=order[b:b+128];opt.zero_grad(set_to_none=True)
                    score=model(*(t[ix] for t in train));loss=objective(score,target[ix],train[-1][ix])
                    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();updates+=1;loss_sum+=float(loss.detach().cpu())*len(ix)
                model.eval()
                with torch.no_grad():
                    vv=torch.cat([model(*(t[b:b+128] for t in val)) for b in range(0,len(vb),128)])
                    lossval=float(objective(vv[tvh],tvy[tvh],val[-1][tvh]).cpu())
                    pred=vv.mean(1).masked_fill(~val[-1],-1e6).argmax(-1)
                    correct=int(tvy[torch.arange(len(vb),device='cuda'),pred][tvh].sum().cpu())
                history.append({'epoch':epoch,'fit_loss':loss_sum/len(fit),'validation_hard_correct':correct,'validation_loss':lossval})
                if (correct,-lossval)>best:
                    best=(correct,-lossval);best_epoch=epoch;stale=0;best_state={k:v.detach().cpu().numpy().copy() for k,v in model.named_parameters()}
                else:stale+=1
                if epoch%5==0 or epoch==1:
                    save(OUT/'progress.json',{'state':'running','mode':mode,'seed':seed,'epoch':epoch,'best_validation_hard_correct':best[0],
                        'completed_models':len(all_records),'seconds':time.perf_counter()-started})
                if stale>=20:break
            assert best_state is not None
            with torch.no_grad():
                for k,p in model.named_parameters():p.copy_(torch.as_tensor(best_state[k],device='cuda'))
                exported=torch.cat([model(*(t[b:b+128] for t in val)) for b in range(0,len(vb),128)]).cpu().numpy()
            native=forward(vabs[:23],vrel[:23],vscalar[:23],vsame[:23],vvalid[:23],best_state,mode)
            np.testing.assert_allclose(native,exported[:23],atol=3e-4,rtol=3e-4)
            filename=f'{mode}_{seed}.npz';np.savez(OUT/filename,**best_state)
            all_scores.append(exported.mean(1));all_members.append(filename)
            record={'mode':mode,'seed':seed,'selected_epoch':best_epoch,'epochs_run':len(history),'updates':updates,
                    'best_validation_hard_correct':best[0],'best_validation_loss':-best[1],
                    'seconds':time.perf_counter()-tick,'numpy_export_max_error':float(np.max(np.abs(native-exported[:23]))),
                    'parameters':sum(p.numel() for p in model.parameters())}
            all_records.append(record);save(OUT/f'{mode}_{seed}_training.json',{'record':record,'history':history})
            del model,opt;torch.cuda.empty_cache()
    np.savez(OUT/'private_validation_scores.npz',scores=np.asarray(all_scores),labels=vy,hard=vh)
    combinations=[]
    for mode in (args.mode,):
        indices=[i for i,r in enumerate(all_records) if r['mode']==mode]
        for subset in ([i] for i in indices):
            score=np.mean([all_scores[i] for i in subset],axis=0)
            for bonus in (0.,.5,1.):
                choice=group_scores(score,vp,bonus).argmax(1);good=vy[np.arange(len(vy)),choice]
                combinations.append({'mode':mode,'members':[all_members[i] for i in subset],'vote_bonus':bonus,
                    'validation_hard_correct':int(good[vh].sum()),'validation_hard_n':int(vh.sum()),'validation_all_correct':int(good.sum())})
        score=np.mean([all_scores[i] for i in indices],axis=0)
        for bonus in (0.,.5,1.):
            choice=group_scores(score,vp,bonus).argmax(1);good=vy[np.arange(len(vy)),choice]
            combinations.append({'mode':mode,'members':[all_members[i] for i in indices],'vote_bonus':bonus,
                'validation_hard_correct':int(good[vh].sum()),'validation_hard_n':int(vh.sum()),'validation_all_correct':int(good.sum())})
    winner=max(combinations,key=lambda r:(r['validation_hard_correct'],-len(r['members']),-r['vote_bonus']))
    selected=OUT/'checkpoint';selected.mkdir()
    import shutil
    shutil.copyfile(OUT/'preprocessing.npz',selected/'preprocessing.npz')
    for name in winner['members']:shutil.copyfile(OUT/name,selected/name)
    save(selected/'config.json',{'format':'mp2-joint-standalone-v1',**winner})
    summary={'state':'complete','method':'new_standalone_candidate_network','gpu_uuid':GPU,'real_fits':len(all_records),
        'total_seconds':time.perf_counter()-started,'winner':winner,'models':all_records,'candidates':combinations,
        'asset_bytes':sum(p.stat().st_size for p in selected.rglob('*') if p.is_file()),
        'test_opened':False,'data_access_audit':dict(round2_guard.EVENTS)}
    save(OUT/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ('models','candidates')}),flush=True)


if __name__=='__main__':main()
