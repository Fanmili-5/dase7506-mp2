"""Direct independent verifier training on GPU0, official validation selection."""
import os
GPU = os.environ.get('CUDA_VISIBLE_DEVICES', '')
for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','GOTO_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import hashlib
import json
from pathlib import Path
import shutil
import signal
import socket
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ('scripts','methods','code')]
import round2_guard
round2_guard.install(allow_validation=True, offline=True)
import numpy as np
from screen_methods import load_training
os.environ['CUDA_VISIBLE_DEVICES'] = GPU
from common import load_inputs, read_jsonl, DATA, parse_answer
from relative_verifier_inputs import inputs, group_scores

OUT = ROOT / 'runs/tj_standalone_relative_20261031_v1'
CONFIGS = [(a,b) for a in (.01,.05,.2) for b in (.003,.01)]


def save(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def validation_data():
    bank, arrays = load_inputs('validation')
    refs = read_jsonl(DATA / 'gsm8k_validation.jsonl')
    labels=[]
    for row, ref in zip(bank, refs):
        assert row['id']==ref['id'] and row['question']==ref['question']
        gold=parse_answer(ref['answer'])
        labels.append([parse_answer(c['text'])==gold for c in row['candidates']])
    return bank, arrays, np.asarray(labels,bool)


def main():
    assert socket.gethostname()=='muyunfei518' and GPU=='GPU-26a4129f-be11-f46b-ecfa-ee7ced76267d'
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Direct training budget1800s')))
    signal.alarm(1800)
    OUT.mkdir(exist_ok=False)
    started=time.perf_counter()
    save(OUT/'registration.json', {'configs':CONFIGS,'vote_bonuses':[0,.5,1.], 'seed':17,
        'all_weights_from_scratch':True,'train_scope':'official train informative questions',
        'selection':'official validation hard accuracy then validation BCE',
        'no_upstream_weights_data_predictions':True,'gpu_uuid':GPU,'concurrent_other_job_single_gpu':True,
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ('scripts/train_relative_direct_20261031.py','methods/relative_verifier_inputs.py','methods/relative_verifier_selector.py')}})
    bank, emb, y, hard=load_training()
    x, parsed=inputs(bank,emb)
    vb, va, vy=validation_data()
    vx, vp=inputs(vb,va['embeddings'])
    vh=np.asarray([len({v for v in a if v is not None})>1 for a in vp])
    raw=x[hard].reshape(-1,x.shape[-1]);target=y[hard].ravel().astype(np.float64)
    mu=raw.mean(0);sd=raw.std(0)+1e-6;z=(raw-mu)/sd
    vz=(vx-mu)/sd
    np.savez(OUT/'private_inputs.npz', train=x, validation=vx, train_labels=y, train_hard=hard, validation_labels=vy, validation_hard=vh)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(17)
    assert torch.cuda.device_count()==1 and str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix('gpu-')==GPU[4:]
    tx=torch.as_tensor(z,device='cuda');ty=torch.as_tensor(target,device='cuda')
    records=[];rawscores=[]
    for index,(lam,scalar) in enumerate(CONFIGS):
        tick=time.perf_counter();w=torch.nn.Parameter(torch.zeros(z.shape[-1],dtype=torch.float64,device='cuda'));b=torch.nn.Parameter(torch.zeros((),dtype=torch.float64,device='cuda'))
        penalty=np.full(z.shape[-1],lam);penalty[1024:]=scalar;tp=torch.as_tensor(penalty,device='cuda')
        opt=torch.optim.LBFGS([w,b],lr=1,max_iter=350,max_eval=500,tolerance_grad=1e-9,tolerance_change=1e-14,history_size=30,line_search_fn='strong_wolfe')
        calls=0
        def closure():
            nonlocal calls
            opt.zero_grad();logit=tx@w+b;loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,ty)+.5*(tp*w*w).sum();loss.backward();calls+=1;return loss
        opt.step(closure);loss=float(closure().detach().cpu());ww=w.detach().cpu().numpy();bb=float(b.detach().cpu());grad=max(float(w.grad.abs().max().cpu()),abs(float(b.grad.cpu())))
        zz=z@ww+bb;prob=1/(1+np.exp(np.clip(-zz,-700,700)));cpu_grad=z.T@(prob-target)/len(target)+penalty*ww
        assert np.max(np.abs(cpu_grad-w.grad.cpu().numpy()))<1e-9
        score=vz@ww+bb;rawscores.append(score)
        lossval=float(np.mean(np.logaddexp(0,score[vh])-vy[vh]*score[vh]))
        checkpoint=OUT/f'fit_{index}';checkpoint.mkdir();np.savez(checkpoint/'weights.npz',w=ww,b=np.float64(bb),mu=mu,sd=sd)
        for bonus in (0.,.5,1.):
            grouped=group_scores(score,vp,bonus);choice=grouped.argmax(1);good=vy[np.arange(len(vy)),choice]
            record={'fit_index':index,'lambda_embedding':lam,'lambda_scalar':scalar,'vote_bonus':bonus,
                    'validation_hard_correct':int(good[vh].sum()),'validation_hard_n':int(vh.sum()),
                    'validation_all_correct':int(good.sum()),'validation_all_n':len(vy),'validation_bce':lossval,
                    'fit_seconds':time.perf_counter()-tick,'closure_calls':calls,'gradient_infinity':grad}
            records.append(record)
        save(checkpoint/'config.json',{'format':'mp2-standalone-relative-v1','vote_bonus':.5,'lambda_embedding':lam,'lambda_scalar':scalar})
        save(OUT/'progress.json',{'state':'running','finished_fits':index+1,'planned_fits':len(CONFIGS),'best_validation_hard_correct':max(r['validation_hard_correct'] for r in records),'seconds':time.perf_counter()-started})
        del opt,w,b
    winner=max(records,key=lambda r:(r['validation_hard_correct'],-r['validation_bce'],-r['fit_index'],-abs(r['vote_bonus']-.5)))
    selected=OUT/'checkpoint';shutil.copytree(OUT/f"fit_{winner['fit_index']}",selected)
    cfg=json.loads((selected/'config.json').read_text());cfg['vote_bonus']=winner['vote_bonus'];save(selected/'config.json',cfg)
    np.savez(OUT/'private_validation_scores.npz',raw=np.asarray(rawscores),labels=vy,hard=vh)
    summary={'state':'complete','method':'new_standalone_relative','winner':winner,'candidates':records,
             'real_fits':len(CONFIGS),'total_seconds':time.perf_counter()-started,'gpu_uuid':GPU,
             'asset_bytes':sum(p.stat().st_size for p in selected.rglob('*') if p.is_file()),
             'test_opened':False,'data_access_audit':dict(round2_guard.EVENTS)}
    save(OUT/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='candidates'}),flush=True)


if __name__=='__main__':main()
