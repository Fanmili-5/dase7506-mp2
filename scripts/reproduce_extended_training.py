"""Replay train/validation recipes serially on one GPU; never score test."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();release=Path(__file__).resolve().parents[1]
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'runs').mkdir()
    for original,target in [('inference/methods','methods'),('inference/code','code'),('training','scripts')]:
        shutil.copytree(release/original,out/target)
    data=out/'code/data';(data/'candidates').mkdir(parents=True)
    names=['manifest.json']+[name for split in ('train','validation') for name in
        (f'candidates/gsm8k_{split}.jsonl',f'candidates/gsm8k_{split}.npz',f'gsm8k_{split}.jsonl')]
    for name in names:shutil.copyfile(args.data_dir/name,data/name)
    for name in ('train_relative_direct_20261031.py','train_joint_direct_20261031.py','train_objective_direct_20261031.py'):
        path=out/'scripts'/name;text=path.read_text()
        text=text.replace("assert socket.gethostname()=='muyunfei518' and GPU=='GPU-26a4129f-be11-f46b-ecfa-ee7ced76267d'", "assert sys.platform.startswith('linux') and GPU")
        text=text.replace("assert socket.gethostname()=='muyunfei518' and GPU==expected", "assert sys.platform.startswith('linux') and GPU")
        text=text.replace("torch.cuda.device_count()==1 and str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix('gpu-')==GPU[4:]", "torch.cuda.device_count()==1")
        text=text.replace("'gpu_uuid':GPU", "'visible_device':GPU");path.write_text(text)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    if not env.get('CUDA_VISIBLE_DEVICES') or ',' in env['CUDA_VISIBLE_DEVICES']:
        parser.error('Expose exactly one qualifying GPU, e.g. CUDA_VISIBLE_DEVICES=0')
    runs=[('train_relative_direct_20261031.py',[]),
          ('train_joint_direct_20261031.py',['--mode','independent']),
          ('train_joint_direct_20261031.py',['--mode','joint']),
          ('train_objective_direct_20261031.py',['--objective','pairwise']),
          ('train_objective_direct_20261031.py',['--objective','answer_group'])]
    for script,options in runs:
        subprocess.run([sys.executable,str(out/'scripts'/script),*options],cwd=out,env=env,check=True)

    def assemble(frozen,target):
        target.mkdir()
        cfg=json.loads((frozen/'config.json').read_text());fmt=cfg['format']
        if fmt=='mp2-standalone-relative-v1':
            options=[(a,b) for a in (.01,.05,.2) for b in (.003,.01)]
            index=options.index((cfg['lambda_embedding'],cfg['lambda_scalar']))
            source=out/f'runs/tj_standalone_relative_20261031_v1/fit_{index}'
            shutil.copyfile(source/'weights.npz',target/'weights.npz')
        elif fmt in ('mp2-joint-standalone-v1','mp2-objective-joint-v1'):
            source=(out/f"runs/tj_joint_direct_20261031_{cfg['mode']}_v1" if fmt=='mp2-joint-standalone-v1'
                    else out/f"runs/tj_objective_direct_20261031_{cfg['objective']}_v1")
            for name in ['preprocessing.npz',*cfg['members']]:shutil.copyfile(source/name,target/name)
        elif fmt in ('mp2-direct-selected-verifier-v1','mp2-objective-extension-selected-v1'):
            for component in cfg['components']:
                relative=Path(component['path'])
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe component path')
                assemble(frozen/relative,target/relative)
        else:raise ValueError('Unknown checkpoint format')
        shutil.copyfile(frozen/'config.json',target/'config.json')
    checkpoint=out/'reproduced_checkpoint'
    assemble(release/'inference/checkpoint',checkpoint)
    print(json.dumps({'state':'complete','checkpoint':str(checkpoint),'test_scored':False,
        'note':'This replays the training recipes and frozen chosen configuration. New weights do not automatically inherit the shipped test score.'}))


if __name__=='__main__':main()
