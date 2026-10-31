"""Validate submission hashes and records, without importing the model/data."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads((ROOT/path).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ready-to-submit',action='store_true')
    parser.parse_args()
    manifest=read('RELEASE_MANIFEST.json')
    for name,expected in manifest['files'].items():
        p=ROOT/name
        assert p.is_file() and not p.is_symlink(),name
        assert p.stat().st_size==expected['bytes'] and digest(p)==expected['sha256'],name
    fixed=ROOT/'inference';freeze=read('inference/freeze.json')
    for name,expected in freeze['files'].items():
        p=fixed/name
        assert p.is_file() and p.stat().st_size==expected['bytes'] and digest(p)==expected['sha256'],name
    checkpoint=read('CHECKPOINT_MANIFEST.json');cp=fixed/'checkpoint'
    actual={p.relative_to(cp).as_posix():p for p in cp.rglob('*') if p.is_file()}
    assert set(actual)==set(checkpoint['files'])
    for name,p in actual.items():assert digest(p)==checkpoint['files'][name]['sha256'],name
    assets=sum(p.stat().st_size for p in actual.values())
    assert assets==11808974 and assets==freeze['declared_inference_asset_bytes'] and assets<=16*1024**2
    official=read('evidence/official_release_manifest.json')['files']
    for name in ('common.py','predict.py','score_predictions.py'):
        assert digest(fixed/'code'/name)==official['code/'+name]
    score=read('evaluation/test-score.json');resource=read('evaluation/test-resources.json')
    assert score['split']=='test' and score['correct']==399 and score['n']==467
    assert abs(score['accuracy']-100*399/467)<1e-10
    assert digest(ROOT/'evaluation/test-score.json')==resource['official_score_sha256']
    assert resource['snapshot_freeze_sha256']==digest(fixed/'freeze.json') and resource['local_resource_pass']
    assert resource['asset_bytes']==assets and resource['complete_cpu_seconds']<=30
    assert all(p['sampled_child_threads']+p['controller_threads']<=4 for p in resource['processes'])
    report=read('report/REPORT_BUILD.json');assert 1<=report['pages']<=10
    assert digest(ROOT/'report/MP2_Report.pdf')==report['sha256']
    assert (ROOT/'report/MP2_Report.pdf').read_bytes().startswith(b'%PDF-')
    reproduction=read('evidence/training_reproduction.json');assert reproduction['state']=='complete'
    assert reproduction['real_fits']==18 and reproduction['serial_fits']
    assert reproduction['all_arrays_exact'] and reproduction['all_files_byte_identical']
    assert reproduction['test_read_attempts']==0 and not reproduction['test_scored']
    for name,record in reproduction['checkpoints'].items():
        assert record['frozen_sha256']==checkpoint['files'][name]['sha256']
    tree=ast.parse((fixed/freeze['method']).read_text())
    signatures=[f for f in tree.body if isinstance(f,ast.FunctionDef) and f.name=='predict']
    assert len(signatures)==1 and [x.arg for x in signatures[0].args.args]==['bank','arrays','checkpoint']
    for name in manifest['files']:
        p=Path(name)
        assert p.suffix!='.jsonl' and p.name not in ('test-predictions.json','private_inputs.npz','private_validation_scores.npz')
        if p.suffix=='.npz':assert p.parts[:2]==('inference','checkpoint')
    print(json.dumps({'materials_ready':True,'official_test_accuracy':score['accuracy'],'asset_files':len(actual),
        'asset_bytes':assets,'report_pages':report['pages'],'frozen_source_hashes':'passed','release_hashes':'passed',
        'single_GPU_training_replay':reproduction['state'],'published_or_submitted':False,
        'user_actions_remaining':['publish public code/checkpoint links','fill student ID','create official GitHub submission issue']},indent=2))


if __name__=='__main__':main()
