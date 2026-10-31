"""Validate submission hashes and records, without importing the model/data."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads((ROOT/path).read_text())


def semantic_digest(path, registration_metadata=False):
    tree=ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
                node.body.pop(0)
        if registration_metadata and isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='save' and len(node.args)>=2:
            target=node.args[0]
            if isinstance(target,ast.BinOp) and isinstance(target.right,ast.Constant) and target.right.value=='registration.json':
                value=node.args[1]
                assert isinstance(value,ast.Dict)
                pairs=[(k,v) for k,v in zip(value.keys,value.values) if not (isinstance(k,ast.Constant) and k.value=='source')]
                value.keys=[k for k,v in pairs];value.values=[v for k,v in pairs]
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def verify_source_revision(fixed,freeze,scored_hash):
    revision=read('evidence/source_revision.json')
    assert revision['scored_freeze_sha256']==scored_hash
    assert revision['current_freeze_sha256']==digest(fixed/'freeze.json')
    original_text=revision['scored_freeze_json']
    assert hashlib.sha256(original_text.encode()).hexdigest()==scored_hash
    original=json.loads(original_text)
    assert original['files'].keys()==freeze['files'].keys()
    assert freeze['documentation_revision_of']==scored_hash
    changed={k for k in original['files'] if original['files'][k]!=freeze['files'][k]}
    assert changed=={'methods/relative_verifier_inputs.py'}
    assert {item['path'] for item in revision['files']}=={'inference/methods/relative_verifier_inputs.py','training/train_relative_direct_20261031.py'}
    for item in revision['files']:
        path=ROOT/item['path']
        assert digest(path)==item['after_sha256']
        assert semantic_digest(path,item['registration_metadata'])==item['semantic_ast_sha256']
    assert revision['files'][0]['before_sha256']==original['files']['methods/relative_verifier_inputs.py']['sha256']
    assert revision['checkpoint_bytes_unchanged'] and revision['official_scores_unchanged']


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
    current_freeze=digest(fixed/'freeze.json');version=read('SOURCE_VERSION.json')
    assert version['model_freeze_sha256']==current_freeze
    if resource['snapshot_freeze_sha256']!=current_freeze:
        assert version['scored_model_freeze_sha256']==resource['snapshot_freeze_sha256']
        verify_source_revision(fixed,freeze,resource['snapshot_freeze_sha256'])
    assert resource['local_resource_pass']
    assert resource['asset_bytes']==assets and resource['complete_cpu_seconds']<=30
    assert all(p['sampled_child_threads']+p['controller_threads']<=4 for p in resource['processes'])
    report=read('report/REPORT_BUILD.json');assert 1<=report['pages']<=10
    assert report['format']=='LaTeX' and report['source']=='report/MP2_Report.tex'
    assert digest(ROOT/report['source'])==report['source_sha256']
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
        'single_GPU_training_replay':reproduction['state'],
        'scope':'file integrity, official score record and recorded resource measurements',
        'reference_cpu':'Xeon Platinum 8457C',
        'training_resources':'one RTX 4090 per training run; independent experiments ran concurrently',
        'score_submission':'student action; this checker does not create submission issues'},indent=2))


if __name__=='__main__':main()
