"""Freeze one already-trained validation-tied comparator for final scoring."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from freeze_inference import inventory,verify_snapshot


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    base=ROOT/'runs/tj_direct_selection_20261031_v1/candidate_0_snapshot'
    evidence=ROOT/'runs/tj_objective_selection_20261031_v1/validation_selection.json'
    candidates=json.loads(evidence.read_text())['candidates']
    ranking=sorted([c for c in candidates if c['name']!='incumbent' and c['within_asset_limit']],
        key=lambda c:(-c['validation_hard_correct'],c['conservative_asset_bytes'],c['name']))
    chosen=ranking[0]
    assert chosen['name']=='incumbent_0.5_pairwise' and chosen['validation_hard_correct']==309
    out=ROOT/'runs/tj_extension_comparator_final_20261031_v1';out.mkdir(exist_ok=False)
    save(out/'selection_before_test.json',{'candidate':chosen,'selection_source_sha256':sha(evidence),
        'rule':'highest existing validation accuracy among non-incumbent candidates, then asset bytes/name',
        'user_authorized_final_scoring_despite_validation_tie':True,
        'no_new_training_or_test_based_selection':True,'original_gain_gate_relaxed_by_latest_user_request':True})
    verify_snapshot(base)
    snap=out/'snapshot';snap.mkdir()
    for folder in ('code','methods'):shutil.copytree(base/folder,snap/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('group_objective_selector.py','improvement_selected_verifier.py'):
        shutil.copyfile(ROOT/'methods'/name,snap/'methods'/name)
    cp=snap/'checkpoint';cp.mkdir()
    shutil.copytree(base/'checkpoint',cp/'incumbent')
    shutil.copytree(ROOT/'runs/tj_objective_direct_20261031_pairwise_v1/checkpoint',cp/'pairwise')
    save(cp/'config.json',{'format':'mp2-objective-extension-selected-v1','normalize_components':True,
        'components':[{'family':'incumbent','path':'incumbent','weight':.5},{'family':'objective','path':'pairwise','weight':.5}]})
    assets=sum(p.stat().st_size for p in cp.rglob('*') if p.is_file());assert assets<=16*1024**2
    save(snap/'freeze.json',{'format_version':1,'method':'methods/improvement_selected_verifier.py','checkpoint':'checkpoint',
        'selection_frozen_before_test':True,'declared_embedded_asset_bytes':0,'declared_inference_asset_bytes':assets,
        'asset_limit_bytes':16*1024**2,'files':inventory(snap,ignore_python_cache=True),
        'selection_record_sha256':sha(out/'selection_before_test.json')})
    verify_snapshot(snap);verify_snapshot(base)
    summary={'state':'frozen_before_final_score','candidate':chosen,'asset_bytes':assets,
        'snapshot':str(snap.relative_to(ROOT)),'freeze_sha256':sha(snap/'freeze.json'),
        'new_fits':0,'test_opened':False}
    save(out/'freeze_summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':main()
