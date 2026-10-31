"""Fixed v2 plus a frozen, CPU sparse score of existing arithmetic paths."""
import json
from pathlib import Path
from quantity_shared_selector import candidate_scores as quantity
from annotated_paths_selector import candidate_scores as paths
from round4_selector import question_standardize
from round2_features import groups
from selector import select_choices


def predict(bank,arrays,checkpoint):
    if not bank:return []
    root=Path(checkpoint).resolve();cfg=json.loads((root/'config.json').read_text())
    assert cfg['format']=='mp2-annotated-paths-combined-v1'
    base=(root/cfg['base']).resolve();head=(root/cfg['head']).resolve()
    assert base.is_relative_to(root) and head.is_relative_to(root)
    valid=groups(bank)[1];a=quantity(bank,arrays,base);b=paths(bank,arrays,head)
    return select_choices(bank,question_standardize(a,valid)+cfg['beta']*question_standardize(b,valid),'max')
