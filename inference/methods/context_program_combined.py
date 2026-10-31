"""Frozen v2 plus a small score for candidate programs already present in input."""
import json
from pathlib import Path
from quantity_shared_selector import candidate_scores as quantity
from context_program_selector import candidate_scores as program
from round4_selector import question_standardize
from round2_features import groups
from selector import select_choices


def predict(bank,arrays,checkpoint):
    if not bank:return []
    root=Path(checkpoint).resolve();cfg=json.loads((root/'config.json').read_text());assert cfg['format']=='mp2-context-program-combined-v1';base=(root/cfg['base']).resolve();head=(root/cfg['head']).resolve();assert base.is_relative_to(root) and head.is_relative_to(root);b=quantity(bank,arrays,base);r=program(bank,arrays,head);valid=groups(bank)[1];return select_choices(bank,question_standardize(b,valid)+cfg['beta']*r,'max')
