"""Sparse ranking with optional shared target/source/operation relations."""
from pathlib import Path
import json
import numpy as np
from final_program_paths import features as original_features
from annotated_program_data import candidate_programs,question_context
from annotated_paths_selector import recovered_tokens
from target_source_paths import prepare,features as target_features
from final_paths_selector import score_rows
from selector import answers_for,select_choices


def rows(bank,use_target):
    result=[]
    for row in bank:
        q=row['question'];context=question_context(q);target=prepare(q) if use_target else None
        for candidate in row['candidates']:
            text=candidate['text'];parsed=candidate_programs(q,text,context)
            tokens=original_features(q,text,False)[0]+recovered_tokens(parsed['prefixes'][parsed['old_prefix_count']:])
            if use_target:tokens+=target_features(target,parsed['prefixes'])
            result.append(tokens)
    return result


def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float32)
    root=Path(checkpoint);cfg=json.loads((root/'config.json').read_text());assert cfg['format']=='mp2-target-source-v1'
    parsed=answers_for(bank);ix=[i for i,v in enumerate(parsed) if len({a for a in v if a is not None})>=2];scores=np.zeros((len(bank),4),np.float32)
    if ix:
        vocab=json.loads((root/'vocabulary.json').read_text())
        with np.load(root/'weights.npz') as w:scores[ix]=score_rows(rows([bank[i] for i in ix],cfg['use_target']),vocab,w['idf'],w['weights'])
    return scores


def predict(bank,arrays,checkpoint):return select_choices(bank,candidate_scores(bank,arrays,checkpoint),'max') if bank else []
