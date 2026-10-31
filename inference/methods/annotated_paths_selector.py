"""Fixed sparse path ranking with optional recovered explicit arithmetic."""
from pathlib import Path
import json
import numpy as np
from final_program_paths import features,paths
from annotated_program_data import candidate_programs,question_context
from program_sequence_data import decode_prefix
from gold_programs import size,slots
from final_paths_selector import score_rows
from selector import answers_for,select_choices


def recovered_tokens(prefixes):
    result=['recovered_present:'+str(bool(prefixes)),'recovered_variants:'+str(min(4,len(prefixes)))]
    def node(p):
        if p[0]=='q':return ('q',(p[1],))
        if p[0]=='c':return ('c',str(p[1]))
        return (p[0],node(p[1]),node(p[2]))
    for tokens in prefixes:
        program=decode_prefix(tokens);result+=['recovered_operations:'+str(min(8,size(program))),'recovered_sources:'+str(min(8,len(slots(program))))]
        for index,path in paths(node(program)):
            signature='_'.join(path[-4:]) or 'direct'
            result+=['recovered_path:'+signature,'recovered_slot:'+str(index)+'|'+signature]
    return result[:1024]


def rows(bank,recover):
    result=[]
    for row in bank:
        q=row['question'];context=question_context(q) if recover else None
        for candidate in row['candidates']:
            text=candidate['text'];tokens=features(q,text,False)[0]
            if recover:
                parsed=candidate_programs(q,text,context)
                tokens=tokens+recovered_tokens(parsed['prefixes'][parsed['old_prefix_count']:])
            result.append(tokens)
    return result


def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),np.float32)
    root=Path(checkpoint);cfg=json.loads((root/'config.json').read_text());assert cfg['format']=='mp2-annotated-paths-v1'
    parsed=answers_for(bank);ix=[i for i,v in enumerate(parsed) if len({a for a in v if a is not None})>=2]
    scores=np.zeros((len(bank),4),np.float32)
    if ix:
        vocab=json.loads((root/'vocabulary.json').read_text())
        with np.load(root/'weights.npz') as weights:
            scores[ix]=score_rows(rows([bank[i] for i in ix],cfg['recover']),vocab,weights['idf'],weights['weights'])
    return scores


def predict(bank,arrays,checkpoint):return select_choices(bank,candidate_scores(bank,arrays,checkpoint),'max') if bank else []
