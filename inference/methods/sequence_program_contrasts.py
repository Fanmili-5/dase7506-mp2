"""Training-only length-matched numeric counterexamples, with occurrence slots."""
from fractions import Fraction
import hashlib,json
from common import parse_answer
from gold_programs import evaluate,size,slots
from program_sequence_data import reference_programs,question_values,normalize_program,prefix,decode_prefix,MAX_PREFIX_TOKENS
from program_contrasts import variants,text

MAX_PER_KIND=64
KINDS=('operator','quantity')


def construct(question,reference):
    d=reference_programs(question,reference);values,_=question_values(question)
    if not d['prefixes']:return {'groups':[],'reason':d['reason']}
    target=Fraction(parse_answer(reference));positives=[tuple(p) for p in d['prefixes']];pool={k:{} for k in KINDS}
    for tokens in positives:
        p=decode_prefix(tokens);assert evaluate(p,values)==target
        for kind,items in variants(p,len(values)).items():
            if kind not in KINDS:continue
            for candidate in items:
                if len(pool[kind])>=MAX_PER_KIND:break
                try:
                    if not slots(candidate) or size(candidate)>15:continue
                    before=evaluate(candidate,values);candidate=normalize_program(candidate);seq=tuple(prefix(candidate))
                    if len(seq)!=len(tokens) or len(seq)>MAX_PREFIX_TOKENS or evaluate(candidate,values)!=before or before==target:continue
                    if parse_answer(text(candidate,values))==parse_answer(reference):continue
                    assert decode_prefix(seq)==candidate;pool[kind][seq]=before
                except (ValueError,ZeroDivisionError,OverflowError,RecursionError,IndexError):continue
    groups=[];seen=set()
    for kind in KINDS:
        ordered=sorted(pool[kind],key=lambda p:hashlib.sha256((question+'|'+json.dumps(p)).encode()).hexdigest())
        for negative in ordered:
            if negative in seen:continue
            matching=[list(p) for p in positives if len(p)==len(negative)];assert matching;seen.add(negative);groups.append({'kind':kind,'positive_prefixes':matching,'negative_prefix':list(negative),'prefix_length':len(negative)});break
    return {'groups':groups,'reason':'covered' if groups else 'no_length_matched_negative','question_slots':len(values),'same_value_occurrences_kept':d.get('equal_value_aliases_retained',0),'max_negative_groups_per_question':2}
