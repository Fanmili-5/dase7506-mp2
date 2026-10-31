"""Fit-only arithmetic contrast construction from supplied references.

No file access, pretrained state or LLM generation. Every retained negative is
evaluated exactly and has a different official final answer from the reference.
"""
from fractions import Fraction
import hashlib
from common import parse_answer
from gold_programs import extract,question_values,evaluate,dumps,size,slots,canonical
from final_program_paths import trace,paths

MAX_NEGATIVES=3
MAX_VARIANTS_PER_KIND=64


def locations(program,path=()):
    if program[0] in ('q','c'):return [(path,program)]
    return [(path,program)]+locations(program[1],path+(1,))+locations(program[2],path+(2,))


def replace(program,path,node):
    if not path:return node
    branch=path[0];left=replace(program[1],path[1:],node) if branch==1 else program[1];right=replace(program[2],path[1:],node) if branch==2 else program[2]
    return canonical(program[0],left,right)


def variants(program,quantity_count):
    result={'operator':[],'quantity':[],'omission':[]}
    for path,node in locations(program):
        if node[0] not in ('q','c'):
            for op in ('+','-','*','/'):
                if op!=node[0] and len(result['operator'])<MAX_VARIANTS_PER_KIND:result['operator'].append(replace(program,path,canonical(op,node[1],node[2])))
            if node[0] in ('-','/') and len(result['operator'])<MAX_VARIANTS_PER_KIND:result['operator'].append(replace(program,path,(node[0],node[2],node[1])))
            for branch in (1,2):
                if len(result['omission'])<MAX_VARIANTS_PER_KIND:result['omission'].append(replace(program,path,node[branch]))
        elif node[0]=='q':
            for index in range(quantity_count):
                if index!=node[1] and len(result['quantity'])<MAX_VARIANTS_PER_KIND:result['quantity'].append(replace(program,path,('q',index)))
    return result


def expression(program,values):
    if program[0]=='q':return '('+str(values[program[1]])+')'
    if program[0]=='c':return str(program[1])
    return '('+expression(program[1],values)+program[0]+expression(program[2],values)+')'


def text(program,values):
    value=evaluate(program,values);numeric=str(value)
    return expression(program,values)+' = '+numeric+'\n#### '+numeric


def usable(question,candidate):
    roots,facts,*_=trace(question,candidate)
    return not facts['false'] and any(paths(root.node) for root in roots)


def construct(question,reference):
    parsed=parse_answer(reference)
    if parsed is None:return {'candidates':[],'labels':[],'reason':'unknown_reference_final'}
    target=Fraction(parsed);result=extract(question,reference);values,_=question_values(question)
    if not result['programs']:return {'candidates':[],'labels':[],'reason':result['reason']}
    program=result['programs'][0]
    try:
        assert evaluate(program,values)==target
        positive=text(program,values)
    except (ValueError,ZeroDivisionError,OverflowError,RecursionError):return {'candidates':[],'labels':[],'reason':'invalid_positive'}
    if not usable(question,positive):return {'candidates':[],'labels':[],'reason':'positive_has_no_grounded_path'}
    source=variants(program,len(values));kept=[];seen={dumps(program)};types=[];seed=question+'|'+dumps(program)
    def order(items):return sorted(items,key=lambda p:hashlib.sha256((seed+'|'+dumps(p)).encode()).hexdigest())
    def keep(p,kind):
        key=dumps(p)
        if key in seen or not slots(p):return False
        seen.add(key)
        try:
            value=evaluate(p,values)
            if value==target:return False
            candidate=text(p,values)
            if parse_answer(candidate)==parsed or not usable(question,candidate):return False
        except (ValueError,ZeroDivisionError,OverflowError,RecursionError,IndexError):return False
        kept.append((p,candidate));types.append(kind);return True
    # Include distinct error mechanisms before filling any remaining slots.
    ordered={kind:order(items) for kind,items in source.items()}
    for kind in ('operator','quantity','omission'):
        for p in ordered[kind]:
            if keep(p,kind):break
    if len(kept)<MAX_NEGATIVES:
        for kind in ('operator','quantity','omission'):
            for p in ordered[kind]:
                if len(kept)>=MAX_NEGATIVES:break
                keep(p,kind)
    if not kept:return {'candidates':[],'labels':[],'reason':'no_distinct_valid_negative'}
    candidates=[positive]+[t for p,t in kept[:MAX_NEGATIVES]];programs=[program]+[p for p,t in kept[:MAX_NEGATIVES]]
    assert parse_answer(candidates[0])==parsed and all(parse_answer(t) is not None and parse_answer(t)!=parsed for t in candidates[1:])
    return {'candidates':candidates,'labels':[1]+[0]*(len(candidates)-1),'programs':[dumps(p) for p in programs],'negative_types':types[:MAX_NEGATIVES],'reason':'covered','positive_operations':size(program),'question_slots':len(values),'same_value_slots_collapsed':result['equal_value_mentions_collapsed'],'only_supplied_reference_and_question_used':True}
