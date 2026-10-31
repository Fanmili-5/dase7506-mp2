"""Name-independent requested-object/source/path relations, not a solver."""
import re
from collections import Counter
from program_sequence_data import question_values,decode_prefix
from quantity_features import requested,STOP as QUERY_STOP,CUES
from quantity_role_features import TARGET_STOP
from unit_propagation import prepare as unit_prepare,lemma,words,STOP,STANDARD,coarse
from final_program_paths import paths
from gold_programs import slots

PREFIX='target@'
REMOVE=STOP|QUERY_STOP|TARGET_STOP|set(CUES)|{'which','what','be','does','did','s','is','are'}


def entities(text):
    return [lemma(w) for w in words(text) if w not in REMOVE and lemma(w) not in STANDARD and len(w)>1]


def match_level(target,context):
    hit=target&context
    return 0 if not hit else 2 if hit==target else 1


def prepare(question):
    values,meta=question_values(question);q=meta['normalized_question'];ask=requested(q)
    explicit=bool(re.search(r'\b(?:how (?:many|much|long|far)|what|which)\b',q,re.I))
    start=q.rfind(ask) if explicit and ask else len(q)
    target=set(entities(ask)) if explicit else set()
    _,mentions,sources,_,_,unit_targets=unit_prepare(question)
    by_position={(m[3],m[4]):source for m,source in zip(mentions,sources)}
    target_dims=[s.dimensions for s in unit_targets if explicit and s.dimensions is not None]
    source_rows=[];counts=Counter(values)
    for i,(value,(a,b)) in enumerate(zip(values,meta['positions'])):
        left=q[max(0,a-80):a];right=q[b:min(len(q),b+80,start if b<=start else len(q))]
        # Numeric neighbours and clause boundaries bound the role context.
        if i and meta['positions'][i-1][1]>=max(0,a-80):left=q[meta['positions'][i-1][1]:a]
        if i+1<len(values) and meta['positions'][i+1][0]<=b+80:right=right[:max(0,meta['positions'][i+1][0]-b)]
        left=re.split(r'[.!?;,\n]',left)[-1];right=re.split(r'[.!?;,\n]',right)[0]
        pre=set(entities(left)[-3:]);post=set(entities(right)[:3]);states=by_position.get((a,b),[])
        dims=[s.dimensions for s in states if s.dimensions is not None]
        fine=bool(target_dims and any(d==t for d in dims for t in target_dims))
        rough=bool(target_dims and any(coarse(d)==coarse(t) for d in dims for t in target_dims))
        target_keys={k for t in target_dims for k,v in t if v>0}
        source_rows.append(dict(pre=match_level(target,pre),post=match_level(target,post),
            any=match_level(target,pre|post),entity_known=bool(pre|post),unit_known=bool(dims),
            fine=fine,coarse=rough,rate=any(s.rates for s in states),
            target_unit_numerator=any(v>0 and k in target_keys for d in dims for k,v in d),
            target_unit_denominator=any(v<0 and k in target_keys for d in dims for k,v in d),
            equal_value_alias=counts[value]>1,in_query=a>=start))
    return dict(target_known=bool(target),unit_target_known=bool(target_dims),
        rate_requested=any(v<0 for t in target_dims for _,v in t) or bool(re.search(r'\bper\b',ask,re.I)),
        sources=source_rows,question_slots=len(values))


def features(context,prefixes):
    result=[]
    def add(key,value):result.append(PREFIX+key+':'+str(value))
    add('object_known',context['target_known']);add('unit_known',context['unit_target_known'])
    add('rate_requested',context['rate_requested']);add('program_present',bool(prefixes))
    def node(program):
        if program[0]=='q':return ('q',(program[1],))
        if program[0]=='c':return ('c',str(program[1]))
        return (program[0],node(program[1]),node(program[2]))
    sources=context['sources']
    for tokens in prefixes:
        p=decode_prefix(tokens);used=slots(p)
        if any(i>=len(sources) for i in used):raise ValueError('Out-of-range source slot')
        for i,path in paths(node(p)):
            sig='_'.join(path[-4:]) or 'direct';s=sources[i]
            for field in ('pre','post','any','entity_known','unit_known','fine','coarse','rate','target_unit_numerator','target_unit_denominator','equal_value_alias','in_query'):
                add(field+'|'+sig,s[field])
            add('rate_query='+str(context['rate_requested'])+'|source_rate='+str(s['rate'])+'|'+sig,True)
        for side in ('pre','post','any','fine','rate'):
            relevant={i for i,s in enumerate(sources) if s[side]}
            add(side+'_matched_used',min(3,len(relevant&used)))
            add(side+'_matched_missing',min(3,len(relevant-used)))
        def edges(program):
            if program[0] in ('q','c'):return
            ls,rs=slots(program[1]),slots(program[2]);op=program[0]
            for field in ('pre','post','fine','rate'):
                l=max((int(sources[i][field]) for i in ls),default=-1)
                r=max((int(sources[i][field]) for i in rs),default=-1)
                add('edge|'+op+'|'+field,f'{l}/{r}')
            edges(program[1]);edges(program[2])
        edges(p)
    # Relation tokens contain no object names, raw numbers, IDs or references.
    return result[:2048]
