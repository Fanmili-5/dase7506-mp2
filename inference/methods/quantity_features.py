"""Bounded, reference-free question/quantity compatibility signals.

These are fallible soft features, not a solver or hard validity constraints.
Only public unit definitions are fixed; no corpus vocabulary or labels live here.
"""
import re
from fractions import Fraction
import numpy as np
from round2_features import NUMBER, numbers, surface_features

STOP = set('a an the of in on at to for and or is are was were be by with as this that it he she they his her their how many much what does do did will have has had'.split())
UNIT_GROUPS = (
    ('second', 'seconds'), ('minute', 'minutes'), ('hour', 'hours'),
    ('day', 'days'), ('week', 'weeks'), ('month', 'months'), ('year', 'years'),
    ('dollar', 'dollars', '$'), ('cent', 'cents'), ('dozen', 'dozens'),
    ('inch', 'inches'), ('foot', 'feet'), ('yard', 'yards'), ('mile', 'miles'),
    ('meter', 'meters', 'metre', 'metres'), ('kilometer', 'kilometers'),
    ('ounce', 'ounces'), ('pound', 'pounds'), ('liter', 'liters'), ('gallon', 'gallons'))
UNIT_MAP = {v:i for i,g in enumerate(UNIT_GROUPS) for v in g}
CUES = ('total', 'remain', 'more', 'less', 'difference', 'each', 'per', 'every',
        'rate', 'percent', 'average', 'both', 'neither', 'cheaper', 'expensive', 'times')

def tokens(text):
    result = []
    for x in re.findall(r"[a-z]+|\$|%", text.lower()):
        if len(x)>3 and x.endswith('s') and x not in ('less','times'): x=x[:-1]
        if x not in STOP: result.append(x)
    return result

def overlap(a,b):
    return len(a & b)/max(1,len(a | b))

def units(text):
    return {UNIT_MAP[x] for x in re.findall(r'[a-z]+|\$',text.lower()) if x in UNIT_MAP}

def requested(text):
    matches=list(re.finditer(r'\b(?:how (?:many|much|long|far)|what|which)\b',text,re.I))
    return text[matches[-1].start():] if matches else re.split(r'[?!\n]',text)[-1]

def slots(text):
    result=[]
    for m in list(NUMBER.finditer(text))[:64]:
        try:
            value=Fraction(m.group().replace(',','').rstrip('%'))/(100 if m.group().endswith('%') else 1)
            if abs(value)>10**12: continue
        except (ValueError,ZeroDivisionError):continue
        # Context is a fixed local approximation, not a learned/dependency parse.
        left=text[max(0,m.start()-55):m.start()]
        right=text[m.end():m.end()+55]
        result.append((value,set(tokens(left+' '+right)),units(right[:35]), 'per' in tokens(left+' '+right) or '/' in right[:18]))
    return result

def summarize(xs):
    return [float(np.mean(xs)) if xs else 0., max(xs,default=0.), min(xs,default=0.)]

def extract(bank):
    from common import parse_answer
    surface=surface_features(bank)
    aligned=np.zeros((len(bank),4,24),np.float32)
    unit=np.zeros((len(bank),4,4*len(UNIT_GROUPS)+16+len(CUES)*5),np.float32)
    for q,row in enumerate(bank):
        question=row['question']; query=requested(question); qt=set(tokens(query))
        qs=slots(question); qu=units(question); ru=units(query); qwords=set(tokens(question))
        for i,c in enumerate(row['candidates']):
            text=c['text']; body=text.rsplit('####',1)[0]; tail=body[-250:]
            cs=slots(body); ct=set(tokens(body)); tt=set(tokens(tail)); cu=units(body); tu=units(tail)
            same=[]; role=[]; compatibility=[]; rate=[]
            for value,context,u,r in qs:
                matches=[s for s in cs if s[0]==value]
                same.append(float(bool(matches)))
                role.append(max((overlap(context,s[1]) for s in matches),default=0.))
                compatibility.append(max((float(bool(u&s[2])) for s in matches),default=0.))
                rate.append(max((float(r==s[3]) for s in matches),default=0.))
            parsed=parse_answer(text)
            answer=Fraction(parsed) if parsed is not None else None
            answer_slots=[s for s in cs if answer is not None and abs(float(s[0])-float(answer))<=1e-6+1e-4*abs(float(answer))]
            aligned[q,i]=summarize(same)+summarize(role)+summarize(compatibility)+summarize(rate)+[
                overlap(qt,ct),overlap(qt,tt),overlap(qwords,ct),
                len(qt&tt)/max(1,len(qt)),len(ru&tu)/max(1,len(ru)),float(bool(ru)),
                max((overlap(qt,s[1]) for s in answer_slots),default=0.),
                float(bool(answer_slots)),len(qs)/64,len(cs)/64,
                float(bool(qs)),float(bool(cs))]
            values=[]
            for u in range(len(UNIT_GROUPS)):
                values.extend([float(u in ru),float(u in tu),float(u in ru and u in tu),float(u in ru and u not in tu)])
            answer_value=float(answer) if answer is not None else 0.
            bn=numbers(body);qn={s[0] for s in qs}
            values.extend([float(answer is None),float(answer_value<0),float(abs(answer_value-round(answer_value))<1e-6),
                np.log1p(min(abs(answer_value),1e12)),float(bool(ru&tu)),float(bool(qu&cu)),
                len(ru-tu)/max(1,len(ru)),float('/' in query or 'per' in qt),
                float('/' in tail or 'per' in tt),float(bool(ru & set(range(7)))),
                float(bool(ru & {7,8})),float(9 in ru),
                float(answer is not None and Fraction(answer) in qn),
                # Conversion plausibility only, never a veto. Presence is not proof.
                float(answer is not None and any(abs(float(n)/12-answer_value)<1e-6 for n in bn)),
                float(answer is not None and any(abs(float(n)/100-answer_value)<1e-6 for n in bn)),
                float(answer is not None and any(abs(float(n)/60-answer_value)<1e-6 for n in bn))])
            for cue in CUES:
                present=cue in qt or cue in qwords
                values.extend([float(present and cue in ct),float(present and cue not in ct),
                    float(present and ('+' in body)),float(present and ('-' in body)),
                    float(present and ('*' in body or '/' in body or '×' in body or '÷' in body))])
            unit[q,i]=values
    result={'surface':surface,'alignment':np.concatenate((surface,aligned),axis=-1),
            'units':np.concatenate((surface,unit),axis=-1),
            'full':np.concatenate((surface,aligned,unit),axis=-1)}
    if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('Nonfinite quantities')
    return result
