"""Fallible calculation-use and requested-object signals, without references.

All grammars and limits are fixed. A mention is not proof that a quantity was
used correctly, and a failed local check never removes a candidate.
"""
from fractions import Fraction
import re
import numpy as np
from common import parse_answer
from quantity_features import extract as old_extract, requested, tokens, units, overlap
from round2_features import NUMBER, numbers, safe_arithmetic

SMALL = dict(zip(('zero one two three four five six seven eight nine ten eleven '
                 'twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split(), range(20)))
TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10)))
FRACTIONS = {'half': Fraction(1, 2), 'halves': Fraction(1, 2),
             'quarter': Fraction(1, 4), 'quarters': Fraction(1, 4),
             'third': Fraction(1, 3), 'thirds': Fraction(1, 3),
             'dozen': Fraction(12), 'dozens': Fraction(12)}
BASE = '(?:' + '|'.join(sorted(set(SMALL)|set(TENS)|set(FRACTIONS), key=len, reverse=True)) + ')'
REST = '(?:' + BASE + '|hundred|thousand|and|a)'
PHRASE = re.compile(r'\b(?:' + BASE + r'|a(?=\s+(?:half|quarter|third|dozen)\b))'
                    r'(?:[ -]+' + REST + r'){0,15}\b', re.I)
TARGET_STOP = set('there here any number amount total altogether answer final solution find calculate '
                  'determine need needed would could should can get given result left after before '
                  'then now all if s length long far cost money time'.split())
OP = re.compile(r'=|(?<=\d)\s*[+*/-]\s*(?=[(-]?\d)')
RHS = re.compile(r'=\s*([-+*/().\d, \t]{1,100})')
MAX_CHARS = 16000


def phrase_value(text):
    parts=text.lower().replace('-', ' ').split();value=Fraction(0);current=0;i=0
    while i<len(parts):
        w=parts[i]
        if w=='and' and parts[i+1:i+2]==['a'] and i+2<len(parts) and parts[i+2] in FRACTIONS:
            value+=current+FRACTIONS[parts[i+2]];current=0;i+=3;continue
        previous=parts[i-1] if i else None
        if w in SMALL:
            if previous in SMALL or (previous in TENS and not 1<=SMALL[w]<=9):return None
            current+=SMALL[w]
        elif w in TENS:
            if previous in SMALL or previous in TENS:return None
            current+=TENS[w]
        elif w=='hundred':current=max(1,current)*100
        elif w=='thousand':value+=max(1,current)*1000;current=0
        elif w in FRACTIONS:current=Fraction((value+current) or 1)*FRACTIONS[w];value=Fraction(0)
        elif w=='and':
            if previous not in ('hundred','thousand') or i+1==len(parts):return None
        elif w=='a':
            if i+1==len(parts) or parts[i+1] not in FRACTIONS:return None
        else:return None
        i+=1
    return value+current


def normalize(text):
    text=text if len(text)<=MAX_CHARS else text[:12000]+'\n'+text[-4000:]
    text=text.replace('×','*').replace('÷','/').replace('−','-')
    text=text.replace(r'\times','*').replace(r'\cdot','*').replace(r'\div','/')
    text=re.sub(r'\\(?:d?frac)\s*\{\s*([-\d.]+)\s*\}\s*\{\s*([-\d.]+)\s*\}', r'(\1/\2)', text)
    count=0
    def replace(m):
        nonlocal count
        value=phrase_value(m.group())
        if value is None:return m.group()
        count+=1
        return str(value.numerator) if value.denominator==1 else str(float(value))
    return PHRASE.sub(replace,text),count


def mentions(text):
    rows=[]
    for m in list(NUMBER.finditer(text))[:96]:
        try:value=Fraction(m.group().replace(',','').rstrip('%'))/(100 if m.group().endswith('%') else 1)
        except (ValueError,ZeroDivisionError):continue
        if abs(value)>10**12:continue
        context=text[max(0,m.start()-35):min(len(text),m.end()+35)]
        rows.append((value,set(tokens(context)),units(context),m.start(),m.end()))
    return rows


def summary(xs):
    return [float(np.mean(xs)) if xs else 0.,max(xs,default=0.),min(xs,default=0.)]


def calculation_segments(text):
    # Lines are bounded. Prose containing a rate or a displayed result can be a
    # false positive; the features explicitly report coverage rather than truth.
    result=[];prior=''
    for line in text.splitlines()[:512]:
        line=line[:1800]
        if OP.search(line):result.append((line,(prior+' '+line)[-2000:]))
        if line.strip() and not re.fullmatch(r'[\s$*#_-]+',line):prior=line[-180:]
    return result


def feature_groups(question,text):
    q,qcount=normalize(question);c,ccount=normalize(text)
    body=c.rsplit('####',1)[0];qm=mentions(q);cm=mentions(body)
    qn={x[0] for x in qm};cn={x[0] for x in cm};literal=numbers(question)
    query=set(tokens(requested(q)));target=query-TARGET_STOP
    segments=calculation_segments(body);calc='\n'.join(x[0] for x in segments)
    calc_mentions=mentions(calc);operands={x[0] for x in calc_mentions}
    answer=parse_answer(text);av=Fraction(answer) if answer is not None else None
    word=[qcount,ccount,len(qn),len(cn),len(qn-literal),
          len(qn&cn)/max(1,len(qn)),len(qn-cn)/max(1,len(qn)),
          len((qn-literal)&cn)/max(1,len(qn-literal)),float(bool(qn)),float(bool(qcount)),
          float(av is not None and av in qn),float(av is not None and av in cn)]
    exact=[];role=[];unit=[];repeated=[]
    for value,context,u,_,_ in qm:
        matching=[x for x in calc_mentions if x[0]==value or x[0]==-value]
        exact.append(float(bool(matching)))
        role.append(max((overlap(context,x[1]) for x in matching),default=0.))
        unit.append(max((float(bool(u&x[2])) for x in matching),default=0.))
        repeated.append(sum(x[0]==value for x in qm)>1)
    calc_features=summary(exact)+summary(role)+summary(unit)+[
        len(segments),len(calc_mentions),len(operands),len(qn&operands)/max(1,len(qn)),
        len((qn&cn)-operands)/max(1,len(qn)),len(qn-operands)/max(1,len(qn)),
        float(bool(segments)),float(bool(qm)),float(any(repeated)),float(np.mean(repeated)) if repeated else 0.,
        sum('=' in x[0] for x in segments),sum('*' in x[0] for x in segments),
        sum('/' in x[0] for x in segments),sum('+' in x[0] for x in segments),
        sum('-' in x[0] for x in segments)]
    containing=[context for line,context in segments if av is not None and av in numbers(line)]
    final_context=containing[-1] if containing else ''
    last_context=segments[-1][1] if segments else ''
    at=set(tokens(final_context));lt=set(tokens(last_context));tail=set(tokens(body[-180:]))
    values=[]
    for line,_ in segments:
        for m in RHS.finditer(line):
            # Unknown symbolic suffixes must not be silently truncated to a value.
            end=m.end(1)-len(m.group(1))+len(m.group(1).rstrip())
            if end<len(line) and line[end].isalnum():continue
            v=safe_arithmetic(m.group(1))
            if v is not None:values.append(v)
    last=values[-1] if values else None
    target_features=[len(target),float(bool(target)),float(bool(containing)),
        overlap(target,at),overlap(target,lt),overlap(target,tail),
        len(target&at)/max(1,len(target)),len(target&lt)/max(1,len(target)),len(target&tail)/max(1,len(target)),
        len(units(requested(q))&units(final_context))/max(1,len(units(requested(q)))),
        float(last is not None),len(values),float(av is not None),
        float(last is not None and av is not None and abs(float(last-av))<=1e-6+1e-4*abs(float(av))),
        float(last is not None and av is not None and last==-av and av!=0),
        float(last is not None and av is not None and last.denominator!=1 and av.denominator==1),
        min(20.,abs(float(last-av))/max(1.,abs(float(av)))) if last is not None and av is not None else 0.,
        float('per' in query),float('per' in at),float(bool(target-at)),float(bool(target-lt))]
    return word,calc_features,target_features


def extract(bank):
    old=old_extract(bank)['full'];groups=[[],[],[]]
    for row in bank:
        rows=[feature_groups(row['question'],candidate['text']) for candidate in row['candidates']]
        for k in range(3):groups[k].append([r[k] for r in rows])
    word,calc,target=(np.asarray(g,np.float32).reshape(len(bank),4,d) for g,d in zip(groups,(12,24,21)))
    result={'full':old,'word':np.concatenate((old,word),axis=-1),
            'calculation':np.concatenate((old,word,calc),axis=-1),
            'target':np.concatenate((old,word,target),axis=-1),
            'role_all':np.concatenate((old,word,calc,target),axis=-1)}
    if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('Nonfinite role features')
    return result
