"""Recover existing, explicit candidate arithmetic without inventing operations.

The old interpretation is retained. Additional expressions must be numeric,
arithmetically consistent and grounded in question quantities or earlier explicit
subexpressions. Text annotations and explicit currency/percent scales are fallible
interpretations, never correctness labels or hard selection rules.
"""
from fractions import Fraction
import itertools,re
from calculation_trace import latex,MAX_CHARS,MAX_EQUATIONS
from gold_programs import arithmetic,evaluate,size,slots,CONSTANTS,MAX_OPERATIONS
from program_sequence_data import (candidate_programs as original_programs,
    question_values,retain,prefix,decode_prefix,MAX_VARIANTS)
from unit_propagation import prepare,lemma,STANDARD
from common import parse_answer

ANNOTATION=re.compile(r'(?<=[0-9)])\s*\(([A-Za-z][A-Za-z ]{1,45})\)(?=\s*(?:[+*/=)\-]|$))')
MATH_WORDS=set('plus minus less more fewer times divided multiplied added subtract subtraction sum difference product quotient than equals equal percent percentage squared cubed power x y z'.split())
INLINE=re.compile(r'\\\((.{1,800}?)\\\)|\\\[(.{1,800}?)\\\]|\$\$(.{1,800}?)\$\$|(?<!\\)\$([^$\n]{1,800})\$(?![0-9])')
EDGE=re.compile(r'^(?:(?:so|thus|therefore|hence|then|now)\b[, :]*)\s*',re.I)
LABEL=re.compile(r'[A-Za-z][A-Za-z\s\-]{1,60}')


def question_context(question):
    values,meta=question_values(question);q=meta['normalized_question'];units=set(STANDARD)
    _,_,source,_,_,_=prepare(question)
    for states in source:
        for state in states:
            if state.dimensions is not None:units.update(k for k,v in state.dimensions)
    units-=MATH_WORDS
    aliases={}
    for i,(v,(start,end)) in enumerate(zip(values,meta['positions'])):
        raw=q[start:end];after=q[end:end+24]
        if raw.endswith('%') or re.match(r'\s*(?:cents?|pennies)\b',after,re.I):
            aliases.setdefault(v/100,[]).append(('/',('q',i),('c',100)))
    return values,units,aliases


def strip_annotations(text,units):
    # Parenthesized words immediately after a numeric atom are annotations only
    # when they contain neither arithmetic words nor a single-letter variable.
    def paren(match):
        tokens=match.group(1).lower().split()
        return ' ' if all(len(w)>1 and w not in MATH_WORDS for w in tokens) else match.group()
    text=ANNOTATION.sub(paren,text)
    # Bare units must already be proposed by the question unit parser. Unknown
    # variables and relation words are not silently discarded.
    pattern=r'(?<=[0-9)])\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,2})(?=\s*(?:[+*/=)\-]|$))'
    def unit(match):
        tokens=match.group(1).lower().split()
        return ' ' if all(lemma(w) in units and w not in MATH_WORDS for w in tokens) else match.group()
    return re.sub(pattern,unit,text)


def segments(text,units):
    text=text[:MAX_CHARS] if len(text)<=MAX_CHARS else text[:12000]+'\n'+text[-4000:]
    # A candidate's final marker is used only to identify the selected root.
    text=text.split('####',1)[0]
    for raw in text.splitlines()[:512]:
        if '=' not in raw:continue
        fragments=[next(g for g in m.groups() if g is not None) for m in INLINE.finditer(raw) if '=' in m.group()]
        for part in fragments or [raw]:
            part=re.sub(r'(?<![0-9A-Za-z)])\*\*([^*]+?)\*\*(?![0-9A-Za-z(])',r'\1',part)
            part=latex(part).replace('<<','').replace('>>','').strip()
            # A colon heading has no equation semantics; require no prior '='.
            if ':' in part and ':' in part.split('=',1)[0]:part=part.split(':',1)[1].strip()
            part=EDGE.sub('',part)
            variants=[part]
            # A leading Markdown list marker is ambiguous with unary minus.
            # Keep both; explicit equality and source grounding decide support.
            if re.match(r'^[-*]\s+',part):variants.append(re.sub(r'^[-*]\s+','',part))
            for variant in variants:yield strip_annotations(variant.strip().rstrip('.;'),units)


def numeric(text):
    text=text.strip().replace(',','')
    text=re.sub(r'(\d+(?:\.\d+)?)\s*%',r'(\1/100)',text)
    return arithmetic(text)


def beam(programs):
    return sorted(set(programs),key=lambda p:(size(p),-len(slots(p)),repr(p)))[:MAX_VARIANTS]


def additional_programs(question,text,context=None):
    values,units,aliases=context if context is not None else question_context(question);final=parse_answer(text)
    if not values or len(values)>12 or final is None:return [],{'scope_rejected':True}
    final=Fraction(final);history={};roots=[];counts=dict(equations=0,consistent=0,inconsistent=0,unknown=0,label_expressions=0,grounded_expressions=0,scale_aliases=sum(map(len,aliases.values())))
    def bind(node,pending):
        if node[0]=='v':
            v=node[1];found=[('q',i) for i,x in enumerate(values) if x==v]+history.get(v,[])+aliases.get(v,[])
            if not found and v.denominator==1 and int(v) in CONSTANTS:found=[('c',int(v))]
            return beam(found)
        a,b=bind(node[1],pending),bind(node[2],pending)
        found=beam((node[0],x,y) for x,y in itertools.product(a,b) if size(x)+size(y)+1<=MAX_OPERATIONS)
        grounded=[p for p in found if slots(p)]
        if grounded:
            v=evaluate(node);pending[v]=beam(pending.get(v,[])+grounded)
        return found
    for segment in segments(text,units):
        parts=segment.split('=')[:12]
        for left,right in zip(parts,parts[1:]):
            if counts['equations']>=MAX_EQUATIONS:break
            counts['equations']+=1
            try:
                left_label=bool(LABEL.fullmatch(left.strip()))
                a=None if left_label else numeric(left);b=numeric(right)
                if a is not None and evaluate(a)!=evaluate(b):counts['inconsistent']+=1;continue
                counts['label_expressions' if left_label else 'consistent']+=1
                pending={}
                for tree in [b] if a is None else [a,b]:
                    programs=bind(tree,pending);value=evaluate(tree)
                    grounded=[p for p in programs if slots(p)]
                    if grounded:
                        counts['grounded_expressions']+=1
                        if value==final:roots.extend(grounded)
                # Only publish after a full numeric expression/equality passed.
                for value,programs in pending.items():history[value]=beam(history.get(value,[])+programs)
            except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError):counts['unknown']+=1
        if counts['equations']>=MAX_EQUATIONS:break
    return retain(roots,values,final),counts


def candidate_programs(question,text,context=None):
    old=original_programs(question,text);new,facts=additional_programs(question,text,context)
    prefixes=[list(x) for x in old['prefixes']];seen={tuple(x) for x in prefixes};added=0
    for p in new:
        tokens=prefix(p)
        if tuple(tokens) not in seen and len(prefixes)<MAX_VARIANTS:
            seen.add(tuple(tokens));prefixes.append(tokens);added+=1
    return dict(programs=[decode_prefix(p) for p in prefixes],prefixes=prefixes,
        reason='covered' if prefixes else 'no_grounded_self_consistent_final_program',
        old_prefix_count=len(old['prefixes']),additional_prefix_count=added,annotation_trace=facts,
        original_prefixes_preserved=True,only_current_question_and_candidate_used=True)
