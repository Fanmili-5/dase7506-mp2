"""Bounded existing-program sequences for a prospective non-generating scorer.

No fitted state, files, IDs or reference lookup. Reference markers are accepted
only by the training helper; candidate programs use that candidate's own text.
"""
from fractions import Fraction
import itertools,re
from common import parse_answer
from gold_programs import (CONSTANTS,MARKER,MAX_OPERATIONS,arithmetic,bind,evaluate,
                           ordered,parse_numeric,size,slots)
from final_program_paths import trace
from quantity_role_features import normalize
from round2_features import NUMBER

MAX_VARIANTS=8
MAX_PREFIX_TOKENS=31
OPS={'+':'ADD','-':'SUB','*':'MUL','/':'DIV'}


def question_values(question):
    text,count=normalize(question);values=[];positions=[]
    for m in list(NUMBER.finditer(text))[:64]:
        try:v=Fraction(m.group().replace(',','').rstrip('%'))
        except (ValueError,ZeroDivisionError):continue
        if abs(v)>10**9:continue
        values.append(v);positions.append((m.start(),m.end()))
    return values,{'normalized_question':text,'positions':positions,'equal_value_mentions_collapsed':0,'equal_value_aliases_retained':len(values)-len(set(values)),'english_number_phrases_normalized':count}


def normalize_program(program):
    if program[0] in ('q','c'):return program
    op=program[0];a=normalize_program(program[1]);b=normalize_program(program[2])
    def fold(operation,children,neutral):
        children=sorted(children,key=repr)
        if not children:return ('c',neutral)
        p=children[0]
        for child in children[1:]:p=(operation,p,child)
        return p
    if op=='+':
        parts=[]
        def add(p):
            if p[0]=='+':add(p[1]);add(p[2])
            elif p!=('c',0):parts.append(p)
        add(a);add(b);return fold('+',parts,0)
    if op in ('*','/'):
        numerator=[];denominator=[]
        def mul(p,sign=1):
            if p[0]=='*':mul(p[1],sign);mul(p[2],sign)
            elif p[0]=='/':mul(p[1],sign);mul(p[2],-sign)
            elif p!=('c',1):(numerator if sign==1 else denominator).append(p)
        mul(a);mul(b,1 if op=='*' else -1);top=fold('*',numerator,1)
        return ('/',top,fold('*',denominator,1)) if denominator else top
    if op=='-':return a if b==('c',0) else ('-',a,b)
    raise ValueError('Unsupported program node')


def prefix(program):
    if program[0]=='q':return ['Q'+str(program[1])]
    if program[0]=='c':
        if program[1] not in CONSTANTS:raise ValueError('Non-public constant')
        return ['C'+str(program[1])]
    return [OPS[program[0]]]+prefix(program[1])+prefix(program[2])


def decode_prefix(tokens):
    def read(i):
        if i>=len(tokens):raise ValueError('Truncated prefix')
        t=tokens[i];inverse={v:k for k,v in OPS.items()}
        if t in inverse:
            a,j=read(i+1);b,k=read(j);return (inverse[t],a,b),k
        if t.startswith('Q') and t[1:].isdigit() and 0<=int(t[1:])<12:return ('q',int(t[1:])),i+1
        if t.startswith('C') and t[1:].isdigit() and int(t[1:]) in CONSTANTS:return ('c',int(t[1:])),i+1
        raise ValueError('Unknown prefix token')
    p,n=read(0)
    if n!=len(tokens):raise ValueError('Extra prefix tokens')
    return p


def retain(programs,values,final):
    found={}
    for p in programs:
        try:
            if evaluate(p,values)!=final:continue
            p=normalize_program(p);tokens=prefix(p)
            if not slots(p) or size(p)>MAX_OPERATIONS or len(tokens)>MAX_PREFIX_TOKENS or evaluate(p,values)!=final:continue
            assert decode_prefix(tokens)==p;found[tuple(tokens)]=p
        except (ValueError,ZeroDivisionError,OverflowError,RecursionError,IndexError):continue
    keys=sorted(found,key=lambda x:(len(x),x))[:MAX_VARIANTS]
    return [found[k] for k in keys]


def question_tokens(question):
    values,meta=question_values(question);text=meta['normalized_question'];table={p:i for i,p in enumerate(meta['positions'])}
    def replace(m):
        try:v=Fraction(m.group().replace(',','').rstrip('%'))
        except (ValueError,ZeroDivisionError):return ' NUMBER_UNKNOWN '
        position=(m.start(),m.end());token='QSLOT_'+str(table[position]) if position in table and table[position]<12 else 'NUMBER_UNKNOWN'
        return ' '+token+(' percent ' if m.group().endswith('%') else ' ')
    text=NUMBER.sub(replace,text);return [t.lower() for t in re.findall(r'[A-Za-z_][A-Za-z_0-9]*|[$?.,;:!]',text)]


def reference_programs(question,reference):
    values,meta=question_values(question);answer=parse_answer(reference)
    if not values or len(values)>12 or answer is None:return {'programs':[],'prefixes':[],'reason':'question_or_reference_scope'}
    final=Fraction(answer);history={};roots=[]
    # Keep the last grounded marker reaching the provided final, rather than
    # prefer an accidentally equal early intermediate just because it is short.
    for marker in list(MARKER.finditer(reference))[:32]:
        if marker.group(1).count('=')!=1:continue
        left,right=marker.group(1).split('=')
        try:
            tree,value=arithmetic(left),parse_numeric(right)
            if evaluate(tree)!=value:continue
            programs=[p for p in bind(tree,values,history) if slots(p) and evaluate(p,values)==value]
        except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError):continue
        if programs:
            history[value]=ordered(history.get(value,[])+programs)
            if value==final:roots=programs
    kept=retain(roots,values,final);return {'programs':kept,'prefixes':[prefix(p) for p in kept],'question_slots':len(values),'equal_value_mentions_collapsed':0,'equal_value_aliases_retained':meta['equal_value_aliases_retained'],'reason':'covered' if kept else 'no_grounded_final_program'}


def candidate_programs(question,text):
    values,meta=question_values(question);answer=parse_answer(text)
    if not values or len(values)>12 or answer is None:return {'programs':[],'prefixes':[],'reason':'question_or_candidate_scope'}
    roots,facts,q,qm=trace(question,text);table={p:i for i,p in enumerate(meta['positions'])}
    def keep(items):return sorted(set(items),key=lambda p:(size(p),repr(p)))[:MAX_VARIANTS]
    memo={};visits=[0]
    def convert(node):
        key=id(node)
        if key in memo:return memo[key]
        visits[0]+=1
        if visits[0]>512:return []
        result=convert_node(node);memo[key]=result;return result
    def convert_node(node):
        if node[0]=='u':return []
        if node[0]=='c':
            v=Fraction(node[1]);return [('c',int(v))] if v.denominator==1 and int(v) in CONSTANTS else []
        if node[0]=='q':
            result=[]
            for index in node[1]:
                position=(qm[index][3],qm[index][4]);raw=q[position[0]:position[1]]
                try:v=Fraction(raw.replace(',','').rstrip('%'))
                except (ValueError,ZeroDivisionError):continue
                if position not in table:continue
                p=('q',table[position]);result.append(('/',p,('c',100)) if raw.endswith('%') else p)
            return keep(result)
        if node[0]=='alt':return keep(convert(node[1])+convert(node[2]))
        return keep((node[0],a,b) for a,b in itertools.product(convert(node[1]),convert(node[2])) if size(a)+size(b)+1<=MAX_OPERATIONS)
    programs=[]
    for root in roots:programs.extend(convert(root.node))
    kept=retain(programs,values,Fraction(answer));return {'programs':kept,'prefixes':[prefix(p) for p in kept],'question_slots':len(values),'reason':'covered' if kept else 'no_grounded_self_consistent_final_program','trace_final_roots':len(roots),'trace_false_equations':facts['false'],'conversion_visit_limit_reached':visits[0]>512,'only_current_question_and_candidate_used':True}
