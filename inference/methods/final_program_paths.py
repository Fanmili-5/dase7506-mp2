"""Trace candidate calculations to final-value paths, preserving uncertainty.

This representation never labels a step or candidate as correct. Numerical
equality only connects declared intermediate values; source mentions with equal
values remain ambiguous, and unsupported mathematics stays unknown.
"""
import ast
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import re
from calculation_trace import expression,latex,MAX_CHARS,MAX_EQUATIONS
from quantity_role_features import normalize,mentions
from quantity_features import requested
from quantity_typed_text import word
from common import parse_answer


@dataclass(frozen=True)
class Value:
    value: Fraction
    node: tuple
    sound: bool=True
    depth: int=0


def constant(v):
    v=Fraction(v)
    return Value(v,('c',str(v)) if v in (0,1,2,3,4,7,10,12,24,60,100,1000) else ('u',))


ZERO,ONE=constant(0),constant(1)


def operation(op,a,b):
    # Only ungrounded constant zeros/ones are simplified. Question quantities with
    # the same numerical values retain their provenance.
    if op in ('+','-') and b.node==('c','0'):return a
    if op=='+' and a.node==('c','0'):return b
    if op=='*' and (a.node==('c','0') or b.node==('c','0')):return ZERO
    if op=='/' and a.node==('c','0') and b.value:return ZERO
    if op=='*' and a.node==('c','1'):return b
    if op in ('*','/') and b.node==('c','1'):return a
    if op=='+':v=a.value+b.value
    elif op=='-':v=a.value-b.value
    elif op=='*':v=a.value*b.value
    elif op=='/':v=a.value/b.value
    else:raise ValueError('Unknown operation')
    depth=1+max(a.depth,b.depth)
    if depth>15 or abs(v)>10**12 or v.denominator>10**15:raise ValueError('Trace limit')
    return Value(v,(op,a.node,b.node),a.sound and b.sound,depth)


def affine(parsed,bindings,by_value,history):
    root,text=parsed
    def literal(value):
        if value in history:
            prior=history[value]
            if value in by_value and prior.node!=('q',tuple(by_value[value])):
                if prior.depth>=14:raise ValueError('Ambiguous trace depth')
                return Value(value,('alt',prior.node,('q',tuple(by_value[value]))),prior.sound,prior.depth+1)
            return prior
        if value in by_value:return Value(value,('q',tuple(by_value[value])))
        return constant(value)
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):return ZERO,literal(Fraction(ast.get_source_segment(text,n)))
        if isinstance(n,ast.Name):return (ZERO,bindings[n.id]) if n.id in bindings else (ONE,ZERO)
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)):
            a,b=visit(n.operand)
            return (operation('-',ZERO,a),operation('-',ZERO,b)) if isinstance(n.op,ast.USub) else (a,b)
        if not isinstance(n,ast.BinOp) or not isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):raise ValueError('Unsupported expression')
        a,b=visit(n.left);c,e=visit(n.right)
        if isinstance(n.op,ast.Add):return operation('+',a,c),operation('+',b,e)
        if isinstance(n.op,ast.Sub):return operation('-',a,c),operation('-',b,e)
        if isinstance(n.op,ast.Mult):
            if a.value and c.value:raise ValueError('Nonlinear equation')
            return operation('+',operation('*',a,e),operation('*',c,b)),operation('*',b,e)
        if c.value or not e.value:raise ValueError('Variable or zero denominator')
        coefficient=operation('/',a,e);result=operation('/',b,e)
        # A literal fraction can be the question's one numeric mention, e.g.
        # 70% or one half. Bind this only when neither operand was grounded.
        if isinstance(n.left,ast.Constant) and isinstance(n.right,ast.Constant) and result.value in by_value:
            result=Value(result.value,('q',tuple(by_value[result.value])))
        return coefficient,result
    return visit(root)


def trace(question,text):
    q,_=normalize(question);qm=mentions(q);by_value={}
    for i,(v,*_) in enumerate(qm):by_value.setdefault(v,[]).append(i)
    text=text if len(text)<=MAX_CHARS else text[:12000]+'\n'+text[-4000:];clean=latex(text).replace('<<','').replace('>>','');bindings={};history={};roots=[];checked=0;false=0;unknown=0;solved=0;equations=0
    for line in clean.splitlines()[:512]:
        if '=' not in line:continue
        line=re.sub(r'(?<![0-9A-Za-z)])\*\*([^*]+?)\*\*(?![0-9A-Za-z(])',r'\1',line.rsplit(':',1)[-1]).strip();parts=line.split('=')[:12]
        for left,right in zip(parts,parts[1:]):
            if equations>=MAX_EQUATIONS:break
            equations+=1;l,r=expression(left),expression(right)
            if l is None or r is None:unknown+=1;continue
            try:
                a,b=affine(l,bindings,by_value,history);c,e=affine(r,bindings,by_value,history)
                if not a.value and not c.value:
                    good=abs(float(b.value-e.value))<=1e-7+1e-7*abs(float(e.value));checked+=1;false+=not good
                    reverse=isinstance(l[0],ast.Constant) and isinstance(r[0],ast.BinOp);origin=e if reverse else b;declared=b.value if reverse else e.value
                    value=Value(declared,origin.node,origin.sound and good,origin.depth)
                elif a.value!=c.value:
                    names={n.id for p in (l,r) for n in ast.walk(p[0]) if isinstance(n,ast.Name)}
                    if len(names)!=1:unknown+=1;continue
                    value=operation('/',operation('-',e,b),operation('-',a,c));bindings[next(iter(names))]=value;solved+=1
                else:unknown+=1;continue
                history[value.value]=value;roots.append(value)
                if isinstance(l[0],ast.Name):bindings[l[0].id]=value
            except (ValueError,ZeroDivisionError,OverflowError,RecursionError):unknown+=1
        if equations>=MAX_EQUATIONS:break
    answer=parse_answer(text);answer=Fraction(answer) if answer is not None else None
    matching=[v for v in roots if answer is not None and abs(float(v.value-answer))<=1e-7+1e-7*abs(float(answer))]
    selected=matching[-2:]
    return selected,{'equations':equations,'checked':checked,'false':int(false),'unknown':unknown,'solved':solved,'roots':len(roots),'final_roots':len(matching)},q,qm


def context_words(text):
    return [word(t) for t in re.findall(r'[A-Za-z]+|\$',text) if len(t)>1 or t=='$']


def paths(node,suffix=()):
    result=[];stack=[(node,suffix)];visits=0
    while stack and len(result)<64 and visits<512:
        current,p=stack.pop();visits+=1
        if current[0]=='q':result.extend((i,p) for i in current[1]);continue
        if current[0] in ('c','u'):continue
        op=current[0];left={'+':'A','-':'Splus','*':'M','/':'Dnum','alt':'ambiguous'}[op];right={'+':'A','-':'Sminus','*':'M','/':'Dden','alt':'ambiguous'}[op]
        stack.append((current[2],(right,)+p));stack.append((current[1],(left,)+p))
    return result[:64]


def features(question,text,roles,parsed=None):
    roots,facts,q,qm=parsed if parsed is not None else trace(question,text);result=['trace_present:'+str(bool(roots))]
    for k in ('checked','false','unknown','solved','final_roots'):result.append(k+':'+str(min(4,facts[k])))
    ask=list(dict.fromkeys(context_words(requested(q))))[-8:];contexts=[]
    for _,_,_,start,end in qm:
        pre=context_words(q[max(0,start-60):start])[-3:];post=context_words(q[end:end+60])[:3];contexts.append(list(dict.fromkeys(pre+post)))
    used=set()
    for root in roots:
        result.append('provenance_sound:'+str(root.sound));pp=paths(root.node);used.update(i for i,p in pp)
        result.append('depth:'+str(min(8,root.depth)));result.append('source_count:'+str(min(8,len({i for i,p in pp}))))
        for i,p in pp[:64]:
            signature='_'.join(p[-4:]) or 'direct';result.extend(['path:'+signature,'slot:'+str(min(i,12))+'|'+signature])
            if roles:
                for w in contexts[i]:
                    result.append('context:'+w+'|'+signature)
                    for a in ask:result.append('ask:'+a+'|context:'+w+'|'+signature)
                for a in ask:result.append('ask:'+a+'|path:'+signature)
        if roles:
            edge_visits=[0]
            def edges(node):
                edge_visits[0]+=1
                if node[0] in ('q','c','u') or edge_visits[0]>64:return
                lp=sorted({i for i,p in paths(node[1])});rp=sorted({i for i,p in paths(node[2])});lw=sorted({w for i in lp for w in contexts[i]})[:4];rw=sorted({w for i in rp for w in contexts[i]})[:4]
                if node[0]!='alt':
                    for l in lw:
                        for r in rw:result.append('pair:'+l+'|'+node[0]+'|'+r)
                edges(node[1]);edges(node[2])
            edges(root.node)
    result.append('missing_count:'+str(min(8,len(qm)-len(used))))
    if roles and roots:
        for i in range(len(qm)):
            if i not in used:
                for w in contexts[i][:4]:
                    for a in ask:result.append('ask:'+a+'|missing:'+w)
    return result[:2048],facts


def rows(bank,roles):
    return [features(row['question'],c['text'],roles)[0] for row in bank for c in row['candidates']]
