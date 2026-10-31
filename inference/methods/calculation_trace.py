"""Conservative LaTeX arithmetic/affine checks and final-value provenance.

No files, references, labels, ID access or learned constants. Local consistency
is a fallible feature, never a correctness veto. Unsupported syntax stays unknown.
"""
import ast
from fractions import Fraction
import re
import numpy as np
from common import parse_answer
from quantity_role_features import normalize,mentions

MAX_CHARS=16000
MAX_EQUATIONS=96


def group(text,position):
    if position>=len(text) or text[position]!='{':raise ValueError('Expected brace')
    depth=1;i=position+1
    while i<len(text) and depth:
        if text[i]=='{':depth+=1
        elif text[i]=='}':depth-=1
        i+=1
    if depth:raise ValueError('Unbalanced braces')
    return text[position+1:i-1],i


def latex(text,depth=0):
    if depth>8:return '?'
    text=text.replace('×','*').replace('÷','/').replace('−','-').replace('–','-').replace(r'\$','')
    text=text.replace(r'\times','*').replace(r'\cdot','*').replace(r'\div','/')
    text=re.sub(r'\\(?:left|right|quad|qquad)\b',' ',text)
    text=re.sub(r'\\[,!;:]',' ',text);text=text.replace(r'\%', '%')
    commands=re.compile(r'\\(dfrac|tfrac|frac|boxed|text|mathrm|operatorname)\s*')
    for _ in range(48):
        m=commands.search(text)
        if not m:break
        try:
            value,end=group(text,m.end())
            if m.group(1) in ('frac','dfrac','tfrac'):
                position=end
                while position<len(text) and text[position].isspace():position+=1
                denominator,end=group(text,position);replacement='(('+latex(value,depth+1)+')/('+latex(denominator,depth+1)+'))'
            elif m.group(1)=='boxed':replacement='('+latex(value,depth+1)+')'
            else:
                prefix=text[:m.start()].rstrip()
                # A standalone text label may be an unknown mathematical
                # quantity. Dropping it would turn Cost+2=4 into +2=4.
                attached=bool(prefix) and (prefix[-1].isdigit() or prefix[-1]==')' or bool(re.search(r'\b[a-zA-Z]$',prefix)))
                replacement=value if re.fullmatch(r'[a-zA-Z]',value.strip()) else ' ' if attached else '?'
            text=text[:m.start()]+replacement+text[end:]
        except ValueError:
            text=text[:m.start()]+'?'+text[m.end():];break
    text=text.replace('{','(').replace('}',')').replace('$','').replace(r'\[','').replace(r'\]','').replace(r'\(','').replace(r'\)','')
    return text


def expression(text):
    text=text.strip().replace(',','')
    if not text or len(text)>240 or any(c in text for c in ('\\','?','^','_','[',']')):return None
    # Only edge prose can be discarded. Unknown words inside a mathematical
    # expression make the entire expression unknown; never take numeric suffixes.
    prefix=re.match(r'^\s*(?:(?:[A-Za-z]{2,})[\s:]+)+',text)
    if prefix:
        remainder=text[prefix.end():].lstrip()
        if remainder.startswith(('+','-','*','/')):return None
        text=text[prefix.end():]
    text=re.sub(r'(?:\s+[A-Za-z]{2,})+[.!?, ]*$','',text).strip(' !:')
    text=re.sub(r'(?<=\d)\s+x\s+(?=\d)','*',text)
    text=re.sub(r'(\d+(?:\.\d+)?)\s*%',r'(\1/100)',text)
    text=re.sub(r'(?<=\d)(?=[A-Za-z(])|(?<=[A-Za-z)])(?=\d|\()|(?<=\))(?=[A-Za-z])','*',text)
    if not re.fullmatch(r'[\dA-Za-z+*/().\s-]+',text):return None
    try:
        root=ast.parse(text,mode='eval').body
        if sum(1 for _ in ast.walk(root))>80:return None
        names={n.id for n in ast.walk(root) if isinstance(n,ast.Name)}
        if any(len(n)!=1 for n in names) or len(names)>1:return None
        return root,text
    except (ValueError,SyntaxError,RecursionError):return None


def affine(parsed,bindings,question_values,history):
    root,text=parsed
    def value(v,dep=frozenset(),sound=True):
        if abs(v)>10**12 or v.denominator>10**15:raise ValueError('Arithmetic limit')
        own={i for i,q in enumerate(question_values) if q==v}
        return Fraction(0),v,frozenset(dep)|own,sound
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):
            v=Fraction(ast.get_source_segment(text,n));own={i for i,q in enumerate(question_values) if q==v}
            if own:return value(v,own)
            if v in history:
                dep,sound=history[v];return value(v,dep,sound)
            return value(v)
        if isinstance(n,ast.Name):
            if n.id in bindings:
                v,dep,sound=bindings[n.id];return value(v,dep,sound)
            return Fraction(1),Fraction(0),frozenset(),True
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)):
            a,b,d,s=visit(n.operand);sign=-1 if isinstance(n.op,ast.USub) else 1;return a*sign,b*sign,d,s
        if not isinstance(n,ast.BinOp) or not isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):raise ValueError('Unsupported arithmetic')
        a,b,d,s=visit(n.left);c,e,f,t=visit(n.right);dep=d|f;sound=s and t
        if isinstance(n.op,ast.Add):out=(a+c,b+e,dep,sound)
        elif isinstance(n.op,ast.Sub):out=(a-c,b-e,dep,sound)
        elif isinstance(n.op,ast.Mult):
            if a and c:raise ValueError('Non-affine multiplication')
            out=(a*e+c*b,b*e,dep,sound)
        else:
            if c or not e:raise ValueError('Nonconstant denominator')
            out=(a/e,b/e,dep,sound)
        if abs(out[0])>10**12 or abs(out[1])>10**12:raise ValueError('Arithmetic limit')
        return value(out[1],out[2],out[3]) if not out[0] else out
    return visit(root)


def analyze(question,text):
    question,_=normalize(question);quantities=list(dict.fromkeys(v for v,*_ in mentions(question)))
    text=text if len(text)<=MAX_CHARS else text[:12000]+'\n'+text[-4000:];clean=latex(text);bindings={};history={};checked=[];roots=[];solved=0;unknown=0;equations=0;variable_checks=0
    for line in clean.splitlines()[:512]:
        if '=' not in line:continue
        # Headings/prose before a colon cannot supply a numeric operand.
        line=line.rsplit(':',1)[-1].replace('**','').strip();parts=line.split('=')[:12]
        for left,right in zip(parts,parts[1:]):
            if equations>=MAX_EQUATIONS:break
            equations+=1;l,r=expression(left),expression(right)
            if l is None or r is None:unknown+=1;continue
            try:
                a,b,d,s=affine(l,bindings,quantities,history);c,e,f,t=affine(r,bindings,quantities,history);dep=d|f
                if not a and not c:
                    good=abs(float(b-e))<=1e-7+1e-7*abs(float(e));checked.append(good);sound=s and t and good
                    if isinstance(l[0],ast.Name):variable_checks+=1
                    history[e]=(dep,sound);roots.append((e,dep,sound))
                    # A conflicting explicit assignment remains available as a
                    # declared value, but its provenance is marked inconsistent.
                    if isinstance(l[0],ast.Name):bindings[l[0].id]=(e,dep,sound)
                elif a!=c:
                    names={n.id for parsed in (l,r) for n in ast.walk(parsed[0]) if isinstance(n,ast.Name)}
                    if len(names)!=1:unknown+=1;continue
                    v=(e-b)/(a-c)
                    if abs(v)>10**12:unknown+=1;continue
                    name=next(iter(names));bindings[name]=(v,dep,s and t);history[v]=(dep,s and t);roots.append((v,dep,s and t));solved+=1
                else:unknown+=1
            except (ValueError,ZeroDivisionError,OverflowError,RecursionError):unknown+=1
        if equations>=MAX_EQUATIONS:break
    answer=parse_answer(text);answer=Fraction(answer) if answer is not None else None
    matching=[r for r in roots if answer is not None and abs(float(r[0]-answer))<=1e-7+1e-7*abs(float(answer))]
    coverage=[len(r[1])/max(1,len(quantities)) for r in matching];last=roots[-1] if roots else None
    facts={'equals_seen':equations,'numeric_checks':len(checked),'numeric_false':sum(not x for x in checked),'affine_solutions':solved,'unknown':unknown,'roots':len(roots),'final_roots':len(matching),'final_verified_roots':sum(r[2] for r in matching),'question_quantities':len(quantities)}
    features=[np.log1p(equations),np.log1p(len(checked)),np.log1p(sum(not x for x in checked)),len(checked)/max(1,equations),sum(not x for x in checked)/max(1,len(checked)),unknown/max(1,equations),
              np.log1p(solved),np.log1p(variable_checks),np.log1p(len(bindings)),np.log1p(len(roots)),np.log1p(len(matching)),sum(r[2] for r in matching)/max(1,len(matching)),
              max(coverage,default=0.),min(coverage,default=0.),float(np.mean(coverage)) if coverage else 0.,float(bool(matching)),
              float(last is not None and answer is not None and last[0]==answer),float(last[2]) if last else 0.,len(last[1])/max(1,len(quantities)) if last else 0.,len(quantities),
              float(answer is None),float(bool(checked)),float(bool(solved)),float(bool(quantities))]
    return np.asarray(features,np.float32),facts


def extract(bank):
    result=np.asarray([[analyze(row['question'],c['text'])[0] for c in row['candidates']] for row in bank],np.float32).reshape(len(bank),4,24)
    if not np.isfinite(result).all():raise ValueError('Nonfinite calculation features')
    return result
