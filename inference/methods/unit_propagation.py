"""Bounded, fallible unit/rate propagation through candidate calculations.

Only question/candidate inputs are used. Unknown units remain unknown; competing
unit interpretations are retained. These are soft features, never a veto.
"""
from dataclasses import dataclass,replace
from fractions import Fraction
import re
import numpy as np
from final_program_paths import trace,paths
from quantity_role_features import normalize,mentions
from quantity_features import requested

BEAM=8
ALIASES={'$':'money','dollar':'money','dollars':'money','cent':'money','cents':'money','penny':'money','pennies':'money','people':'person','persons':'person','feet':'foot','inches':'inch','metres':'meter','metre':'meter','kilometres':'kilometer','kilometre':'kilometer','litres':'liter','litre':'liter'}
TIME={'second','minute','hour','day','week','month','year'}
LENGTH={'inch','foot','yard','mile','meter','kilometer','centimeter'}
MASS={'gram','kilogram','ounce','pound'}
VOLUME={'liter','gallon'}
STANDARD=TIME|LENGTH|MASS|VOLUME|{'money'}
STOP=set('a an the and or of in on at to for from with by as per each every how many much more less fewer extra additional total remaining new other is are was were be being been has have had do does did will would can could should must cost costs earn earns earned pay pays paid spend spent make makes made work works worked used use uses last lasts need needs needed remain remains left long far now then still she he they it his her their there this that these those'.split())
RATE=re.compile(r'\b(?:per|each|every|a|an|in\s+(?:a|an|each|every)|for\s+(?:each|every))\s+([A-Za-z]+)',re.I)
BEFORE_RATE=re.compile(r'\b(?:each|every)\s+([A-Za-z]+)',re.I)
CONVERSIONS={7:[('day','week')],12:[('inch','foot')],24:[('hour','day')],60:[('minute','hour'),('second','minute')],100:[('centimeter','meter')],1000:[('meter','kilometer'),('gram','kilogram')]}


def lemma(token):
    token=token.lower()
    if token in ALIASES:return ALIASES[token]
    if token.endswith('ies') and len(token)>4:token=token[:-3]+'y'
    elif token.endswith(('ches','shes','xes','sses')):token=token[:-2]
    elif token.endswith('s') and len(token)>3 and token not in ('less','times'):token=token[:-1]
    return ALIASES.get(token,token)


def words(text):return re.findall(r'[A-Za-z]+|\$',text.lower())


def dim(unit):return ((unit,1),) if unit else ()


def combine_dims(a,b,sign=1):
    if a is None or b is None:return None
    d=dict(a)
    for k,v in b:d[k]=d.get(k,0)+sign*v
    if len(d)>12 or any(abs(v)>8 for v in d.values()):return None
    return tuple(sorted((k,v) for k,v in d.items() if v))


def coarse(d):
    if d is None:return None
    result={}
    for k,v in d:
        key='@time' if k in TIME else '@length' if k in LENGTH else '@mass' if k in MASS else '@volume' if k in VOLUME else k
        result[key]=result.get(key,0)+v
    return tuple(sorted((k,v) for k,v in result.items() if v))


def distance(a,b):
    aa,bb=dict(a),dict(b)
    return min(16,sum(abs(aa.get(k,0)-bb.get(k,0)) for k in aa.keys()|bb.keys()))


@dataclass(frozen=True)
class Unit:
    dimensions: tuple|None
    conflicts: int=0
    unknown: int=0
    known: int=0
    cost: float=0.
    rates: int=0
    conversions: int=0


def beam(states):
    unique={}
    for s in states:
        key=(s.dimensions,s.conflicts,s.unknown,s.known,s.rates,s.conversions)
        if key not in unique or s.cost<unique[key].cost:unique[key]=s
    key=lambda s:(s.conflicts,s.unknown,s.cost,-s.known,repr(s.dimensions))
    known=sorted((s for s in unique.values() if not s.unknown),key=key);unknown=sorted((s for s in unique.values() if s.unknown),key=key)
    if known and unknown:
        kept=known[:4]+unknown[:4];rest=sorted(known[4:]+unknown[4:],key=key);return kept+rest[:BEAM-len(kept)]
    return sorted(unique.values(),key=key)[:BEAM]


def heads(text):
    ww=words(text);result=[]
    for token in ww[:5]:
        if token in ('per','each','every','in','for','of','and','or'):break
        if result and (token in ('is','are','was','were','has','have','had','do','does','did','will','would','can','could','should') or token.endswith(('ed','ing'))):break
        if token in STOP:continue
        u=lemma(token)
        if u not in result:result.append(u)
        if u in STANDARD or len(result)==2:break
    return result


def quantity_units(q,mention):
    value,_,_,start,end=mention;left=re.split(r'[.!?;:,\n]',q[max(0,start-90):start])[-1];right=re.split(r'[.!?;:,\n]',q[end:end+90])[0];right=re.split(r'\d',right,maxsplit=1)[0];h=heads(right);currency=bool(re.search(r'\$\s*$',left));text=q[start:end];scalar=text.endswith('%') or bool(re.match(r'\s*(?:times|percent)\b',right,re.I)) or (abs(value)<=1 and bool(re.match(r'\s*(?:as|of)\b',right,re.I)));num=['money'] if currency else [] if scalar else h;den=[]
    for m in RATE.finditer(right[:55]):
        token=lemma(m.group(1))
        if m.group().lower().startswith(('a ','an ')) and token not in TIME and len([w for w in words(right[:m.start()]) if w not in STOP])>1:continue
        if token not in STOP and token not in ('one','another'):den.append(token)
    before=list(BEFORE_RATE.finditer(left))
    if before:
        token=lemma(before[-1].group(1))
        if token not in STOP:den.append(token)
    # A week/month/year containing N smaller periods is an implicit rate.
    if not den and num and num[0] in TIME:
        m=re.search(r'\b(?:a|an)\s+([A-Za-z]+)\s+(?:has|contains)\s*$',left,re.I)
        if m and lemma(m.group(1)) in TIME:den.append(lemma(m.group(1)))
    states=[]
    for unit in num:
        base=dim(unit);cost=0. if unit in STANDARD else .25
        if den:
            for d in list(dict.fromkeys(den))[:2]:states.append(Unit(combine_dims(base,dim(d),-1),known=1,cost=cost,rates=1))
            states.append(Unit(base,known=1,cost=cost+1.))
        else:states.append(Unit(base,known=1,cost=cost))
    if scalar:
        if den:
            states.extend(Unit(combine_dims((),dim(d),-1),known=1,rates=1) for d in den[:2])
        else:states.append(Unit((),known=1))
    # An unparsed quantity is never silently made dimensionless.
    states.append(Unit(None,unknown=1,cost=1. if states else 0.))
    return beam(states),bool(den),bool(scalar)


def target_units(q):
    ask=requested(q);m=re.search(r'\bhow\s+many\s+(.*)',ask,re.I);fragment=m.group(1) if m else ''
    if not fragment and re.search(r'\b(?:money|dollars?|cost|price|earn|pay|spend)\b|\$',ask,re.I):fragment='money '+ask
    h=heads(fragment)
    den=[]
    for match in RATE.finditer(fragment):
        token=lemma(match.group(1))
        if match.group().lower().startswith(('a ','an ')) and token not in TIME and len([w for w in words(fragment[:match.start()]) if w not in STOP])>1:continue
        if token not in STOP:den.append(token)
    result=[]
    if h and h[0]=='times':result=[Unit(())]
    else:
        for unit in h[:2]:
            if den:result.extend(Unit(combine_dims(dim(unit),dim(d),-1),cost=.25) for d in den[:2])
            else:result.append(Unit(dim(unit),cost=.25))
    return beam(result or [Unit(None,unknown=1)])


def prepare(question):
    q,_=normalize(question);qm=mentions(q);units=[];rate=[];scalar=[]
    for m in qm:
        u,r,s=quantity_units(q,m);units.append(u);rate.append(r);scalar.append(s)
    return q,qm,units,rate,scalar,target_units(q)


def propagate(roots,source_units,seen_units):
    memo={};visits=[0];limited=[False]
    def run(node):
        key=id(node)
        if key in memo:return memo[key]
        if visits[0]>=256:limited[0]=True;return [Unit(None,unknown=1)]
        visits[0]+=1
        op=node[0]
        if op=='q':out=beam([s for i in node[1] for s in source_units[i]])
        elif op=='u':out=[Unit(None,unknown=1)]
        elif op=='c':
            value=Fraction(node[1]);out=[Unit(())]
            for numerator,denominator in CONVERSIONS.get(value,[]):
                if numerator in seen_units and denominator in seen_units:out.append(Unit(combine_dims(dim(numerator),dim(denominator),-1),cost=.5,conversions=1))
        elif op=='alt':out=beam([replace(s,cost=s.cost+.25) for s in run(node[1])+run(node[2])])
        else:
            a,b=run(node[1]),run(node[2]);out=[]
            for x in a:
                for y in b:
                    conflict=x.conflicts+y.conflicts;unknown=x.unknown+y.unknown
                    if op in ('*','/'):d=combine_dims(x.dimensions,y.dimensions,1 if op=='*' else -1)
                    elif op=='-' and node[1]==('c','0'):d=y.dimensions
                    elif x.dimensions is None or y.dimensions is None:d=None
                    elif x.dimensions==y.dimensions:d=x.dimensions
                    else:d=None;conflict+=1
                    out.append(Unit(d,conflict,unknown,x.known+y.known,x.cost+y.cost,x.rates+y.rates,x.conversions+y.conversions))
            out=beam(out)
        memo[key]=out;return out
    result=beam([s for root in roots for s in run(root.node)]) if roots else []
    return result,visits[0],limited[0]


FEATURE_NAMES=('question_sources','known_source_fraction','rate_source_fraction','scalar_source_fraction','ambiguous_source_fraction','known_target','target_hypotheses','has_final_root','final_roots','has_output_unit','has_no_unknown_state','min_additive_conflicts','min_conflicts_known_sources','has_known_source_state','min_unknown_sources','mean_unknown_sources','min_assumption_cost','mean_assumption_cost','min_fine_distance','min_coarse_distance','has_comparable_fine_units','fine_compatible','coarse_compatible','fine_compatible_fraction','coarse_compatible_fraction','conflict_free_fraction','known_output_fraction','min_conversion_assumptions','max_conversion_assumptions','max_rates_used','min_rates_used','used_source_fraction','missing_rate_fraction','missing_known_fraction','arithmetic_sound_fraction','node_visits','node_limit_reached','output_hypotheses','fine_match_min_cost','fine_match_extra_cost','coarse_match_extra_cost','fine_match_rate_coverage')


def analyze(question,text,parsed=None,prepared=None):
    prepared=prepared or prepare(question);q,qm,source,rate,scalar,target=prepared;parsed=parsed or trace(question,text);roots=parsed[0];seen={lemma(w) for w in words(q+' '+text[:16000])};out,visits,limited=propagate(roots,source,seen);known_targets=[s.dimensions for s in target if s.dimensions is not None];valid=[s for s in out if s.dimensions is not None];known=[s for s in out if not s.unknown];used={i for root in roots for i,p in paths(root.node)};known_source=[any(s.dimensions is not None for s in states) for states in source]
    fine=[distance(s.dimensions,t) for s in valid for t in known_targets];rough=[distance(coarse(s.dimensions),coarse(t)) for s in valid for t in known_targets];fc=[s for s in valid if any(s.dimensions==t for t in known_targets)];cc=[s for s in valid if any(coarse(s.dimensions)==coarse(t) for t in known_targets)]
    n=max(1,len(qm));m=max(1,len(out));values=[np.log1p(len(qm)),sum(known_source)/n,sum(rate)/n,sum(scalar)/n,sum(sum(s.dimensions is not None for s in states)>1 for states in source)/n,float(bool(known_targets)),np.log1p(len(target)),float(bool(roots)),np.log1p(len(roots)),float(bool(valid)),float(bool(known)),min((s.conflicts for s in out),default=0),min((s.conflicts for s in known),default=0),float(bool(known)),min((s.unknown for s in out),default=0),sum(s.unknown for s in out)/m,min((s.cost for s in out),default=0),sum(s.cost for s in out)/m,min(fine,default=0),min(rough,default=0),float(bool(fine)),float(bool(fc)),float(bool(cc)),len(fc)/m,len(cc)/m,sum(s.conflicts==0 for s in out)/m,len(valid)/m,min((s.conversions for s in out),default=0),max((s.conversions for s in out),default=0),max((s.rates for s in out),default=0),min((s.rates for s in out),default=0),len(used)/n,sum(rate[i] for i in range(len(qm)) if i not in used)/n,sum(known_source[i] for i in range(len(qm)) if i not in used)/n,sum(r.sound for r in roots)/max(1,len(roots)),np.log1p(visits),float(limited),np.log1p(len(out))]
    low=min((s.cost for s in out),default=0.);fine_cost=min((s.cost for s in fc),default=low+4.);coarse_cost=min((s.cost for s in cc),default=low+4.);values.extend([min(16,fine_cost),min(16,fine_cost-low),min(16,coarse_cost-low),min(1,max((s.rates for s in fc),default=0)/max(1,sum(rate)))])
    values=np.asarray(values,np.float32);assert len(values)==len(FEATURE_NAMES) and np.isfinite(values).all()
    facts={'sources':len(qm),'known_sources':sum(known_source),'rate_sources':sum(rate),'target_known':bool(known_targets),'final_roots':len(roots),'output_units':len(valid),'fine_compatible':bool(fc),'coarse_compatible':bool(cc),'fine_match_extra_cost':fine_cost-low,'minimum_conflicts_known_sources':min((s.conflicts for s in known),default=None),'node_limit':limited}
    return values,facts


def extract(bank):
    values=[]
    for row in bank:
        prepared=prepare(row['question']);values.append([analyze(row['question'],c['text'],prepared=prepared)[0] for c in row['candidates']])
    return np.asarray(values,np.float32).reshape(len(bank),4,len(FEATURE_NAMES))
