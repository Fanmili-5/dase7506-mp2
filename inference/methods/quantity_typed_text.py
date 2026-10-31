"""Question-conditioned number slots for a prospective sparse verifier.

No vocabulary, fitted state, labels or ID access. This is a fallible number
binding representation, not equation solving or step-correctness supervision.
"""
from fractions import Fraction
import re
from quantity_features import STOP,UNIT_MAP,requested
from round2_features import NUMBER

CONSTANTS={0,1,2,10,12,60,100}
OPERATOR={'+':'op_add','-':'op_sub','*':'op_mul','/':'op_div','=':'op_equal',
          '(':'paren_left',')':'paren_right','%':'percent'}

def number_value(token):
    try:
        result=Fraction(token.replace(',','').rstrip('%'))/(100 if token.endswith('%') else 1)
        return result if abs(result)<=10**12 else None
    except (ValueError,ZeroDivisionError):return None

def word(token):
    token=token.lower()
    if token in UNIT_MAP:return f'unit_{UNIT_MAP[token]}'
    if token=='$':return 'unit_7'
    if len(token)>3 and token.endswith('s') and token not in ('less','times'):return token[:-1]
    return token

def words(text):
    return [word(t) for t in re.findall(r'[a-zA-Z]+|\$',text) if t.lower() not in STOP]

def mapping(question):
    result={};contexts={}
    for m in list(NUMBER.finditer(question))[:64]:
        value=number_value(m.group())
        if value is None or value in result:continue
        name=f'qslot_{len(result)}';result[value]=name
        local=words(question[max(0,m.start()-30):m.start()])[-2:]+words(question[m.end():m.end()+30])[:2]
        contexts[name]=list(dict.fromkeys(local))[:4]
    return result,contexts

def normalize(text,mode,table,contexts,expand=False):
    text=(text if len(text)<=16000 else text[:12000]+' truncated_middle '+text[-4000:]).replace('×','*').replace('÷','/').replace('−','-')
    def replace(m):
        value=number_value(m.group())
        if mode=='raw':return ' '+m.group().replace(',','')+' '
        if '/' in m.group() and value not in table:
            # Keep unbound compact arithmetic division visible. A fraction equal
            # to a question quantity is instead treated as that numeric slot.
            parts=m.group().split('/')
            normalized=[]
            for part in parts:
                val=number_value(part)
                if mode=='bound' and val in table:normalized.append(table[val])
                elif mode=='bound' and val is not None and val in CONSTANTS:normalized.append(f'const_{int(val)}')
                else:normalized.append('number' if mode=='masked' else 'intermediate_number')
            return ' '+ ' op_div '.join(normalized)+' '
        if mode=='masked':return ' number '
        if value in table:
            slot=table[value]
            return ' '+slot+' '+(' '.join('qcontext_'+x for x in contexts[slot]) if expand else '')+' '
        if value is not None and value in CONSTANTS:return f' const_{int(value)} '
        return ' intermediate_number '
    text=NUMBER.sub(replace,text)
    tokens=re.findall(r'[a-zA-Z][a-zA-Z_0-9]*|\$|[+*/=()%-]|\d+(?:\.\d+)?',text)
    result=[]
    for t in tokens:
        if t in OPERATOR:result.append(OPERATOR[t])
        elif t.startswith(('qslot_','qcontext_','const_')) or t in ('number','intermediate_number'):result.append(t)
        else:result.append(word(t))
    return result if len(result)<=2048 else result[:1536]+result[-512:]

def text_rows(bank,mode):
    if mode not in ('raw','masked','bound'):raise ValueError('Unknown number binding scope')
    rows=[]
    for row in bank:
        table,contexts=mapping(row['question']);query=words(requested(row['question']))[-8:]
        question=normalize(row['question'],mode,table,contexts)
        for candidate in row['candidates']:
            text=candidate['text'];body=normalize(text,mode,table,contexts,expand=mode=='bound')
            # Operator/query interactions are shared by masked/bound controls.
            basic=normalize(text,mode,table,contexts)
            operators=list(dict.fromkeys(t for t in basic if t.startswith('op_')))
            events=[f'ask_{q}__{op}' for q in query for op in operators]
            if mode=='bound':
                for k in range(1,len(basic)-1):
                    if basic[k].startswith('op_') and basic[k-1].startswith(('qslot_','const_')) and basic[k+1].startswith(('qslot_','const_')):
                        triple='__'.join(basic[k-1:k+2]);events.append('binding__'+triple)
                        events.extend(f'ask_{q}__{triple}' for q in query)
            rows.append(' '.join(['question']+question+['solution']+body+['relations']+events[:512]))
    return rows
