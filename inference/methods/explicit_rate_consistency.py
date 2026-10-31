"""Conservative literal rate claims, compared as exact rational numbers.

This is a fallible input feature, not a semantic oracle or a hard veto.
No references, identifiers, learned vocabulary or external models are used.
"""
from dataclasses import dataclass
from fractions import Fraction
import re

SMALL = dict(zip(('zero one two three four five six seven eight nine ten eleven '
                 'twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split(), range(20)))
TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10)))
NUMBER_WORDS = set(SMALL) | set(TENS) | {'hundred', 'thousand', 'half', 'quarter', 'dozen'}
TOKEN = re.compile(r'\d[\d,]*(?:\.\d+)?(?:/\d+)?|[A-Za-z]+|[^\s]')
VERBS = set(('has have holds hold contains contain makes make produces produce yields yield '
             'carries carry fits fit provides provide earns earn pays pay costs cost weighs weigh '
             'takes take needs need sells sell buys buy uses use travels travel runs run walks walk '
             'is are was were be can will would must should did does do gets get').split())
STOP = VERBS | set(('the a an and or but to of from in on at with for by as per each every '
                    'that which who this these those it its his her their he she they we you '
                    'then than so when if after before during into out more less total altogether '
                    'left remaining enough now initially originally later first second third '
                    'how many much what how whose all both only apiece respectively approximately').split())
DAYS = set('monday tuesday wednesday thursday friday saturday sunday'.split())
CHANGED = re.compile(r'\b(?:instead|revised|changed|changes|increased|decreased|discounted|discount|new rate)\b', re.I)
PLURALS = {'feet': 'foot', 'children': 'child', 'people': 'person', 'men': 'man', 'women': 'woman',
           'hours': 'hour', 'minutes': 'minute', 'seconds': 'second', 'dollars': 'dollar', 'cents': 'cent'}


@dataclass(frozen=True)
class Fact:
    numerator: tuple
    denominator: tuple
    value: Fraction
    scope: tuple
    start: int
    end: int
    grammar: str


def singular(word):
    if word in PLURALS:
        return PLURALS[word]
    if word.endswith('ies') and len(word) > 4:
        return word[:-3] + 'y'
    if word.endswith('s') and not word.endswith(('ss', 'us')) and len(word) > 3:
        return word[:-1]
    return word


def number(tokens, i):
    if i >= len(tokens):
        return None
    word = tokens[i][0]
    if word[0].isdigit():
        try:
            value = Fraction(word.replace(',', ''))
        except (ValueError, ZeroDivisionError):
            return None
        return (value, i + 1) if abs(value) <= 10**12 else None
    if word not in NUMBER_WORDS:
        return None
    if word in ('half', 'quarter', 'dozen'):
        return ({'half': Fraction(1, 2), 'quarter': Fraction(1, 4), 'dozen': Fraction(12)}[word], i + 1)
    total = current = 0
    j = i
    previous = None
    while j < min(len(tokens), i + 9):
        w = tokens[j][0]
        if w in SMALL:
            if previous in SMALL or (previous in TENS and SMALL[w] > 9):
                break
            current += SMALL[w]
        elif w in TENS:
            if previous in SMALL or previous in TENS:
                break
            current += TENS[w]
        elif w == 'hundred':
            current = max(1, current) * 100
        elif w == 'thousand':
            total += max(1, current) * 1000
            current = 0
        elif w == '-' and previous in TENS and j + 1 < len(tokens) and tokens[j + 1][0] in SMALL:
            j += 1
            continue
        elif w == 'and' and previous in ('hundred', 'thousand') and j + 1 < len(tokens) and tokens[j + 1][0] in NUMBER_WORDS:
            j += 1
            continue
        else:
            break
        previous = w
        j += 1
    return Fraction(total + current), j


def noun(tokens, i):
    words = []
    j = i
    while j < len(tokens) and len(words) < 3:
        w = tokens[j][0]
        if not w.isalpha() or w in STOP or w in NUMBER_WORDS or w in DAYS:
            break
        words.append(singular(w))
        j += 1
    # Reject an overlong noun rather than silently truncate its meaning.
    if j < len(tokens) and len(words) == 3:
        w = tokens[j][0]
        if w.isalpha() and w not in STOP and w not in DAYS:
            return None
    return (tuple(words), j) if words else None


def context(text, start, end):
    left = max(text.rfind('.', 0, start), text.rfind('\n', 0, start), text.rfind('?', 0, start)) + 1
    right = len(text)
    for mark in ('.', '\n', '?', ';'):
        k = text.find(mark, end)
        if k >= 0:
            right = min(right, k)
    sentence = text[left:right]
    if CHANGED.search(sentence):
        return None
    days = tuple(sorted(set(re.findall(r'\b[a-z]+\b', sentence.lower())) & DAYS))
    return days if len(days) <= 1 else None


def normalize_money(unit, value):
    if unit == ('cent',):
        return ('money',), value / 100
    if unit in (('dollar',), ('money',)):
        return ('money',), value
    return unit, value


def extract(text):
    text = str(text)[:16000]
    text = text.split('####', 1)[0]
    tokens = [(m.group().lower(), m.start(), m.end()) for m in TOKEN.finditer(text)][:6000]
    found = []

    def add(numunit, denunit, value, start, end, grammar):
        scope = context(text, start, end)
        if scope is None or not numunit or not denunit or value <= 0:
            return
        numunit, value = normalize_money(numunit, value)
        if numunit == denunit:
            return
        fact = Fact(numunit, denunit, value, scope, start, end, grammar)
        if fact not in found and len(found) < 96:
            found.append(fact)

    # N units per M objects; N units each/every object; $N per object.
    for i in range(len(tokens)):
        parsed = number(tokens, i)
        if parsed is None:
            continue
        value, j = parsed
        unit = noun(tokens, j)
        if unit:
            numunit, j = unit
        elif i > 0 and tokens[i - 1][0] == '$':
            numunit = ('money',)
        else:
            continue
        if j >= len(tokens):
            continue
        marker = tokens[j][0]
        if marker == 'for' and j + 1 < len(tokens) and tokens[j + 1][0] in ('each', 'every'):
            j += 1
            marker = tokens[j][0]
        if marker not in ('per', 'each', 'every', 'a', 'an', '/'):
            continue
        j += 1
        denominator_number = number(tokens, j)
        denominator = Fraction(1)
        if denominator_number:
            denominator, j = denominator_number
        den = noun(tokens, j)
        if not den or denominator <= 0:
            continue
        denunit, end = den
        # A slash whose RHS itself contains a rate is a nested expression.
        if marker == '/' and end < len(tokens) and tokens[end][0] in ('per', 'each', 'every', '/'):
            continue
        add(numunit, denunit, value / denominator, tokens[i][1], tokens[end - 1][2], 'post_rate')

    # Each container [can] hold N contents. A/an costs clauses are explicit too.
    for i, (word, _, _) in enumerate(tokens):
        if word not in ('each', 'every', 'a', 'an'):
            continue
        den = noun(tokens, i + 1)
        if not den:
            continue
        denunit, j = den
        if j < len(tokens) and tokens[j][0] in ('can', 'will'):
            j += 1
        if j >= len(tokens) or tokens[j][0] not in VERBS - {'is', 'are', 'was', 'were', 'be', 'can', 'will', 'would', 'must', 'should', 'do', 'does', 'did'}:
            continue
        j += 1
        currency = j < len(tokens) and tokens[j][0] == '$'
        j += int(currency)
        parsed = number(tokens, j)
        if parsed is None:
            continue
        value, j = parsed
        num = noun(tokens, j)
        if num:
            numunit, end = num
        elif currency:
            numunit, end = ('money',), j
        else:
            continue
        add(numunit, denunit, value, tokens[i][1], tokens[end - 1][2], 'distributive')

    # Containers that can make N contents each (no inferred uniform division).
    for i, (word, _, _) in enumerate(tokens):
        if word not in ('that', 'which') or i == 0:
            continue
        j = i + 1
        if j < len(tokens) and tokens[j][0] == 'can':
            j += 1
        if j >= len(tokens) or tokens[j][0] not in VERBS:
            continue
        j += 1
        parsed = number(tokens, j)
        if not parsed:
            continue
        value, j = parsed
        num = noun(tokens, j)
        if not num:
            continue
        numunit, end = num
        if end >= len(tokens) or tokens[end][0] not in ('each', 'apiece'):
            continue
        # Require a local noun immediately before that/which; keep modifiers.
        start = i - 1
        while start > max(0, i - 3) and tokens[start - 1][0].isalpha() and tokens[start - 1][0] not in STOP and tokens[start - 1][0] not in NUMBER_WORDS:
            start -= 1
        den = noun(tokens, start)
        if not den or den[1] != i:
            continue
        add(numunit, den[0], value, tokens[start][1], tokens[end][2], 'relative_each')
    return found


def analyze(question, candidate):
    given = {}
    for fact in extract(question):
        key = (fact.numerator, fact.denominator, fact.scope)
        given.setdefault(key, set()).add(fact.value)
    comparisons = contradictions = unknown = 0
    witnesses = []
    for fact in extract(candidate):
        key = (fact.numerator, fact.denominator, fact.scope)
        values = given.get(key, set())
        if len(values) != 1:
            unknown += 1
            continue
        comparisons += 1
        expected = next(iter(values))
        if fact.value != expected:
            contradictions += 1
            witnesses.append({'numerator': fact.numerator, 'denominator': fact.denominator,
                              'scope': fact.scope, 'expected': str(expected), 'claimed': str(fact.value),
                              'candidate_start': fact.start, 'candidate_end': fact.end, 'grammar': fact.grammar})
    return {'comparisons': comparisons, 'contradictions': contradictions, 'unknown': unknown}, witnesses


def feature_matrix(bank):
    import numpy as np
    result = np.zeros((len(bank), 4), np.float64)
    for i, row in enumerate(bank):
        for j, candidate in enumerate(row['candidates']):
            result[i, j] = analyze(row['question'], candidate['text'])[0]['contradictions']
    return np.log1p(result)
