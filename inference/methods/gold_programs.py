"""Bounded arithmetic templates from provided training reference markers.

The extractor has no file access. Only training callers may pass a reference.
Programs contain ordered *distinct-value* question slots and fixed constants;
they never contain arbitrary reference-answer literals. Equal-value mentions
are explicitly collapsed, so these slots do not claim to identify entity roles.
"""
import ast
from fractions import Fraction
import itertools
import re
from quantity_role_features import normalize
from round2_features import NUMBER

CONSTANTS = frozenset((0, 1, 2, 3, 4, 7, 10, 12, 24, 60, 100, 1000))
MARKER = re.compile(r'<<([^<>]{1,240})>>')
OPERATORS = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}
MAX_PROGRAMS = 12
MAX_OPERATIONS = 15


def question_values(question):
    normalized, word_count = normalize(question)
    values, positions = [], []
    duplicates = 0
    for m in list(NUMBER.finditer(normalized))[:64]:
        try:
            # Percent stays in percentage points. Gold arithmetic explicitly
            # divides by 100, which remains a public conversion constant.
            value = Fraction(m.group().replace(',', '').rstrip('%'))
        except (ValueError, ZeroDivisionError):
            continue
        if abs(value) > 10**9:
            continue
        if value in values:
            duplicates += 1
            continue
        values.append(value)
        positions.append((m.start(), m.end()))
    return values, {'equal_value_mentions_collapsed': duplicates,
                    'english_number_phrases_normalized': word_count,
                    'normalized_question': normalized,
                    'positions': positions}


def parse_numeric(text):
    text = text.replace(',', '').replace('−', '-').strip()
    if not re.fullmatch(r'[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?', text):
        raise ValueError('Not a numeric value')
    return Fraction(text)


def arithmetic(text):
    text = text.replace(',', '').replace('×', '*').replace('÷', '/').replace('−', '-')
    if not re.fullmatch(r'[\d+*/().\s-]{1,200}', text):
        raise ValueError('Unsupported arithmetic characters')
    root = ast.parse(text.strip(), mode='eval').body
    def visit(node, depth=0):
        if depth > 16:
            raise ValueError('Arithmetic nesting limit')
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return ('v', Fraction(ast.get_source_segment(text.strip(), node)))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            child = visit(node.operand, depth+1)
            if child[0] != 'v':
                return ('-', ('v', Fraction(0)), child) if isinstance(node.op, ast.USub) else child
            return ('v', -child[1] if isinstance(node.op, ast.USub) else child[1])
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            return (OPERATORS[type(node.op)], visit(node.left, depth+1), visit(node.right, depth+1))
        raise ValueError('Unsupported arithmetic syntax')
    return visit(root)


def evaluate(program, values=()):
    op = program[0]
    if op == 'v':
        return program[1]
    if op == 'q':
        return values[program[1]]
    if op == 'c':
        if program[1] not in CONSTANTS:
            raise ValueError('Undeclared constant')
        return Fraction(program[1])
    a, b = evaluate(program[1], values), evaluate(program[2], values)
    if op == '+': value = a+b
    elif op == '-': value = a-b
    elif op == '*': value = a*b
    elif op == '/': value = a/b
    else: raise ValueError('Unsupported operation')
    if abs(value) > 10**12 or value.denominator > 10**12:
        raise ValueError('Arithmetic value limit')
    return value


def size(program):
    return 0 if program[0] in ('q', 'c') else 1+size(program[1])+size(program[2])


def slots(program):
    if program[0] == 'q': return {program[1]}
    if program[0] == 'c': return set()
    return slots(program[1]) | slots(program[2])


def canonical(op, a, b):
    if op in ('+', '*') and repr(a) > repr(b): a, b = b, a
    return (op, a, b)


def ordered(programs):
    # A small deterministic beam avoids exploding aliases of the same number.
    return sorted(set(programs), key=lambda p: (size(p), -len(slots(p)), repr(p)))[:MAX_PROGRAMS]


def bind(tree, values, history):
    if tree[0] == 'v':
        value = tree[1]
        result = [('q', i) for i, v in enumerate(values) if v == value]
        result += history.get(value, [])
        # Never replace an available question quantity or computed intermediate
        # with a coincidentally equal public constant (e.g. 16-4 -> literal 12).
        if not result and value.denominator == 1 and value.numerator in CONSTANTS:
            result.append(('c', value.numerator))
        return ordered(result)
    left, right = bind(tree[1], values, history), bind(tree[2], values, history)
    return ordered(canonical(tree[0], a, b) for a, b in itertools.product(left, right)
                   if size(a)+size(b)+1 <= MAX_OPERATIONS)


def extract(question, reference):
    """Return reusable labels, with unknown cases kept unknown.

    A reference's numerical result is used to verify its own training label,
    never serialized as a question-independent program constant.
    """
    values, meta = question_values(question)
    if not values or len(values) > 12:
        return {'programs': [], 'reason': 'question_quantity_scope', **meta}
    try:
        final = parse_numeric(reference.rsplit('####', 1)[-1])
    except ValueError:
        return {'programs': [], 'reason': 'missing_exact_final', **meta}
    history = {}
    roots = []
    counts = {'markers': 0, 'valid_arithmetic': 0, 'grounded_steps': 0,
              'rounded_or_inconsistent_markers': 0, 'unsupported_markers': 0}
    for marker in list(MARKER.finditer(reference))[:32]:
        counts['markers'] += 1
        text = marker.group(1)
        if text.count('=') != 1:
            counts['unsupported_markers'] += 1
            continue
        lhs, rhs = text.split('=')
        try:
            tree, value = arithmetic(lhs), parse_numeric(rhs)
            actual = evaluate(tree)
            if actual != value:
                counts['rounded_or_inconsistent_markers'] += 1
                continue
            counts['valid_arithmetic'] += 1
            programs = [p for p in bind(tree, values, history) if slots(p)
                        and evaluate(p, values) == value]
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError):
            counts['unsupported_markers'] += 1
            continue
        if programs:
            counts['grounded_steps'] += 1
            history[value] = ordered(history.get(value, [])+programs)
            if value == final:
                roots += programs
    roots = ordered(p for p in roots if size(p)>0 and evaluate(p, values) == final)
    return {'programs': roots, 'reason': 'covered' if roots else 'no_grounded_final_program',
            'question_slots': len(values), **counts, **meta}


def dumps(program):
    import json
    return json.dumps(program, separators=(',', ':'))


def loads(text):
    import json
    def convert(p):
        if p[0] in ('q', 'c'): return tuple(p)
        return (p[0], convert(p[1]), convert(p[2]))
    return convert(json.loads(text))
