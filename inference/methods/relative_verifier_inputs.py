"""Independent implementation of the relative-verifier design.

Design reference: Anchmy/dase7506-mp2-group-relative-verifier, commit
08813bf6ad85fea5242fc40392f25dd8ba26e1cf. No upstream weights/data are used.
Arithmetic checks are fallible input signals, not step correctness labels.
"""
import ast
from fractions import Fraction
import re
import numpy as np
from common import parse_answer

NUM = re.compile(r'\d+(?:\.\d+)?')
ARITH = ('eq_count', 'bad_fraction', 'has_bad', 'first_bad', 'last_bad', 'bad_position',
         'exact_fraction', 'final_last_rhs', 'final_any_rhs', 'final_any_lhs')
QUEST = ('operand_count', 'given_fraction', 'derived_fraction', 'new_fraction', 'new_count',
         'first_new', 'has_new', 'answer_given', 'integer', 'negative', 'relative_magnitude',
         'count_noninteger', 'equations_per_given', 'all_accounted', 'money_question',
         'given_count', 'denominator', 'round_tens', 'zero', 'subunit', 'magnitude', 'tenths')


def normalize_equations(text):
    text = re.sub(r'\*\*|__', ' ', str(text)[:16000])
    text = re.sub(r'\\(?:text|mathrm|mbox|operatorname)\{[^{}]*\}', ' ', text)
    text = re.sub(r'\\boxed\{([^{}]*)\}', r' \1 ', text)
    for _ in range(4):
        updated = re.sub(r'\\(?:d|t)?frac\{([^{}]*)\}\{([^{}]*)\}', r'((\1)/(\2))', text)
        if updated == text:
            break
        text = updated
    text = re.sub(r'\\(?:left|right|bigl|bigr|Bigl|Bigr)\b|\\[(),;:!]|\\quad|\\qquad', ' ', text)
    text = re.sub(r'\([^()\d]*\)', ' ', text)
    text = re.sub(r'(\d+(?:\.\d+)?)\s*(?:\\?%|percent)\s*(?:of\s*)?', r'((\1)/100)*', text)
    replacements = [(r'\\times|\\cdot|\\ast|\\star|[×·⋅*]', '*'),
                    (r'\\div|÷', '/'), (r'\\approx|≈', '='),
                    (r'\\le|\\ge|\\leq|\\geq|[<>]=?', ' '),
                    (r'\\%|%', ' '), (r'\\\$|\$|\\,|\\;|\\!', ' '),
                    (r'[−–—]', '-'), (r'[→⇒]', ' '), (r'\\[a-zA-Z]+', ' '), (r'[{}]', ' ')]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', text)


def evaluate_arithmetic(text):
    text = re.sub(r'(?<![\d.])0+(\d)', r'\1', text.strip())
    if not text or len(text) > 4096 or not set(text) <= set('0123456789.+-*/() ') or not any(c.isdigit() for c in text):
        return None
    try:
        root = ast.parse(text, mode='eval')
        if sum(1 for _ in ast.walk(root)) > 512:
            return None

        def visit(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                result = float(node.value)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            elif isinstance(node, ast.BinOp):
                a, b = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Add): result = a + b
                elif isinstance(node.op, ast.Sub): result = a - b
                elif isinstance(node.op, ast.Mult): result = a * b
                elif isinstance(node.op, ast.Div): result = a / b
                elif isinstance(node.op, ast.FloorDiv): result = a // b
                elif isinstance(node.op, ast.Pow) and abs(b) <= 12: result = a ** b
                else: raise ValueError('Unsupported arithmetic')
            else:
                raise ValueError('Unsupported expression')
            if not np.isfinite(result) or abs(result) > 1e100:
                raise ValueError('Out of bounds')
            return result
        return visit(root.body)
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError, OverflowError, RecursionError):
        return None


def equations(text):
    output = []
    for run in re.findall(r'[0-9()+\-*/=.\s]{3,}', normalize_equations(text)):
        parts = run.split('=')
        for left, right in zip(parts, parts[1:]):
            left, right = left.strip(), right.strip()
            if not re.search(r'[+\-*/]', left):
                continue
            a, b = evaluate_arithmetic(left), evaluate_arithmetic(right)
            if a is not None and b is not None:
                output.append((a, b, left))
    return output


def near(a, b, tolerance=1e-6):
    return abs(a - b) <= tolerance * max(1., abs(a), abs(b))


def scalar_features(question, text, answer):
    eq = equations(text)
    try:
        final = float(Fraction(answer)) if answer is not None else None
        if final is not None and not np.isfinite(final): final = None
    except (ValueError, ZeroDivisionError, OverflowError):
        final = None
    a = np.zeros(10, np.float64)
    if eq:
        exact = [near(left, right) for left, right, _ in eq]
        bad = [not near(left, right) and abs(left - right) > max(.01, .005 * max(1., abs(left))) for left, right, _ in eq]
        a[:] = [np.log1p(len(eq)), np.mean(bad), any(bad), bad[0], bad[-1],
                next((i / len(eq) for i, b in enumerate(bad) if b), 0.), np.mean(exact),
                final is not None and near(eq[-1][1], final),
                final is not None and any(near(right, final) for _, right, _ in eq),
                final is not None and any(near(left, final) for left, _, _ in eq)]
    question_numbers = [float(x) for x in NUM.findall(re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', str(question)))]
    seen, given, derived, fresh, first = set(), 0, 0, 0, -1.
    for i, (_, _, expression) in enumerate(eq):
        for token in NUM.findall(expression):
            value = float(token)
            if any(near(value, old) for old in seen):
                continue
            seen.add(value)
            if any(near(value, number) for number in question_numbers): given += 1
            elif any(near(value, row[1]) for row in eq[:i]): derived += 1
            else:
                fresh += 1
                if first < 0: first = i / max(1, len(eq))
    count = len(seen)
    denominator = max(1, count)
    q = np.zeros(22, np.float64)
    q[:8] = [np.log1p(count), given / denominator, derived / denominator, fresh / denominator,
             np.log1p(fresh), first, fresh > 0,
             final is not None and any(near(final, number) for number in question_numbers)]
    if final is not None:
        fraction = Fraction(answer)
        integer = fraction.denominator == 1
        maximum = max(question_numbers) if question_numbers else 1.
        q[8:12] = [integer, final < 0, min(1., np.log10(1 + abs(final) / (abs(maximum) + 1e-9)) / 4),
                   bool(re.search(r'\bhow many\b', question, re.I)) and abs(final - round(final)) > 1e-9]
        q[16:] = [min(4., np.log2(fraction.denominator)), final != 0 and abs(final) >= 10 and abs(final) % 10 == 0,
                  final == 0, 0 < abs(final) < 1, min(1., np.log10(1 + abs(final)) / 5),
                  abs(final * 10 - round(final * 10)) < 1e-9]
    q[12:16] = [len(eq) / max(1, len(question_numbers)), count > 0 and fresh == 0,
                bool(re.search(r'\$|dollar|cent\b|cost|price|pay|earn|spend', question, re.I)),
                min(1., len(question_numbers) / 12)]
    return np.concatenate((a, q))


def inputs(bank, embeddings):
    e = np.asarray(embeddings, np.float64)
    if e.shape != (len(bank), 4, 1024) or not np.isfinite(e).all():
        raise ValueError('Expected finite [N,4,1024] embeddings')
    parsed = [[parse_answer(c['text']) for c in row['candidates']] for row in bank]
    centered = e - e.mean(axis=1, keepdims=True)
    relative = centered / (np.linalg.norm(centered, axis=-1, keepdims=True).mean(axis=1, keepdims=True) + 1e-6)
    scalar = np.empty((len(bank), 4, 32), np.float64)
    for i, row in enumerate(bank):
        for j, candidate in enumerate(row['candidates']):
            scalar[i, j] = scalar_features(row['question'], candidate['text'], parsed[i][j])
    scalar -= scalar.mean(axis=1, keepdims=True)
    scalar /= np.sqrt(np.mean(scalar**2, axis=1, keepdims=True)) + 1e-3
    x = np.concatenate((relative, scalar), axis=-1)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite relative features')
    return x, parsed


def group_scores(scores, parsed, bonus):
    scores = np.asarray(scores, np.float64)
    output = np.full_like(scores, -1e6)
    for i, answers in enumerate(parsed):
        groups = {}
        for j, answer in enumerate(answers):
            if answer is not None: groups.setdefault(answer, []).append(j)
        if not groups:
            output[i] = scores[i]
        for indices in groups.values():
            output[i, indices] = max(scores[i, j] for j in indices) + bonus * np.log(len(indices))
    return output
