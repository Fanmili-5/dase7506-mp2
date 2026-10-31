"""Reference-free surface features. Unknown arithmetic stays unknown.

No fitted state, labels, question IDs, networking or dataset access lives here.
"""

import ast
from fractions import Fraction
import math
import re

import numpy as np

from selector import answers_for

NUMBER = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+(?:\.\d+)?)?%?")
EQUATION = re.compile(r"([-+*/().\d, \t]{1,100})[ \t]*=[ \t]*([-+*/().\d, \t]{1,100})")
UNIT = re.compile(r"\b(seconds?|minutes?|hours?|days?|weeks?|months?|years?|dollars?|cents?|miles?|kilometers?|meters?|pounds?|ounces?|liters?|gallons?)\b", re.I)
NAMES = ("log_char_count", "log_word_count", "log_number_count", "question_number_recall",
         "new_number_fraction", "log_equation_count", "checked_equation_fraction",
         "correct_equation_fraction", "wrong_equation_fraction", "answer_mentioned_earlier",
         "unit_overlap", "unparsed_equation_fraction")


def safe_arithmetic(text):
    """Return a bounded rational, or None for unsupported/malformed expressions."""
    text = text.strip().replace(",", "")
    if not text or len(text) > 100:
        return None
    try:
        root = ast.parse(text, mode="eval")
        if sum(1 for _ in ast.walk(root)) > 64:
            return None

        def visit(n):
            if isinstance(n, ast.Constant) and type(n.value) in (int, float):
                value = Fraction(str(n.value))
            elif isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
                value = visit(n.operand) * (-1 if isinstance(n.op, ast.USub) else 1)
            elif isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                a, b = visit(n.left), visit(n.right)
                if isinstance(n.op, ast.Add): value = a + b
                elif isinstance(n.op, ast.Sub): value = a - b
                elif isinstance(n.op, ast.Mult): value = a * b
                else: value = a / b
            else:
                raise ValueError("Unsupported arithmetic")
            if abs(value) > 10**12 or value.denominator > 10**18:
                raise ValueError("Out of bounds")
            return value

        return visit(root.body)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError, RecursionError):
        return None


def equation_checks(text):
    total = text.count("=")
    known = []
    normalized = text.replace("×", "*").replace("÷", "/")
    for match in EQUATION.finditer(normalized):
        prefix = normalized[normalized.rfind("\n", 0, match.start()) + 1:match.start()].rstrip()
        if prefix and prefix[-1].isalnum():
            # Never interpret a numeric suffix of x+2=4 as the equation +2=4.
            # Inline prose without a delimiter is conservatively unknown too.
            continue
        right_end = match.end(2) - len(match.group(2)) + len(match.group(2).rstrip())
        if right_end < len(normalized) and normalized[right_end].isalnum():
            continue
        a, b = map(safe_arithmetic, match.groups())
        if a is not None and b is not None:
            known.append(abs(float(a - b)) <= 1e-6 + 1e-4 * abs(float(b)))
    return total, known


def numbers(text):
    result = set()
    for match in NUMBER.finditer(text):
        token = match.group().replace(",", "")
        try:
            value = Fraction(token.rstrip("%")) / (100 if token.endswith("%") else 1)
            if abs(value) <= 10**12:
                result.add(value)
        except (ValueError, ZeroDivisionError):
            continue
    return result


def surface_features(bank):
    out = np.zeros((len(bank), 4, len(NAMES)), dtype=np.float32)
    for q, row in enumerate(bank):
        qnumbers = numbers(row["question"])
        qunits = {x.lower().rstrip("s") for x in UNIT.findall(row["question"])}
        for i, candidate in enumerate(row["candidates"]):
            text = candidate["text"]
            cnumbers = numbers(text)
            cunits = {x.lower().rstrip("s") for x in UNIT.findall(text)}
            total, checks = equation_checks(text)
            earlier = text.rsplit("####", 1)[0] if "####" in text else text[:-80]
            from common import parse_answer

            answer = parse_answer(text)
            # Coverage and unknown fraction explicitly distinguish missing checks
            # from a checked-but-false equation. They never veto a candidate.
            out[q, i] = (math.log1p(len(text)), math.log1p(len(text.split())),
                math.log1p(len(NUMBER.findall(text))),
                len(qnumbers & cnumbers) / max(1, len(qnumbers)),
                len(cnumbers - qnumbers) / max(1, len(cnumbers)), math.log1p(total),
                len(checks) / max(1, total), sum(checks) / max(1, len(checks)),
                sum(not x for x in checks) / max(1, len(checks)),
                float(answer is not None and Fraction(answer) in numbers(earlier)),
                len(qunits & cunits) / max(1, len(qunits)),
                max(0, total - len(checks)) / max(1, total))
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite surface features")
    return out


def groups(bank):
    """Candidate -> answer-group membership, without any reference answer."""
    mask = np.zeros((len(bank), 4, 4), dtype=np.float32)
    valid = np.zeros((len(bank), 4), dtype=bool)
    vote = np.zeros((len(bank), 4, 1), dtype=np.float32)
    for q, parsed in enumerate(answers_for(bank)):
        keys = []
        for i, value in enumerate(parsed):
            if value is None:
                continue
            if value not in keys:
                keys.append(value)
            g = keys.index(value)
            mask[q, g, i] = 1
            valid[q, i] = True
        vote[q, :, 0] = (mask[q].sum(axis=1) @ mask[q]) / 4
    return mask, valid, vote


def raw_dense(bank, embeddings, config, surface=None):
    e = np.asarray(embeddings, dtype=np.float32)
    if e.shape != (len(bank), 4, 1024) or not np.isfinite(e).all():
        raise ValueError("Expected finite [N,4,1024] input embeddings")
    mode = config.get("dense", "numeric")
    _, _, vote = groups(bank)
    if mode == "v1_count":
        return np.concatenate((e, vote, np.zeros((len(bank), 4, 4), np.float32)), axis=-1)
    n = surface_features(bank) if surface is None else surface
    if mode in ("matched", "count"):
        n = np.zeros_like(n)
    if config.get("without_count", False):
        vote = np.zeros_like(vote)
    parts = [vote, n]
    if config.get("embedding", True):
        parts.insert(0, e)
    return np.concatenate(parts, axis=-1)


def text_rows(bank, scope):
    out = []
    for row, parsed in zip(bank, answers_for(bank)):
        for candidate, answer in zip(row["candidates"], parsed):
            final = "invalid" if answer is None else str(answer)
            if scope == "full":
                out.append("question " + row["question"] + " solution " + candidate["text"])
            elif scope == "qfinal":
                out.append("question " + row["question"] + " final " + final)
            elif scope == "final":
                out.append("final " + final)
            else:
                raise ValueError("Unknown text scope")
    return out
