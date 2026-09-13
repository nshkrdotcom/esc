"""Shared evaluator normalization for categorical and scalar numeric answers."""
from decimal import Decimal, DecimalException, InvalidOperation


def answers_equal(predicted: str | None, target: str) -> bool:
    if predicted is None:
        return False
    left, right = predicted.strip().casefold(), target.strip().casefold()
    try:
        a, b = Decimal(left), Decimal(right)
    except InvalidOperation:
        return left == right
    try:
        return a.is_finite() and b.is_finite() and abs(a - b) <= Decimal("0.00005")
    except DecimalException:
        return False


def vote_key(answer: str) -> str:
    """Group equivalent numeric spellings for majority voting without labels."""
    text = answer.strip().casefold()
    try:
        number = Decimal(text)
        return str(number.normalize()) if number.is_finite() else text
    except DecimalException:
        return text
