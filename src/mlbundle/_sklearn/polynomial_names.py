"""Bounded polynomial names without sklearn's dense powers_ allocation."""

from collections import Counter
from itertools import combinations, combinations_with_replacement
from typing import Any

from mlbundle.errors import InvalidModelError, LimitExceededError


def polynomial_names(model: Any, names: list[str]) -> list[str]:
    # Count term visits before constructing tuples; unlike powers_, work follows
    # emitted monomials, not output width multiplied by original feature width.
    maximum = (
        min(model._max_degree, len(names))
        if model.interaction_only
        else model._max_degree
    )
    if model.n_output_features_ * max(1, maximum) > 100_000:
        raise LimitExceededError("Transformer feature-name work exceeds the limit.")
    choose = combinations if model.interaction_only else combinations_with_replacement
    result = ["1"] if model.include_bias else []
    for degree in range(max(1, model._min_degree), maximum + 1):
        for term in choose(range(len(names)), degree):
            parts = [
                names[index] + ("" if power == 1 else f"^{power}")
                for index, power in sorted(Counter(term).items())
            ]
            if sum(map(len, parts)) + len(parts) - 1 > 1024:
                raise InvalidModelError("Invalid transformed feature names.")
            result.append(" ".join(parts))
    return result
