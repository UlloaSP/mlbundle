"""Supplemental skops trust owned by mlbundle, never by serialized declarations."""

OPTIMIZER_TYPES = (
    "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",
    "sklearn.neural_network._stochastic_optimizers.SGDOptimizer",
)
# Numeric Tree storage and MLP optimizer records; no functions/I/O constructors.
# Optimizers use ObjectNode (__new__ plus __dict__.update), never __init__.
TRUSTED_TYPES = ("sklearn.tree._tree.Tree", *OPTIMIZER_TYPES)
