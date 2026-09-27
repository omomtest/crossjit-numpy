"""NumPy-owned atomic recipes for the CrossJIT Generator MVP."""

import hashlib

from crossjit_generator.recipe import (
    AliasKind,
    Constraint,
    EffectKind,
    EffectSchema,
    ErrorSchema,
    FloatingPointErrorKind,
    IterationKind,
    MutationKind,
    Recipe,
    SemanticExpr,
)


_PURE_FRESH = EffectSchema(
    effect=EffectKind.PURE,
    mutation=MutationKind.NONE,
    result_alias=AliasKind.FRESH,
    may_call_host=False,
)
_NUMPY_FPE = ErrorSchema(
    may_report=True,
    floating_point=FloatingPointErrorKind.PROVIDER_DEFINED,
    preserve_operation_order=True,
)
_MVP_CONSTRAINTS = (
    Constraint("preserve_operation_order", True),
    Constraint("requires_effect_free_contract", True),
)


def _binary(name: str) -> Recipe:
    return Recipe(
        operation_id=f"numpy.{name}",
        arity=2,
        semantic_expression=SemanticExpr.operation(
            name, SemanticExpr.argument(0), SemanticExpr.argument(1)),
        iteration_kind=IterationKind.ELEMENTWISE,
        effects=_PURE_FRESH,
        errors=_NUMPY_FPE,
        constraints=_MVP_CONSTRAINTS,
    )


RECIPES = {
    recipe.operation_id: recipe
    for recipe in (
        _binary("add"),
        _binary("subtract"),
        _binary("multiply"),
        _binary("true_divide"),
    )
}

RECIPE_SET_HASH = hashlib.sha256(
    "\n".join(RECIPES[name].stable_hash() for name in sorted(RECIPES)).encode()
).hexdigest()

__all__ = ["RECIPES", "RECIPE_SET_HASH"]
