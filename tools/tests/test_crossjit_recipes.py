import importlib.util
from pathlib import Path


def _recipes_module():
    path = (Path(__file__).parents[2] / "numpy/_core/code_generators" /
            "crossjit_recipes.py")
    spec = importlib.util.spec_from_file_location("numpy_crossjit_recipes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mvp_recipes_are_atomic_and_stable():
    module = _recipes_module()
    assert tuple(sorted(module.RECIPES)) == (
        "numpy.add",
        "numpy.multiply",
        "numpy.subtract",
        "numpy.true_divide",
    )
    assert len(module.RECIPE_SET_HASH) == 64
    for name, recipe in module.RECIPES.items():
        assert recipe.arity == 2
        assert recipe.operation_id == name
        assert recipe.serialize() == recipe.serialize()
        assert recipe.stable_hash() == recipe.stable_hash()


def test_no_fusion_combination_recipe_exists():
    module = _recipes_module()
    forbidden = ("add_mul", "add_multiply", "fusion", "linear_")
    assert not any(
        token in name for name in module.RECIPES for token in forbidden)
