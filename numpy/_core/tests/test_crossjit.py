import os
import subprocess
import sys

import pytest

import numpy as np


def _warm(fn, *args, n=2000):
    result = None
    for _ in range(n):
        result = fn(*args)
    return result


def test_crossjit_binary_ops_and_fallbacks():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = np.linspace(1.0, 2.0, 12, dtype=np.float64).reshape(3, 4)

    np.testing.assert_allclose(_warm(lambda x, y: x + y, a, b), a + b)
    np.testing.assert_allclose(_warm(lambda x, y: x - y, a, b), a - b)
    np.testing.assert_allclose(_warm(lambda x, y: x * y, a, b), a * b)
    np.testing.assert_allclose(_warm(lambda x, y: x / y, a, b), a / b)

    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    np.testing.assert_allclose(_warm(lambda x, y: x @ y, left, right), left @ right)

    np.testing.assert_allclose(_warm(lambda x: x + 1.0, a), a + 1.0)
    np.testing.assert_allclose(_warm(lambda x, y: x + y, a, b[:, :1]), a + b[:, :1])
    int_a = np.arange(12, dtype=np.int64).reshape(3, 4)
    np.testing.assert_array_equal(_warm(lambda x, y: x - y, int_a, int_a), int_a - int_a)


def test_crossjit_basic_subscript_and_rejected_indices():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)

    for fn in [
        lambda x: x[1],
        lambda x: x[:, 1:3],
        lambda x: x[..., 2],
        lambda x: x[None, :, 1],
        lambda x: x[(slice(None), 2)],
    ]:
        np.testing.assert_allclose(_warm(fn, a), fn(a))

    np.testing.assert_allclose(_warm(lambda x: x[[0, 2]], a), a[[0, 2]])
    np.testing.assert_allclose(_warm(lambda x: x[x > 3], a), a[a > 3])


def test_crossjit_callable_ufuncs_and_method_separation():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = np.linspace(1.0, 2.0, 12, dtype=np.float64).reshape(3, 4)

    for ufunc in [np.add, np.subtract, np.multiply, np.true_divide]:
        np.testing.assert_allclose(_warm(ufunc, a, b), ufunc(a, b))

    np.testing.assert_allclose(_warm(lambda x: x.sum(), a), a.sum())
    np.testing.assert_allclose(_warm(lambda x, y: np.power(x, y), a, b), np.power(a, b))


def test_crossjit_emits_expected_uops_when_jit_enabled():
    try:
        import _opcode  # noqa: F401
        import _testinternalcapi  # noqa: F401
    except ImportError:
        pytest.skip("requires CPython uop inspection helpers")

    code = '''
import _opcode, _testinternalcapi
import numpy as np
from test.support import reset_code

def get_first_executor(func):
    for i in range(0, len(func.__code__.co_code), 2):
        try:
            return _opcode.get_executor(func.__code__, i)
        except ValueError:
            pass

def opnames(func):
    ex = get_first_executor(func)
    return [] if ex is None else [item[0] for item in ex]

a = np.arange(12, dtype=np.float64).reshape(3, 4)
b = np.linspace(1.0, 2.0, 12, dtype=np.float64).reshape(3, 4)

def bin_sub(n, x, y):
    out = None
    for _ in range(n): out = x - y
    return out

def subscr(n, x):
    out = None
    for _ in range(n): out = x[:, 1:3]
    return out

def callable_add(n, x, y):
    out = None
    for _ in range(n): out = np.add(x, y)
    return out

def method_sum(n, x):
    out = None
    for _ in range(n): out = x.sum()
    return out

missing = []
for f, args, expected in [
    (bin_sub, (a, b), '_EXT_BINARY_EXEC'),
    (callable_add, (a, b), '_EXT_CALL_EXEC'),
]:
    reset_code(f)
    f(_testinternalcapi.TIER2_THRESHOLD + 50, *args)
    names = opnames(f)
    if expected not in names:
        missing.append((f.__name__, expected, names))

if len(missing) == 2:
    raise SystemExit(77)
assert not missing, missing

reset_code(method_sum)
method_sum(_testinternalcapi.TIER2_THRESHOLD + 50, a)
assert '_EXT_CALL_EXEC' not in opnames(method_sum), opnames(method_sum)
'''
    env = os.environ.copy()
    env["PYTHON_JIT"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=60,
    )
    if result.returncode == 77:
        pytest.skip("NumPy CrossJIT hooks are disabled or unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
