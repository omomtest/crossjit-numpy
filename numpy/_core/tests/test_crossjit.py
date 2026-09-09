import json
import os
import subprocess
import sys
import tempfile

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


@pytest.mark.parametrize("scalar_type", [np.float32, np.float64])
def test_crossjit_numpy_float_scalar_array_ops(scalar_type):
    a = np.linspace(1.0, 4.0, 256, dtype=np.float64)
    scalar = scalar_type(2.5)

    for fn in [
        lambda x, s: x + s,
        lambda x, s: x - s,
        lambda x, s: x * s,
        lambda x, s: x / s,
        lambda x, s: s + x,
        lambda x, s: s - x,
        lambda x, s: s * x,
        lambda x, s: s / x,
    ]:
        np.testing.assert_allclose(_warm(fn, a, scalar), fn(a, scalar))


@pytest.mark.parametrize("scalar_type", [np.float32, np.float64])
def test_crossjit_fused_numpy_float_scalar_chain(scalar_type):
    a = np.linspace(1.0, 4.0, 256, dtype=np.float64)
    b = np.linspace(5.0, 9.0, 256, dtype=np.float64)
    scalar = scalar_type(2.5)

    for fn in [
        lambda x, y, s: s * x + y,
        lambda x, y, s: y + s * x,
        lambda x, y, s: x * s - y,
        lambda x, y, s: y - s * x,
        lambda x, y, s: (x * s) * np.float64(0.75),
        lambda x, y, s: np.float64(0.75) * (s * x),
    ]:
        np.testing.assert_allclose(_warm(fn, a, b, scalar), fn(a, b, scalar))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_crossjit_inplace_array_and_scalar_ops(dtype):
    left = np.linspace(1.0, 4.0, 256, dtype=dtype)
    right = np.linspace(5.0, 9.0, 256, dtype=dtype)

    def inplace_add(x, y):
        x += y

    def inplace_subtract(x, y):
        x -= y

    def inplace_multiply(x, y):
        x *= y

    def inplace_divide(x, y):
        x /= y

    def warm_inplace_array(op, initial, other, n=2000):
        out = initial.copy()
        for _ in range(n):
            op(out, other)
        return out

    def warm_inplace_scalar(op, initial, scalar, n=2000):
        out = initial.copy()
        for _ in range(n):
            op(out, scalar)
        return out

    def reference_inplace(ufunc, initial, other, n=2000):
        out = initial.copy()
        for _ in range(n):
            ufunc(out, other, out=out)
        return out

    for op, expected in [
        (inplace_add, reference_inplace(np.add, left, right)),
        (inplace_subtract, reference_inplace(np.subtract, left, right)),
    ]:
        np.testing.assert_allclose(
            warm_inplace_array(op, left, right), expected,
            rtol=2e-5, atol=2e-5)

    scalar = np.float64(1.0001)
    for op, expected in [
        (inplace_multiply, reference_inplace(np.multiply, left, scalar)),
        (inplace_divide, reference_inplace(np.divide, left, scalar)),
    ]:
        np.testing.assert_allclose(
            warm_inplace_scalar(op, left, scalar), expected,
            rtol=2e-5, atol=2e-5)


def test_crossjit_inplace_overlap_and_readonly_fallbacks():
    original = np.linspace(1.0, 4.0, 256)
    actual = original.copy()
    expected = original.copy()
    actual += actual[::-1]
    expected += expected[::-1]
    np.testing.assert_allclose(actual, expected)

    readonly = original.copy()
    readonly.flags.writeable = False
    with pytest.raises(ValueError):
        readonly += 1.0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_crossjit_fused_scaled_inplace_add_subtract(dtype):
    source = np.linspace(1.0, 4.0, 256, dtype=dtype)
    initial = np.linspace(5.0, 9.0, 256, dtype=dtype)
    scalar = np.float64(0.125)

    def inplace_add_scaled(dst, src, scale):
        dst += scale * src

    def inplace_subtract_scaled(dst, src, scale):
        dst -= scale * src

    for fn, ufunc in [
        (inplace_add_scaled, np.add),
        (inplace_subtract_scaled, np.subtract),
    ]:
        actual = initial.copy()
        expected = initial.copy()
        for _ in range(2000):
            fn(actual, source, scalar)
            ufunc(expected, scalar * source, out=expected)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_crossjit_strided_1d_array_scalar_ops(dtype):
    base = np.arange(1, 257, dtype=dtype).reshape(32, 8)
    column = base[:, 3]
    assert column.ndim == 1
    assert column.strides[0] > column.itemsize

    cases = [
        lambda x: x + 2.5,
        lambda x: x - 2.5,
        lambda x: x * 2.5,
        lambda x: x / 2.5,
        lambda x: 2.5 + x,
        lambda x: 2.5 - x,
        lambda x: 2.5 * x,
        lambda x: 2.5 / x,
    ]
    for fn in cases:
        np.testing.assert_allclose(_warm(fn, column), fn(column))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_crossjit_strided_1d_array_array_ops(dtype):
    left_base = np.arange(1, 257, dtype=dtype).reshape(32, 8)
    right_base = np.arange(257, 513, dtype=dtype).reshape(32, 8)
    left = left_base[:, 2]
    right = right_base[:, 5]

    for fn in [
        lambda x, y: x + y,
        lambda x, y: x - y,
        lambda x, y: x * y,
        lambda x, y: x / y,
    ]:
        np.testing.assert_allclose(_warm(fn, left, right), fn(left, right))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_crossjit_strided_inner_array_array_ops(dtype):
    left_base = np.arange(1, 34 * 10 + 1, dtype=dtype).reshape(34, 10)
    right_base = np.arange(341, 681, dtype=dtype).reshape(34, 10)
    left = left_base[1:-1, 1:-1]
    right = right_base[1:-1, 1:-1]
    assert not left.flags.c_contiguous
    assert left.strides[-1] == left.itemsize

    for fn in [
        lambda x, y: x + y,
        lambda x, y: x - y,
        lambda x, y: x * y,
        lambda x, y: x / y,
    ]:
        np.testing.assert_allclose(_warm(fn, left, right), fn(left, right))


def test_crossjit_fused_binary2_arithmetic_chains():
    left_base = np.linspace(1.0, 4.0, 34 * 10).reshape(34, 10)
    right_base = np.linspace(5.0, 9.0, 34 * 10).reshape(34, 10)
    views = [
        (left_base[1:-1, 1:-1], right_base[1:-1, 1:-1]),
        (left_base[1:-1, 2], right_base[1:-1, 6]),
    ]

    for left, right in views:
        for fn in [
            lambda x, y: x * 2.5 + y,
            lambda x, y: 2.5 * x + y,
            lambda x, y: x * 2.5 - y,
            lambda x, y: y - x * 2.5,
            lambda x, y: y - 2.5 * x,
            lambda x, y: (x + y) * 2.5,
            lambda x, y: 2.5 * (x + y),
            lambda x, y: (x - y) * 2.5,
            lambda x, y: 2.5 * (x - y),
            lambda x, y: (x + y) / 2.5,
            lambda x, y: (x - y) / 2.5,
        ]:
            np.testing.assert_allclose(_warm(fn, left, right), fn(left, right))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_crossjit_fused_add_three_arrays(dtype):
    a = np.linspace(1.0, 2.0, 256, dtype=dtype)
    b = np.linspace(3.0, 4.0, 256, dtype=dtype)
    c = np.linspace(5.0, 6.0, 256, dtype=dtype)

    def add3(x, y, z):
        return (x + y) + z

    np.testing.assert_allclose(_warm(add3, a, b, c), (a + b) + c)

    def add3_right(x, y, z):
        return z + (x + y)

    np.testing.assert_allclose(_warm(add3_right, a, b, c), c + (a + b))

    def inplace_add_pair(out, x, y):
        out += x + y
        return out

    actual = c.copy()
    expected = c.copy()
    for _ in range(2000):
        inplace_add_pair(actual, a, b)
        expected += a + b
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    # Aliasing the destination must safely fall back to NumPy.
    actual = c.copy()
    expected = c.copy()
    inplace_add_pair(actual, actual, b)
    expected += expected + b
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_crossjit_strided_inner_array_scalar_ops(dtype):
    base2 = np.arange(1, 34 * 10 + 1, dtype=dtype).reshape(34, 10)
    base3 = np.arange(1, 6 * 10 * 12 + 1, dtype=dtype).reshape(6, 10, 12)
    views = [base2[1:-1, 1:-1], base3[1:-1, 1:-1, 1:-1]]

    for view in views:
        assert not view.flags.c_contiguous
        assert view.strides[-1] == view.itemsize
        for fn in [
            lambda x: x + 2.5,
            lambda x: x - 2.5,
            lambda x: x * 2.5,
            lambda x: x / 2.5,
            lambda x: 2.5 + x,
            lambda x: 2.5 - x,
            lambda x: 2.5 * x,
            lambda x: 2.5 / x,
        ]:
            np.testing.assert_allclose(_warm(fn, view), fn(view))


def test_crossjit_strided_1d_array_scalar_rejected_layouts():
    base = np.arange(1, 257, dtype=np.float64).reshape(32, 8)
    rejected = [
        base[:, 3][::-1],
        np.broadcast_to(np.array([3.0]), (32,)),
        base[:, ::2],
    ]
    for view in rejected:
        np.testing.assert_allclose(_warm(lambda x: x * 2.5, view), view * 2.5)


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
        [sys.executable, "-P", "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=60,
    )
    if result.returncode == 77:
        pytest.skip("NumPy CrossJIT hooks are disabled or unavailable")
    assert result.returncode == 0, result.stdout + result.stderr


def test_crossjit_binary_matmul_region_epilogue():
    code = r'''
import _opcode
import _testinternalcapi
import gc
import numpy as np
import sys
import warnings
from test.support import reset_code

def opnames(func):
    for i in range(0, len(func.__code__.co_code), 2):
        try:
            return [item[0] for item in _opcode.get_executor(func.__code__, i)]
        except ValueError:
            pass
    return []

def matmul_add(n, a, b, other):
    out = None
    for _ in range(n):
        out = a @ b + other
    return out

def other_sub_matmul(n, a, b, other):
    out = None
    for _ in range(n):
        out = other - a @ b
    return out

def scalar_div_matmul(n, a, b, scalar):
    out = None
    for _ in range(n):
        out = scalar / (a @ b)
    return out

def standalone_matmul(n, a, b):
    out = None
    for _ in range(n):
        out = a @ b
    return out

warm = _testinternalcapi.TIER2_THRESHOLD + 80
for dtype in (np.float32, np.float64):
    a = np.linspace(0.5, 1.5, 24 * 32, dtype=dtype).reshape(24, 32)
    b = np.linspace(1.0, 2.0, 32 * 20, dtype=dtype).reshape(32, 20)
    full = np.linspace(2.0, 3.0, 24 * 20, dtype=dtype).reshape(24, 20)
    bias = np.linspace(0.1, 0.5, 20, dtype=dtype)
    cases = (
        (matmul_add, full, a @ b + full),
        (matmul_add, bias, a @ b + bias),
        (other_sub_matmul, full, full - a @ b),
        (scalar_div_matmul, dtype(1000), dtype(1000) / (a @ b)),
    )
    for func, other, expected in cases:
        reset_code(func)
        refs = tuple(sys.getrefcount(value) for value in (a, b, other))
        actual = func(warm, a, b, other)
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
        names = opnames(func)
        assert names.count('_EXT_REGION_EXEC') == 1, (func.__name__, names)
        assert '_EXT_BINARY_EXEC' not in names, (func.__name__, names)
        del actual
        gc.collect()
        assert refs == tuple(sys.getrefcount(value) for value in (a, b, other))

    # An anchor that has no elementwise consumer is restored to the ordinary
    # binary uop, so standalone matmul pays no external dispatcher overhead.
    reset_code(standalone_matmul)
    actual = standalone_matmul(warm, a, b)
    np.testing.assert_allclose(actual, a @ b, rtol=3e-5, atol=3e-5)
    names = opnames(standalone_matmul)
    assert '_BINARY_OP' in names, names
    assert '_EXT_BINARY_EXEC' not in names, names

# A property miss stays local to the Region.  A later supported call must
# still hit the same executor after a Fortran-layout negative PIC entry.
a_f = np.asfortranarray(a)
b_f = np.asfortranarray(b)
np.testing.assert_allclose(
    matmul_add(4, a_f, b_f, full), a_f @ b_f + full,
    rtol=3e-5, atol=3e-5)
np.testing.assert_allclose(
    matmul_add(4, a, b, full), a @ b + full,
    rtol=3e-5, atol=3e-5)
assert opnames(matmul_add).count('_EXT_REGION_EXEC') == 1

# Matmul performs its own NumPy FP handling.  The Region must neither lose
# nor duplicate a warning, and raise mode must take the exact local recipe.
huge = np.full((16, 16), 1e308)
zeros = np.zeros((16, 16))
reset_code(matmul_add)
matmul_add(warm, np.eye(16), np.eye(16), zeros)
with np.errstate(over='warn', invalid='warn'):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        warned = matmul_add(1, huge, huge, zeros)
assert np.isinf(warned).all()
assert len([item for item in seen if 'overflow' in str(item.message)]) == 1, seen
with np.errstate(over='raise', invalid='raise'):
    try:
        matmul_add(1, huge, huge, zeros)
    except FloatingPointError:
        pass
    else:
        raise AssertionError('matmul Region lost NumPy seterr=raise')
'''
    env = os.environ.copy()
    env.update({
        "PYTHON_JIT": "1",
        "PYTHON_JIT_EXT_ENABLE_REGIONS": "1",
        "PYTHON_JIT_EXT_ENABLE_FUSED_BINARY2": "0",
        "NUMPY_JIT_ENABLE_GENERIC_REGION_PROGRAM": "1",
    })
    result = subprocess.run(
        [sys.executable, "-P", "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_crossjit_region_exec_without_legacy_binary2():
    code = r'''
import _opcode
import _testinternalcapi
import gc
import numpy as np
import sys
import warnings
from test.support import reset_code

def scaled_add(n, a, b, scale):
    out = None
    for _ in range(n):
        out = a * scale + b
    return out

def arithmetic_chain(n, a, b, c, scale, tail):
    out = None
    for _ in range(n):
        out = ((a / b) / c) / scale / tail
    return out

def admission_chain(n, a, b, scale):
    out = None
    for _ in range(n):
        out = a * scale + b
    return out

def balanced_dag(n, a, b, c, d):
    out = None
    for _ in range(n):
        out = (a * b) + (c * d)
    return out

def square_chain(n, a, exponent):
    out = None
    for _ in range(n):
        out = (a - 1.0) ** exponent + 3.0
    return out

def opnames(func):
    for i in range(0, len(func.__code__.co_code), 2):
        try:
            return [item[0] for item in _opcode.get_executor(func.__code__, i)]
        except ValueError:
            pass
    return []

a = np.linspace(1.0, 2.0, 1024)
b = np.linspace(3.0, 4.0, 1024)
scale = np.float64(0.5)
reset_code(scaled_add)
actual = scaled_add(_testinternalcapi.TIER2_THRESHOLD + 50, a, b, scale)
np.testing.assert_allclose(actual, a * scale + b)
assert '_EXT_REGION_EXEC' in opnames(scaled_add), opnames(scaled_add)

# A four-node chain is lowered from the provider DAG; it has no corresponding
# CPython plan kind or NumPy fused-pattern enum.
c = np.linspace(0.1, 0.2, 1024)
tail = np.float64(2.0)
refs_before = [sys.getrefcount(value) for value in (a, b, c)]
reset_code(arithmetic_chain)
chain = arithmetic_chain(
    _testinternalcapi.TIER2_THRESHOLD + 50, a, b, c, scale, tail)
np.testing.assert_allclose(chain, ((a / b) / c) / scale / tail)
chain_ops = opnames(arithmetic_chain)
assert chain_ops.count('_EXT_REGION_EXEC') == 1, chain_ops
assert '_EXT_BINARY_EXEC' not in chain_ops, chain_ops
del chain
gc.collect()
assert refs_before == [sys.getrefcount(value) for value in (a, b, c)]

# The host builder recovers a non-linear, single-output DAG without knowing
# that the provider will lower multiply/add.
d = np.linspace(7.0, 8.0, 1024)
reset_code(balanced_dag)
dag = balanced_dag(
    _testinternalcapi.TIER2_THRESHOLD + 50, a, b, c, d)
np.testing.assert_allclose(dag, (a * b) + (c * d))
dag_ops = opnames(balanced_dag)
assert dag_ops.count('_EXT_REGION_EXEC') == 1, dag_ops
assert '_EXT_BINARY_EXEC' not in dag_ops, dag_ops

# POWER is admitted atomically for exact fallback, while the provider only
# specializes exponent=2 as a SQUARE node.  A negative exponent signature
# must not poison the later positive specialization at the same site.
reset_code(square_chain)
cube = square_chain(
    _testinternalcapi.TIER2_THRESHOLD + 50, a + 1.0, 3)
np.testing.assert_allclose(cube, a ** 3 + 3.0)
square = square_chain(8, a + 1.0, 2)
np.testing.assert_allclose(square, a ** 2 + 3.0)
square_ops = opnames(square_chain)
assert square_ops.count('_EXT_REGION_EXEC') == 1, square_ops
assert '_EXT_BINARY_EXEC' not in square_ops, square_ops

# A property-independent positive lowering handles compatible dtype/layout
# changes.  An unsupported shape still uses a negative PIC entry and must not
# invalidate the executor.
float32_inputs = tuple(
    np.linspace(i, i + 1, 1024, dtype=np.float32) for i in range(4))
strided_base = np.linspace(1.0, 9.0, 4 * 4096)
strided_inputs = tuple(
    strided_base[i * 4096:(i + 1) * 4096:2] for i in range(4))
inner_base = np.linspace(1.0, 9.0, 4 * 34 * 10).reshape(4, 34, 10)
inner_inputs = tuple(inner_base[i, 1:-1, 1:-1] for i in range(4))
for values in (float32_inputs, strided_inputs):
    actual = balanced_dag(8, *values)
    np.testing.assert_allclose(
        actual, (values[0] * values[1]) + (values[2] * values[3]),
        rtol=1e-5)
small_inputs = tuple(value.ravel()[:8] for value in float32_inputs)
np.testing.assert_allclose(
    balanced_dag(4, *small_inputs),
    (small_inputs[0] * small_inputs[1]) +
    (small_inputs[2] * small_inputs[3]), rtol=1e-5)
np.testing.assert_allclose(
    balanced_dag(8, *inner_inputs),
    (inner_inputs[0] * inner_inputs[1]) +
    (inner_inputs[2] * inner_inputs[3]), rtol=1e-5)
assert '_EXT_REGION_EXEC' in opnames(balanced_dag), opnames(balanced_dag)

# warn is speculative: discard the fused result and replay the two ufuncs.
huge = np.full(1024, 1e308)
zeros = np.zeros(1024)
with np.errstate(over='warn'):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        warned = scaled_add(1, huge, zeros, np.float64(1e308))
assert np.isinf(warned).all()
assert len([w for w in seen if 'overflow' in str(w.message)]) == 1, seen

# raise/call/log/print policies reject the fast path before it can write.
reset_code(scaled_add)
scaled_add(_testinternalcapi.TIER2_THRESHOLD + 50, a, b, scale)
with np.errstate(over='raise'):
    try:
        scaled_add(1, huge, zeros, np.float64(1e308))
    except FloatingPointError:
        pass
    else:
        raise AssertionError('region fallback lost NumPy seterr=raise')

# The same plan must preserve semantics when runtime profitability rejects it.
# Eight elements are below both the kernel and region profitability floors.
small_a = a[:8].copy()
small_b = b[:8].copy()
np.testing.assert_allclose(
    scaled_add(1, small_a, small_b, scale), small_a * scale + small_b)

# A medium dedicated two-op region is profitable even though the generic
# expression-program threshold is deliberately higher.
medium_a = a[:98].copy()
medium_b = b[:98].copy()
reset_code(admission_chain)
for _ in range(6):
    medium = admission_chain(
        _testinternalcapi.TIER2_THRESHOLD + 100,
        medium_a, medium_b, scale)
np.testing.assert_allclose(medium, medium_a * scale + medium_b)
admission_ops = opnames(admission_chain)
assert admission_ops.count('_EXT_REGION_EXEC') == 1, admission_ops
assert '_EXT_FUSED_BINARY2_EXEC' not in admission_ops, admission_ops
assert '_EXT_BINARY_EXEC' not in admission_ops, admission_ops
'''
    stats_tmp = tempfile.TemporaryDirectory()
    env = os.environ.copy()
    env.update({
        "PYTHON_JIT": "1",
        "PYTHON_JIT_EXT_STATS": "1",
        "PYTHON_JIT_EXT_STATS_DIR": stats_tmp.name,
        "PYTHON_JIT_EXT_ENABLE_REGIONS": "1",
        "PYTHON_JIT_EXT_ENABLE_FUSED_BINARY2": "0",
        "NUMPY_JIT_ENABLE_GENERIC_REGION_PROGRAM": "1",
        "NUMPY_JIT_LLVM": "off",
    })
    result = subprocess.run(
        [sys.executable, "-P", "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=60,
    )
    try:
        assert result.returncode == 0, result.stdout + result.stderr
        site_path = os.path.join(stats_tmp.name, "crossjit_ext_sites.ndjson")
        with open(site_path, encoding="utf-8") as stream:
            sites = [json.loads(line) for line in stream]
        dag_sites = [
            site for site in sites
            if site["code"] == "balanced_dag" and site["region_site_count"]
        ]
        assert len(dag_sites) == 1, sites
        dag_site = dag_sites[0]
        assert dag_site["region_node_count"] == 3, dag_site
        assert dag_site["region_pic_hit_count"] > 0, dag_site
        # The property-independent RegionProgram accepts compatible positive
        # striding and lets its specialized guard route unsupported dynamic
        # properties.  These calls must therefore reuse the positive entry,
        # rather than manufacturing a redundant negative PIC entry.
        assert dag_site["region_specialize_count"] >= 1, dag_site
        assert dag_site["region_negative_count"] == 0, dag_site
        assert dag_site["region_saturated_count"] == 0, dag_site
        assert dag_site["region_disabled"] == 0, dag_site
    finally:
        stats_tmp.cleanup()


def test_crossjit_llvm_region_backend_semantics():
    """Exercise native Region code and every FP-policy fallback in a child."""
    code = r'''
import _testinternalcapi
import json
import os
import time
import warnings

import numpy as np
from test.support import reset_code

def linear(n, a, b, c):
    out = None
    for _ in range(n):
        out = a * b + c
    return out

def balanced(n, a, b, c, d):
    out = None
    for _ in range(n):
        out = (a + b) * (c - d)
    return out

def six_nodes(n, a, b, c, d, e, f):
    out = None
    for _ in range(n):
        out = ((a + b) * (c - d) + e) * f - a
    return out

def repeated(n, a):
    out = None
    for _ in range(n):
        out = (a + a) * a + a
    return out

def scalar_first(n, scalar, a, b):
    out = None
    for _ in range(n):
        out = scalar * a + b
    return out

def divide_add(n, a, b, c):
    out = None
    for _ in range(n):
        out = a / b + c
    return out

def square_neg_add(n, a, b):
    out = None
    for _ in range(n):
        out = -(a ** 2) + b
    return out

def fork_new_graph(n, a, b, c, d):
    out = None
    for _ in range(n):
        out = (a - b) * (c + d)
    return out

warm = _testinternalcapi.TIER2_THRESHOLD + 1800
sizes = (0, 1, 8, 31, 32, 98, 256, 1024, 65536)
for dtype in (np.float32, np.float64, np.int32, np.int64):
    a = np.arange(1, 1025, dtype=dtype)
    b = np.arange(3, 1027, dtype=dtype)
    c = np.arange(5, 1029, dtype=dtype)
    reset_code(linear)
    np.testing.assert_array_equal(linear(warm, a, b, c), a * b + c)
    time.sleep(0.02)
    for size in sizes:
        aa = np.arange(1, size + 1, dtype=dtype)
        bb = np.arange(3, size + 3, dtype=dtype)
        cc = np.arange(5, size + 5, dtype=dtype)
        np.testing.assert_array_equal(linear(20, aa, bb, cc), aa * bb + cc)

# Explicit wraparound must not acquire LLVM nsw/nuw assumptions.
for dtype in (np.int32, np.int64):
    info = np.iinfo(dtype)
    a = np.array([info.max, info.min, info.max], dtype=dtype)
    b = np.array([2, -1, info.max], dtype=dtype)
    c = np.array([1, info.max, info.min], dtype=dtype)
    with np.errstate(over='ignore'):
        expected = a * b + c
        actual = linear(50, a, b, c)
    np.testing.assert_array_equal(actual, expected)

# Integer native kernels admit integer scalars only.  A floating scalar at
# the same site must retain NumPy's promotion rules through the old path.
for dtype in (np.int32, np.int64):
    ia = np.arange(1, 1025, dtype=dtype)
    ib = np.arange(3, 1027, dtype=dtype)
    reset_code(scalar_first)
    np.testing.assert_array_equal(
        scalar_first(warm, 3, ia, ib), 3 * ia + ib)
    time.sleep(0.02)
    promoted = scalar_first(20, 0.25, ia, ib)
    expected_promoted = 0.25 * ia + ib
    assert promoted.dtype == expected_promoted.dtype
    np.testing.assert_array_equal(promoted, expected_promoted)

a, b, c, d, e, f = [
    np.linspace(i + 0.25, i + 1.25, 1024)
    for i in range(6)
]
for fn, fn_args, expected in (
    (balanced, (a, b, c, d), (a + b) * (c - d)),
    (six_nodes, (a, b, c, d, e, f), ((a + b) * (c - d) + e) * f - a),
    (repeated, (a,), (a + a) * a + a),
    (scalar_first, (np.float64(0.25), a, b), np.float64(0.25) * a + b),
    (square_neg_add, (a, b), -(a ** 2) + b),
):
    reset_code(fn)
    np.testing.assert_array_equal(fn(warm, *fn_args), expected)
    time.sleep(0.02)
    np.testing.assert_array_equal(fn(100, *fn_args), expected)

special_a = np.array([0.0, -0.0, np.nan, np.inf, -np.inf] * 64)
special_b = np.array([-0.0, 0.0, 2.0, 3.0, 4.0] * 64)
special_c = np.ones_like(special_a)
with np.errstate(all='ignore'):
    actual = linear(50, special_a, special_b, special_c)
    expected = special_a * special_b + special_c
np.testing.assert_array_equal(actual, expected)
np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))

reset_code(divide_add)
ones = np.ones(1024)
zeros = np.zeros(1024)
divide_add(warm, ones, ones, zeros)
time.sleep(0.02)
with np.errstate(divide='ignore'):
    np.testing.assert_array_equal(
        divide_add(20, ones, zeros, zeros),
        np.full(1024, np.inf))
with np.errstate(divide='warn'):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        divide_add(1, ones, zeros, zeros)
assert len([w for w in seen if 'divide' in str(w.message)]) == 1, seen
with np.errstate(divide='raise'):
    try:
        divide_add(1, ones, zeros, zeros)
    except FloatingPointError:
        pass
    else:
        raise AssertionError('LLVM Region bypassed seterr=raise')

called = []
def handler(message, flag):
    called.append((message, flag))
old_handler = np.seterrcall(handler)
try:
    with np.errstate(divide='call'):
        divide_add(1, ones, zeros, zeros)
finally:
    np.seterrcall(old_handler)
assert len(called) == 1, called

class LogSink:
    def __init__(self):
        self.messages = []
    def write(self, message):
        self.messages.append(message)

sink = LogSink()
old_handler = np.seterrcall(sink)
try:
    with np.errstate(divide='log'):
        np.testing.assert_array_equal(
            divide_add(1, ones, zeros, zeros), np.full(1024, np.inf))
finally:
    np.seterrcall(old_handler)
assert len(sink.messages) == 1, sink.messages
with np.errstate(divide='print'):
    np.testing.assert_array_equal(
        divide_add(1, ones, zeros, zeros), np.full(1024, np.inf))

# A forked child may call inherited ready code, while a specialization that
# did not exist in the parent must lazily create a child-owned worker/LLJIT.
if hasattr(os, 'fork'):
    pid = os.fork()
    if pid == 0:
        try:
            np.testing.assert_array_equal(linear(20, a, b, c), a * b + c)
            reset_code(fork_new_graph)
            expected = (a - b) * (c + d)
            np.testing.assert_array_equal(
                fork_new_graph(warm, a, b, c, d), expected)
            time.sleep(0.05)
            np.testing.assert_array_equal(
                fork_new_graph(100, a, b, c, d), expected)
        except BaseException:
            import traceback
            traceback.print_exc()
            os._exit(2)
        os._exit(0)
    waited, status = os.waitpid(pid, 0)
    assert waited == pid and os.waitstatus_to_exitcode(status) == 0, status
'''
    stats_tmp = tempfile.TemporaryDirectory()
    env = os.environ.copy()
    env.update({
        "PYTHON_JIT": "1",
        "PYTHON_JIT_EXT_STATS": "1",
        "PYTHON_JIT_EXT_STATS_DIR": stats_tmp.name,
        "PYTHON_JIT_EXT_ENABLE_REGIONS": "1",
        "PYTHON_JIT_EXT_ENABLE_FUSED_BINARY2": "0",
        "NUMPY_JIT_ENABLE_GENERIC_REGION_PROGRAM": "1",
        "NUMPY_JIT_LLVM": "force",
    })
    result = subprocess.run(
        [sys.executable, "-P", "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=120,
    )
    try:
        assert result.returncode == 0, result.stdout + result.stderr
        llvm_path = os.path.join(
            stats_tmp.name, "crossjit_numpy_llvm.ndjson")
        with open(llvm_path, encoding="utf-8") as stream:
            entries = [json.loads(line) for line in stream]
        if not entries:
            pytest.skip("NumPy was built without crossjit-llvm")
        ready = [entry for entry in entries if entry["entry_state"] == 1]
        assert ready, entries
        assert any(entry["native_uses"] > 0 for entry in ready), ready
        assert all(entry["vector_bits"] >= 128 for entry in ready), ready
        assert all(entry["object_bytes"] > 0 for entry in ready), ready
    finally:
        stats_tmp.cleanup()


def test_crossjit_aot_stencil_first_guard():
    """A reviewed hot DAG reaches precompiled code with LLVM disabled."""
    code = r'''
import _testinternalcapi
import time
import numpy as np
from test.support import reset_code

def square_sub(n, a, b):
    out = None
    for _ in range(n):
        out = a - b ** 2
    return out

a = np.linspace(-3.0, 4.0, 1024)
b = np.linspace(-2.0, 3.0, 1024)
reset_code(square_sub)
warm = _testinternalcapi.TIER2_THRESHOLD + 1800
with np.errstate(all='ignore'):
    np.testing.assert_array_equal(square_sub(warm, a, b), a - b ** 2)
    np.testing.assert_array_equal(square_sub(100, a[:98], b[:98]),
                                  a[:98] - b[:98] ** 2)
time.sleep(0.01)
'''
    stats_tmp = tempfile.TemporaryDirectory()
    env = os.environ.copy()
    env.update({
        "PYTHON_JIT": "1",
        "PYTHON_JIT_EXT_STATS": "1",
        "PYTHON_JIT_EXT_STATS_DIR": stats_tmp.name,
        "PYTHON_JIT_EXT_ENABLE_REGIONS": "1",
        "PYTHON_JIT_EXT_ENABLE_FUSED_BINARY2": "0",
        "NUMPY_JIT_ENABLE_GENERIC_REGION_PROGRAM": "1",
        "NUMPY_JIT_AOT": "force",
        "NUMPY_JIT_LLVM": "off",
    })
    result = subprocess.run(
        [sys.executable, "-P", "-c", code], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=env, timeout=120)
    try:
        assert result.returncode == 0, result.stdout + result.stderr
        aot_path = os.path.join(stats_tmp.name, "crossjit_numpy_aot.ndjson")
        with open(aot_path, encoding="utf-8") as stream:
            entries = [json.loads(line) for line in stream]
        if not entries:
            pytest.skip("NumPy was built without crossjit-aot-stencils")
        matched = [entry for entry in entries if entry["lookup_hit"]]
        assert matched, entries
        assert any(sum(entry["native_uses"]) > 0 for entry in matched), matched
        assert all(entry["vector_bits"] >= 128 for entry in matched), matched
    finally:
        stats_tmp.cleanup()
