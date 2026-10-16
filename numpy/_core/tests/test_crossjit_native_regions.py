"""Exercise private native plans; calibration forcing is only a test fixture."""
import gc
import weakref

import numpy as np
import pytest
from numpy._core import _multiarray_umath as provider


@pytest.fixture(autouse=True)
def native_fixture(monkeypatch):
    if not hasattr(provider, '_crossjit_stats'):
        pytest.skip('CrossJIT build required')
    monkeypatch.setenv('NPY_CROSSJIT_NATIVE_CALIBRATION', '1')
    monkeypatch.setenv('NPY_CROSSJIT_DIAGNOSTICS', '1')
    provider._crossjit_reset_stats()


def matrix_loop(a, b, scale):
    for _ in range(300):
        result = (a @ b) * scale
    return result


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape', [((17,), (17,)), ((13, 17), (17,)),
    ((17,), (17, 11)), ((13, 17), (17, 11)),
    ((2, 1, 13, 17), (3, 17, 11))])
def test_matrix_domains(dtype, shape):
    rng = np.random.default_rng(714)
    a, b = [rng.normal(size=s).astype(dtype) for s in shape]
    # Reverse contraction axes together, testing packing without reassociation.
    a = a[..., ::-1]
    b = b[::-1] if b.ndim == 1 else b[..., ::-1, :]
    expected = (a @ b) * 0.75
    actual = matrix_loop(a, b, 0.75)
    assert type(actual) is type(expected)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected, strict=True)
    stats = provider._crossjit_stats()
    assert stats['native_exec_hit'] + stats['mixed_exec_hit'] > 0
    assert stats['exec_miss'] == stats['exec_error'] == 0


def reduction_loop(a, axis):
    for _ in range(300):
        result = np.add.reduce(a, axis=axis, keepdims=True)
    return result


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('axis', [None, 0, -1, (0, 2), ()])
def test_reduction_axis_keepdims(dtype, axis):
    a = np.random.default_rng(91).normal(size=(3, 7, 19)).astype(dtype)
    a = a[:, ::-1, :]
    expected = np.add.reduce(a, axis=axis, keepdims=True)
    actual = reduction_loop(a, axis)
    np.testing.assert_array_equal(actual, expected, strict=True)
    stats = provider._crossjit_stats()
    assert stats['native_exec_hit'] > 0
    assert stats['exec_miss'] == stats['exec_error'] == 0


def scalar_reduce(a):
    for _ in range(300):
        result = np.add.reduce(a)
    return result


def outer_loop(a, b):
    for _ in range(300):
        result = np.add.outer(a, b)
    return result


def test_scalar_publication_and_outer():
    a = np.arange(21, dtype=np.float32)[::-1]
    expected = np.add.reduce(a)
    actual = scalar_reduce(a)
    assert type(actual) is np.float32
    assert actual.tobytes() == expected.tobytes()
    b = np.arange(3, dtype=np.float32)
    np.testing.assert_array_equal(outer_loop(a, b), np.add.outer(a, b), strict=True)
    assert provider._crossjit_stats()['native_exec_hit'] > 0


def broadcast_loop(a, b, c):
    for _ in range(300):
        result = (a + b) * c
    return result


@pytest.mark.parametrize('scalar', [2.25, 2, np.float32(2.25)])
def test_fp32_broadcast_rounding(scalar):
    rng = np.random.default_rng(411)
    a = rng.normal(size=(3, 7, 19)).astype(np.float32)[:, ::-1, ::-1]
    b = rng.normal(size=(1, 7, 1)).astype(np.float32)
    expected = (a + b) * scalar
    actual = broadcast_loop(a, b, scalar)
    assert actual.tobytes() == expected.tobytes()
    assert provider._crossjit_stats()['native_exec_hit'] > 0


def test_changes_empty_and_unsupported_fallback():
    for dtype, shape in [(np.float64, (5, 7)), (np.float32, (3, 7)),
                         (np.complex128, (3, 7)), (np.float64, (0, 7))]:
        a = np.ones(shape, dtype=dtype)
        b = np.ones((7, 2), dtype=dtype)
        np.testing.assert_array_equal(matrix_loop(a, b, 0.5), (a @ b) * 0.5, strict=True)
    assert provider._crossjit_stats()['exec_error'] == 0


def callable_loop(call, a, b):
    for _ in range(300):
        result = call(a, b)
    return result


def test_callable_guard_and_release():
    a, b = np.arange(19.), -np.arange(19.)
    np.testing.assert_array_equal(callable_loop(np.maximum, a, b), np.maximum(a, b))
    np.testing.assert_array_equal(callable_loop(np.minimum, a, b), np.minimum(a, b))
    ref = weakref.ref(a)
    del a
    gc.collect()
    assert ref() is None
    assert provider._crossjit_stats()['exec_error'] == 0


def test_fpe_replay_and_special_values():
    a = np.array([0., -0., 1., -np.inf, np.nan])
    b = np.array([1., 1., 0., 0., 1.])
    def divide(x, y):
        for _ in range(300):
            result = (x / y) + 1.
        return result
    with np.errstate(all='ignore'):
        actual = divide(a, b)
        expected = (a / b) + 1.
    np.testing.assert_array_equal(actual, expected)
    assert np.signbit(actual[0]) == np.signbit(expected[0])
    with np.errstate(all='raise'):
        with pytest.raises(FloatingPointError):
            divide(a, b)
    assert provider._crossjit_stats()['exec_error'] == 0


def test_diagnostic_snapshots_are_immutable():
    a, b = np.ones(11), np.ones(13)
    outer_loop(a, b)
    before = provider._crossjit_stats()['native_candidates']
    import copy
    expected = copy.deepcopy(before)
    outer_loop(a, b)
    assert before == expected


def calls_loop(a, b):
    for _ in range(300):
        result = np.minimum(a + b, b)
    return result


def test_interior_call_has_complete_region():
    a, b = np.arange(103.), -np.arange(103.)
    np.testing.assert_array_equal(calls_loop(a, b), np.minimum(a + b, b))
    keys = provider._crossjit_stats()['native_candidates']
    assert any('|2|2|3|' in key and row['execute'] for key, row in keys.items())


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('operation', ['sqrt', 'exp', 'tanh'])
def test_original_unary_loops(dtype, operation):
    scope = {'np': np}
    exec(f'def f(a):\n    for i in range(300):\n        r=np.{operation}(a)\n    return r', scope)
    a = np.linspace(0.25, 2., 1031, dtype=dtype)[::-1]
    actual = scope['f'](a)
    expected = getattr(np, operation)(a)
    assert actual.tobytes() == expected.tobytes()
    assert provider._crossjit_stats()['native_exec_hit'] > 0


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('operation', ['sqrt', 'exp', 'tanh'])
def test_private_unary_buffer_reuse(dtype, operation):
    scope = {'np': np}
    exec(f'def f(a, b):\n    for i in range(300):\n        r=np.{operation}(a - b)\n    return r', scope)
    a = np.linspace(2., 3., 1031, dtype=dtype)
    b = np.linspace(.25, .5, 1031, dtype=dtype)
    before = a.copy(), b.copy()
    actual = scope['f'](a, b)
    expected = getattr(np, operation)(a - b)
    assert actual.tobytes() == expected.tobytes()
    assert a.tobytes() == before[0].tobytes() and b.tobytes() == before[1].tobytes()
    stats = provider._crossjit_stats()
    assert any('|u:1:0' in key and row.get('execute', 0)>0
               for key, row in stats['native_candidates'].items())
    assert stats['internal_allocations'] == 0
    assert stats['internal_buffer_bytes'] == 0


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('operation', ['add', 'subtract', 'multiply', 'minimum', 'maximum'])
@pytest.mark.parametrize('reverse', [False, True])
def test_private_binary_buffer_reuse(dtype, operation, reverse):
    scope = {'np': np}
    args = 'c, a @ b' if reverse else 'a @ b, c'
    exec(f'def f(a, b, c):\n    for i in range(300):\n        r=np.{operation}({args})\n    return r', scope)
    a = np.random.default_rng(30).uniform(.25, .75, (5, 3)).astype(dtype)
    b = np.random.default_rng(31).uniform(.25, .75, (3, 7)).astype(dtype)
    c = np.ones((5, 7), dtype=dtype)
    before = tuple(v.tobytes() for v in (a, b, c))
    product = a @ b
    expected = getattr(np, operation)(c, product) if reverse else getattr(np, operation)(product, c)
    actual = scope['f'](a, b, c)
    assert actual.tobytes() == expected.tobytes()
    assert tuple(v.tobytes() for v in (a, b, c)) == before
    stats = provider._crossjit_stats()
    assert any('|u:1:0:2' in key and row.get('execute', 0)>0
               for key, row in stats['native_candidates'].items())
    assert stats['internal_allocations'] == stats['internal_buffer_bytes'] == 0


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('operation', ['minimum', 'maximum'])
@pytest.mark.parametrize('shape', [(2, 32, 1), (2, 1, 32, 1, 1)])
def test_minmax_trailing_singleton_signed_zero(dtype, operation, shape):
    scope = {'np': np}
    exec(f'def f(a, b):\n    for i in range(300):\n        r=np.{operation}(a + b, b)\n    return r', scope)
    a, b = np.zeros(shape, dtype=dtype), -np.zeros(shape, dtype=dtype)
    actual = scope['f'](a, b)
    expected = getattr(np, operation)(a + b, b)
    assert actual.tobytes() == expected.tobytes()
    assert provider._crossjit_stats()['native_exec_hit'] > 0


@pytest.mark.parametrize('view', [False, True])
def test_input_cleanup_weakref_callback_is_a_boundary(view):
    state = {'callbacks': False}
    refs = []
    def make(x):
        base = np.ones(34 if view else 17)
        if state['callbacks']:
            refs.append(weakref.ref(base, lambda _ref: x.fill(10.)))
        return base[::2] if view else base
    def f(x, count):
        for _ in range(count):
            x.fill(1.)
            result = (make(x) + 1) + x
        return result
    x = np.zeros(17)
    assert np.all(f(x, 300) == 3.)
    state['callbacks'] = True
    actual = f(x, 300)
    assert np.all(actual == 12.)
    assert all(ref() is None for ref in refs)


def test_callback_added_after_prepare_fails_guard():
    def f(a, b):
        for _ in range(300):
            result = (a + 1) + b
        return result
    a, b = np.ones(17), np.ones(17)
    assert np.all(f(a, b) == 3.)
    prepared = {key for key, row in provider._crossjit_stats()['native_candidates'].items()
                if int(key.split('|')[4]) == 2 and row.get('execute', 0)}
    assert prepared
    calls = []
    ref = weakref.ref(a, lambda _ref: calls.append('released'))
    provider._crossjit_reset_stats()
    assert np.all(f(a, b) == 3.)
    stats = provider._crossjit_stats()
    assert stats['guard_miss'] > 0
    # A newly prepared suffix using only the callback-free temporary and b
    # is safe. The guarded complete plan containing a must stop executing.
    assert all(stats['native_candidates'].get(key, {}).get('execute', 0) == 0
               for key in prepared)
    assert calls == []
    del a
    gc.collect()
    assert ref() is None and calls == ['released']


def test_function_names_do_not_enter_region_keys():
    keys = []
    a, b = np.ones(17), np.ones(17)
    expected = np.sqrt(a + b) + b
    for name in ('atax', 'bicg', 'unrelated_function'):
        scope = {'np': np}
        source = (f'def {name}(a, b):\n'
                  '    for _ in range(300):\n'
                  '        result = np.sqrt(a + b) + b\n'
                  '    return result\n')
        exec(compile(source, name + '_source.py', 'exec'), scope)
        provider._crossjit_reset_stats()
        actual = scope[name](a, b)
        executed = {key for key, row in provider._crossjit_stats()['native_candidates'].items()
                    if row.get('execute', 0)}
        assert executed
        assert np.array_equal(actual, expected)
        keys.append(executed)
    assert keys[0] == keys[1] == keys[2]


@pytest.mark.parametrize('scalar', [int, np.int64, np.longlong])
@pytest.mark.parametrize('operation', ['minimum', 'maximum'])
def test_integer_only_scalar_domain_preserves_publication(scalar, operation):
    scope = {'np': np}
    exec(f'def f(a, b):\n    for _ in range(300):\n        result = np.{operation}(a, b)\n    return result', scope)
    a, b = scalar(2**62 + 1), scalar(2**62 + 3)
    expected = getattr(np, operation)(a, b)
    actual = scope['f'](a, b)
    assert type(actual) is type(expected)
    assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
    assert provider._crossjit_stats()['exec_hit'] == 0


def integer_loop(a, b, c):
    for _ in range(300):
        result = a * c + b * c + c
    return result


@pytest.mark.parametrize('scalar', [np.int64(3), 3])
def test_integer_scalars_preserve_wraparound(scalar):
    a = np.array([2**62, -(2**62), -1, 0, 1], dtype=np.int64)
    b = np.arange(5, dtype=np.int64)
    expected = a * scalar + b * scalar + scalar
    np.testing.assert_array_equal(integer_loop(a, b, scalar), expected, strict=True)
    assert provider._crossjit_stats()['native_exec_hit'] > 0


def test_visible_out_remains_a_boundary():
    def f(a, b, out):
        for _ in range(300):
            result = np.add(a, b, out)
        return result
    a, b, out = np.ones(103), np.ones(103), np.zeros(103)
    assert f(a, b, out) is out
    np.testing.assert_array_equal(out, np.full(103, 2.))
    assert provider._crossjit_stats()['native_exec_hit'] == 0


def floyd_expression(a, left, right):
    for _ in range(300):
        result = np.minimum(a, np.add.outer(left, right))
    return result


@pytest.mark.parametrize('dtype', [np.int32, np.int64])
def test_signed_integer_outer_minimum(dtype):
    limit = np.iinfo(dtype).max
    a = np.arange(143, dtype=dtype).reshape(11, 13)
    left = np.arange(11, dtype=dtype) + limit
    right = -np.arange(13, dtype=dtype)
    expected = np.minimum(a, np.add.outer(left, right))
    np.testing.assert_array_equal(floyd_expression(a, left, right), expected, strict=True)
    stats = provider._crossjit_stats()
    assert any(key.startswith('S1|') and int(key.split('|')[4]) == 2 and row['execute']
               for key, row in stats['native_candidates'].items())
    assert stats['exec_miss'] == stats['exec_error'] == 0


@pytest.mark.parametrize('operation', ['sum', 'max'])
def test_convenience_reduction_parameters(operation):
    scope = {'np': np}
    exec(f'def f(a):\n    for i in range(300):\n        r=np.{operation}(a,axis=1,keepdims=True)\n    return r', scope)
    a = np.random.default_rng(37).normal(size=(11, 137)).astype(np.float32)
    expected = getattr(np, operation)(a, axis=1, keepdims=True)
    np.testing.assert_array_equal(scope['f'](a), expected, strict=True)
    assert provider._crossjit_stats()['native_exec_hit'] > 0


def test_minimum_signed_zero_and_nan():
    a = np.array([0., -0., np.nan, np.inf, -np.inf])
    b = np.array([-0., 0., 1., np.inf, -np.inf])
    actual = callable_loop(np.minimum, a, b)
    expected = np.minimum(a, b)
    assert actual.tobytes() == expected.tobytes()


def test_dot_scalar_publication():
    def f(a, b):
        for _ in range(300):
            result = np.dot(a, b)
        return result
    a = np.arange(103, dtype=np.float32)[::-1]
    b = np.arange(103, dtype=np.float32)
    expected = np.dot(a, b)
    actual = f(a, b)
    assert type(actual) is np.float32
    assert actual.tobytes() == expected.tobytes()
    assert provider._crossjit_stats()['native_exec_hit'] > 0


def test_public_reduction_dependency_mutation_falls_back(monkeypatch):
    from numpy._core import fromnumeric
    def f(a):
        for _ in range(300):
            result = np.sum(a)
        return result
    a = np.arange(103., dtype=np.float64)
    assert f(a) == np.sum(a)
    assert provider._crossjit_stats()['native_exec_hit'] > 0
    calls = []
    def replacement(*args, **kwargs):
        calls.append(1)
        return np.float64(17.)
    monkeypatch.setattr(fromnumeric, '_wrapreduction', replacement)
    assert f(a) == np.float64(17.)
    assert calls
    assert provider._crossjit_stats()['exec_error'] == 0


def test_public_reductions_default_and_positional_contracts():
    def f(a):
        for _ in range(300):
            result = np.max(a, 1, None, True) + np.sum(a, 1, None, None, True)
        return result
    a = np.random.default_rng(82).normal(size=(7, 137)).astype(np.float32)
    expected = np.max(a, 1, None, True) + np.sum(a, 1, None, None, True)
    np.testing.assert_array_equal(f(a), expected, strict=True)
    assert provider._crossjit_stats()['native_exec_hit'] > 0
    assert provider._crossjit_stats()['exec_miss'] == 0


def test_inplace_is_a_structural_boundary():
    def f(a, b):
        for _ in range(300):
            a += b
            a /= b
        return a
    a, b = np.ones(137), np.full(137, 2.)
    expected = a.copy()
    for _ in range(300):
        expected += b
        expected /= b
    assert f(a, b) is a
    np.testing.assert_array_equal(a, expected, strict=True)
    stats = provider._crossjit_stats()
    assert stats['native_exec_hit'] == stats['exec_miss'] == stats['exec_error'] == 0


def test_external_buffer_owner_keeps_a_boundary():
    class Owner:
        def __init__(self):
            self.storage = np.arange(137., dtype=np.float64)
            self.__array_interface__ = self.storage.__array_interface__
    owner = Owner()
    a = np.asarray(owner)
    assert a.base is owner
    def f(a, b):
        for _ in range(300):
            result = a + b
        return result
    b = np.arange(137., dtype=np.float64)
    np.testing.assert_array_equal(f(a, b), a + b, strict=True)
    assert provider._crossjit_stats()['native_exec_hit'] == 0


def test_fortran_elementwise_preserves_publication_layout():
    a = np.asfortranarray(np.arange(77., dtype=np.float64).reshape(7, 11))
    b = np.asfortranarray(a * .25)
    expected = a + b
    def f(a, b):
        for _ in range(300):
            result = a + b
        return result
    actual = f(a, b)
    np.testing.assert_array_equal(actual, expected, strict=True)
    assert actual.strides == expected.strides
    assert provider._crossjit_stats()['native_exec_hit'] == 0


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_zero_dimensional_public_reduction(dtype):
    def f(a):
        for _ in range(300):
            result = np.sum(a, keepdims=True)
        return result
    a = np.array(1.25, dtype=dtype)
    expected = np.sum(a, keepdims=True)
    actual = f(a)
    assert type(actual) is type(expected)
    assert actual.tobytes() == expected.tobytes()
    assert provider._crossjit_stats()['native_exec_hit'] > 0
    matrix = np.arange(77, dtype=dtype).reshape(7, 11)
    np.testing.assert_array_equal(f(matrix), np.sum(matrix, keepdims=True), strict=True)
    assert provider._crossjit_stats()['exec_error'] == 0


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('operation', ['minimum', 'maximum'])
def test_vectorized_minmax_signed_zero_in_complete_region(dtype, operation):
    scope = {'np': np}
    exec(f'def f(a,b):\n    for i in range(300):\n        r=np.{operation}(a+b,b)\n    return r', scope)
    a = np.zeros(64, dtype=dtype)
    b = np.full(64, -0., dtype=dtype)
    actual = scope['f'](a, b)
    expected = getattr(np, operation)(a+b, b)
    assert actual.tobytes() == expected.tobytes()
    assert any(k.startswith('S1|') and int(k.split('|')[4]) == 2 and row['execute']
               for k, row in provider._crossjit_stats()['native_candidates'].items())
