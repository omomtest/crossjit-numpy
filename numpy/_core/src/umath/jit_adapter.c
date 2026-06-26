#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <opcode.h>
#include <stdint.h>

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

#ifndef PYJIT_EXT_ABI_VERSION
#define PYJIT_EXT_ABI_VERSION 1u

typedef enum {
    PYJIT_EXT_HIT = 1,
    PYJIT_EXT_MISS = 0,
    PYJIT_EXT_ERROR = -1,
} PyJitExtResult;

typedef enum {
    PYJIT_EXT_TRACE_BINARY,
    PYJIT_EXT_TRACE_CALL,
} PyJitExtTraceKind;

typedef struct _PyJitExtProvider PyJitExtProvider;
typedef struct _PyJitExtPlan PyJitExtPlan;
typedef struct _PyJitExtSiteState PyJitExtSiteState;

typedef struct {
    PyJitExtTraceKind kind;
    union {
        struct {
            int binary_oparg;
            PyTypeObject *left_type;
            PyTypeObject *right_type;
        } binary;
        struct {
            PyObject *callable;
            Py_ssize_t nargs;
            uint32_t call_flags;
            PyObject *kwnames;
        } call;
    } u;
} PyJitExtTraceRequest;

struct _PyJitExtProvider {
    uint32_t abi_version;
    uint32_t struct_size;
    const char *provider_name;
    uint32_t provider_flags;
    PyJitExtPlan *(*try_plan)(
        const PyJitExtTraceRequest *request,
        PyJitExtSiteState *site);
    void (*free_plan)(PyJitExtPlan *plan);
};

typedef enum {
    PYJIT_EXT_PLAN_BINARY,
    PYJIT_EXT_PLAN_CALL,
} PyJitExtPlanKind;

typedef PyJitExtResult (*PyJitExtBinaryEvalFunc)(
    void *payload, PyObject *left, PyObject *right, PyObject **result);
typedef PyJitExtResult (*PyJitExtCallEvalFunc)(
    void *payload, PyObject *callable, PyObject *const *args,
    Py_ssize_t nargs, PyObject **result);

struct _PyJitExtPlan {
    uint32_t abi_version;
    uint32_t struct_size;
    PyJitExtPlanKind kind;
    PyJitExtProvider *provider;
    uint32_t flags;
    uint32_t op_id;
    void *payload;
    union {
        PyJitExtBinaryEvalFunc binary_eval;
        PyJitExtCallEvalFunc call_eval;
    } eval;
    void (*free_payload)(void *payload);
    const char *debug_name;
};

struct _PyJitExtSiteState {
    uint16_t hit_count;
    uint16_t miss_count;
    uint16_t decline_count;
    uint8_t disabled;
    uint8_t retry_counter;
    uint32_t last_plan_op_id;
    PyJitExtProvider *last_provider;
};

#define PYJIT_EXT_PLAN_REQUIRES_GIL          (1u << 0)
#define PYJIT_EXT_PLAN_NO_PYTHON_CALLBACK    (1u << 2)
#define PYJIT_EXT_PLAN_ALLOCATES_RESULT      (1u << 3)
#define PYJIT_CALL_HAS_KWARGS                (1u << 0)

PyAPI_FUNC(int) PyJit_RegisterBinaryProvider(
    PyTypeObject *left_type,
    PyTypeObject *right_type,
    int binary_oparg,
    PyJitExtProvider *provider);
PyAPI_FUNC(int) PyJit_RegisterCallableProvider(
    PyObject *callable,
    PyJitExtProvider *provider);
#endif

extern PyTypeObject PyUFunc_Type;

typedef enum {
    NUMPY_JIT_OP_ADD,
    NUMPY_JIT_OP_SUBTRACT,
    NUMPY_JIT_OP_MULTIPLY,
    NUMPY_JIT_OP_TRUE_DIVIDE,
    NUMPY_JIT_OP_MATMUL,
} NumpyJitOp;

typedef enum {
    NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE,
} NumpyJitLayoutKind;

typedef struct {
    NumpyJitOp op;
    int binary_oparg;
    PyObject *known_callable;
    int arity;
    const char *name;
} NumpyJitOpDef;

typedef struct {
    int type_num;
    int itemsize;
    uint32_t flags;
    const char *name;
} NumpyJitDTypeDef;

typedef struct {
    NumpyJitOp op;
    int left_dtype;
    int right_dtype;
    int result_dtype;
    NumpyJitLayoutKind layout;
    uint32_t min_elements;
    uint32_t flags;
    PyJitExtBinaryEvalFunc binary_eval;
    PyJitExtCallEvalFunc call_eval;
} NumpyJitCandidate;

typedef struct {
    NumpyJitOp op;
    NumpyJitLayoutKind layout;
    int left_dtype;
    int right_dtype;
    int result_dtype;
    int itemsize;
    PyObject *expected_callable;
    const NumpyJitCandidate *candidate;
} NumpyJitPlanPayload;

static PyJitExtPlan *numpy_jit_try_plan(
    const PyJitExtTraceRequest *request, PyJitExtSiteState *site);
static void numpy_jit_free_plan(PyJitExtPlan *plan);

static PyJitExtProvider numpy_jit_provider = {
    .abi_version = PYJIT_EXT_ABI_VERSION,
    .struct_size = sizeof(PyJitExtProvider),
    .provider_name = "numpy",
    .provider_flags = 0,
    .try_plan = numpy_jit_try_plan,
    .free_plan = numpy_jit_free_plan,
};

static NumpyJitOpDef numpy_jit_ops[] = {
    { NUMPY_JIT_OP_ADD, NB_ADD, NULL, 2, "add" },
    { NUMPY_JIT_OP_SUBTRACT, NB_SUBTRACT, NULL, 2, "subtract" },
    { NUMPY_JIT_OP_MULTIPLY, NB_MULTIPLY, NULL, 2, "multiply" },
    { NUMPY_JIT_OP_TRUE_DIVIDE, NB_TRUE_DIVIDE, NULL, 2, "true_divide" },
    { NUMPY_JIT_OP_MATMUL, NB_MATRIX_MULTIPLY, NULL, 2, "matmul" },
};

static NumpyJitDTypeDef numpy_jit_dtypes[] = {
    { NPY_DOUBLE, sizeof(double), 0, "float64" },
};

static PyJitExtResult numpy_elemwise_f64_binary_eval(
    void *payload, PyObject *left, PyObject *right, PyObject **result);
static PyJitExtResult numpy_ufunc2_f64_call_eval(
    void *payload, PyObject *callable, PyObject *const *args,
    Py_ssize_t nargs, PyObject **result);

static NumpyJitCandidate numpy_jit_candidates[] = {
    {
        NUMPY_JIT_OP_ADD,
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
        NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE,
        0, 0,
        numpy_elemwise_f64_binary_eval,
        numpy_ufunc2_f64_call_eval,
    },
    {
        NUMPY_JIT_OP_SUBTRACT,
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
        NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE,
        0, 0,
        numpy_elemwise_f64_binary_eval,
        numpy_ufunc2_f64_call_eval,
    },
    {
        NUMPY_JIT_OP_MULTIPLY,
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
        NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE,
        0, 0,
        numpy_elemwise_f64_binary_eval,
        numpy_ufunc2_f64_call_eval,
    },
    {
        NUMPY_JIT_OP_TRUE_DIVIDE,
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
        NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE,
        0, 0,
        numpy_elemwise_f64_binary_eval,
        numpy_ufunc2_f64_call_eval,
    },
};

static int
resolve_binary_oparg(int binary_oparg, NumpyJitOp *op)
{
    for (Py_ssize_t i = 0; i < Py_ARRAY_LENGTH(numpy_jit_ops); i++) {
        if (numpy_jit_ops[i].binary_oparg == binary_oparg) {
            *op = numpy_jit_ops[i].op;
            return 1;
        }
    }
    return 0;
}

static int
resolve_known_ufunc(PyObject *callable, NumpyJitOp *op)
{
    for (Py_ssize_t i = 0; i < Py_ARRAY_LENGTH(numpy_jit_ops); i++) {
        if (numpy_jit_ops[i].known_callable == callable) {
            *op = numpy_jit_ops[i].op;
            return 1;
        }
    }
    return 0;
}

static const NumpyJitCandidate *
find_candidate(NumpyJitOp op, int left_dtype, int right_dtype,
               NumpyJitLayoutKind layout)
{
    for (Py_ssize_t i = 0; i < Py_ARRAY_LENGTH(numpy_jit_candidates); i++) {
        const NumpyJitCandidate *candidate = &numpy_jit_candidates[i];
        if (candidate->op == op &&
            candidate->left_dtype == left_dtype &&
            candidate->right_dtype == right_dtype &&
            candidate->layout == layout) {
            return candidate;
        }
    }
    return NULL;
}

static PyJitExtPlan *
make_numpy_plan(const NumpyJitCandidate *candidate, PyObject *callable)
{
    NumpyJitPlanPayload *payload = PyMem_Calloc(1, sizeof(*payload));
    if (payload == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    payload->op = candidate->op;
    payload->layout = candidate->layout;
    payload->left_dtype = candidate->left_dtype;
    payload->right_dtype = candidate->right_dtype;
    payload->result_dtype = candidate->result_dtype;
    payload->itemsize = numpy_jit_dtypes[0].itemsize;
    payload->expected_callable = callable;
    Py_XINCREF(callable);
    payload->candidate = candidate;

    PyJitExtPlan *plan = PyMem_Calloc(1, sizeof(*plan));
    if (plan == NULL) {
        Py_XDECREF(payload->expected_callable);
        PyMem_Free(payload);
        PyErr_NoMemory();
        return NULL;
    }
    plan->abi_version = PYJIT_EXT_ABI_VERSION;
    plan->struct_size = sizeof(*plan);
    plan->kind = callable == NULL ? PYJIT_EXT_PLAN_BINARY : PYJIT_EXT_PLAN_CALL;
    plan->provider = &numpy_jit_provider;
    plan->flags = PYJIT_EXT_PLAN_REQUIRES_GIL |
                  PYJIT_EXT_PLAN_NO_PYTHON_CALLBACK |
                  PYJIT_EXT_PLAN_ALLOCATES_RESULT;
    plan->op_id = (uint32_t)candidate->op;
    plan->payload = payload;
    if (plan->kind == PYJIT_EXT_PLAN_BINARY) {
        plan->eval.binary_eval = candidate->binary_eval;
    }
    else {
        plan->eval.call_eval = candidate->call_eval;
    }
    plan->free_payload = NULL;
    plan->debug_name = "numpy.elemwise.f64.c_contig";
    return plan;
}

static PyJitExtPlan *
numpy_jit_try_binary_plan(const PyJitExtTraceRequest *request)
{
    if (request->u.binary.left_type != &PyArray_Type ||
        request->u.binary.right_type != &PyArray_Type) {
        return NULL;
    }

    NumpyJitOp op;
    if (!resolve_binary_oparg(request->u.binary.binary_oparg, &op)) {
        return NULL;
    }

    const NumpyJitCandidate *candidate = find_candidate(
        op, NPY_DOUBLE, NPY_DOUBLE, NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE);
    if (candidate == NULL) {
        return NULL;
    }
    return make_numpy_plan(candidate, NULL);
}

static PyJitExtPlan *
numpy_jit_try_call_plan(const PyJitExtTraceRequest *request)
{
    if (request->u.call.nargs != 2 ||
        (request->u.call.call_flags & PYJIT_CALL_HAS_KWARGS)) {
        return NULL;
    }

    NumpyJitOp op;
    if (!resolve_known_ufunc(request->u.call.callable, &op)) {
        return NULL;
    }

    const NumpyJitCandidate *candidate = find_candidate(
        op, NPY_DOUBLE, NPY_DOUBLE, NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE);
    if (candidate == NULL) {
        return NULL;
    }
    return make_numpy_plan(candidate, request->u.call.callable);
}

static PyJitExtPlan *
numpy_jit_try_plan(const PyJitExtTraceRequest *request, PyJitExtSiteState *site)
{
    if (site != NULL) {
        site->last_provider = &numpy_jit_provider;
    }
    switch (request->kind) {
        case PYJIT_EXT_TRACE_BINARY:
            return numpy_jit_try_binary_plan(request);
        case PYJIT_EXT_TRACE_CALL:
            return numpy_jit_try_call_plan(request);
        default:
            return NULL;
    }
}

static void
numpy_jit_free_plan(PyJitExtPlan *plan)
{
    if (plan == NULL) {
        return;
    }
    NumpyJitPlanPayload *payload = (NumpyJitPlanPayload *)plan->payload;
    if (payload != NULL) {
        Py_XDECREF(payload->expected_callable);
        PyMem_Free(payload);
    }
    PyMem_Free(plan);
}

static int
same_shape_c_contig_f64(NumpyJitPlanPayload *payload,
                        PyObject *left, PyObject *right,
                        PyArrayObject **a_out, PyArrayObject **b_out,
                        npy_intp *size_out)
{
    if (!PyArray_CheckExact(left) || !PyArray_CheckExact(right)) {
        return 0;
    }
    PyArrayObject *a = (PyArrayObject *)left;
    PyArrayObject *b = (PyArrayObject *)right;
    if (PyArray_TYPE(a) != payload->left_dtype ||
        PyArray_TYPE(b) != payload->right_dtype ||
        !PyArray_IS_C_CONTIGUOUS(a) ||
        !PyArray_IS_C_CONTIGUOUS(b)) {
        return 0;
    }
    int ndim = PyArray_NDIM(a);
    if (ndim != PyArray_NDIM(b)) {
        return 0;
    }
    npy_intp size = 1;
    npy_intp *adims = PyArray_DIMS(a);
    npy_intp *bdims = PyArray_DIMS(b);
    for (int i = 0; i < ndim; i++) {
        if (adims[i] != bdims[i]) {
            return 0;
        }
        size *= adims[i];
    }
    *a_out = a;
    *b_out = b;
    *size_out = size;
    return 1;
}

static PyJitExtResult
numpy_elemwise_eval_common(NumpyJitPlanPayload *payload,
                           PyObject *left, PyObject *right,
                           PyObject **result)
{
    PyArrayObject *a;
    PyArrayObject *b;
    npy_intp size;
    if (!same_shape_c_contig_f64(payload, left, right, &a, &b, &size)) {
        return PYJIT_EXT_MISS;
    }

    PyArrayObject *out = (PyArrayObject *)PyArray_EMPTY(
        PyArray_NDIM(a), PyArray_DIMS(a), payload->result_dtype, 0);
    if (out == NULL) {
        return PYJIT_EXT_ERROR;
    }

    const double *left_data = (const double *)PyArray_BYTES(a);
    const double *right_data = (const double *)PyArray_BYTES(b);
    double *out_data = (double *)PyArray_BYTES(out);
    switch (payload->op) {
        case NUMPY_JIT_OP_ADD:
            for (npy_intp i = 0; i < size; i++) {
                out_data[i] = left_data[i] + right_data[i];
            }
            break;
        case NUMPY_JIT_OP_SUBTRACT:
            for (npy_intp i = 0; i < size; i++) {
                out_data[i] = left_data[i] - right_data[i];
            }
            break;
        case NUMPY_JIT_OP_MULTIPLY:
            for (npy_intp i = 0; i < size; i++) {
                out_data[i] = left_data[i] * right_data[i];
            }
            break;
        case NUMPY_JIT_OP_TRUE_DIVIDE:
            for (npy_intp i = 0; i < size; i++) {
                out_data[i] = left_data[i] / right_data[i];
            }
            break;
        default:
            Py_DECREF(out);
            return PYJIT_EXT_MISS;
    }

    *result = (PyObject *)out;
    return PYJIT_EXT_HIT;
}

static PyJitExtResult
numpy_elemwise_f64_binary_eval(void *payload, PyObject *left,
                               PyObject *right, PyObject **result)
{
    return numpy_elemwise_eval_common(
        (NumpyJitPlanPayload *)payload, left, right, result);
}

static PyJitExtResult
numpy_ufunc2_f64_call_eval(void *payload, PyObject *callable,
                           PyObject *const *args, Py_ssize_t nargs,
                           PyObject **result)
{
    NumpyJitPlanPayload *plan = (NumpyJitPlanPayload *)payload;
    if (nargs != 2 || callable != plan->expected_callable) {
        return PYJIT_EXT_MISS;
    }
    return numpy_elemwise_eval_common(plan, args[0], args[1], result);
}

NPY_NO_EXPORT int
jit_adapter_init(PyObject *umath_module)
{
    PyObject *dict = PyModule_GetDict(umath_module);
    if (dict == NULL) {
        return -1;
    }

    for (Py_ssize_t i = 0; i < Py_ARRAY_LENGTH(numpy_jit_ops); i++) {
        PyObject *ufunc = PyDict_GetItemString(dict, numpy_jit_ops[i].name);
        if (ufunc != NULL && PyObject_TypeCheck(ufunc, &PyUFunc_Type)) {
            numpy_jit_ops[i].known_callable = ufunc;
        }
    }

    for (Py_ssize_t i = 0; i < Py_ARRAY_LENGTH(numpy_jit_candidates); i++) {
        int oparg = -1;
        for (Py_ssize_t j = 0; j < Py_ARRAY_LENGTH(numpy_jit_ops); j++) {
            if (numpy_jit_ops[j].op == numpy_jit_candidates[i].op) {
                oparg = numpy_jit_ops[j].binary_oparg;
                break;
            }
        }
        if (oparg >= 0 &&
            PyJit_RegisterBinaryProvider(
                &PyArray_Type, &PyArray_Type, oparg,
                &numpy_jit_provider) < 0) {
            PyErr_Clear();
        }
    }

    for (Py_ssize_t i = 0; i < Py_ARRAY_LENGTH(numpy_jit_ops); i++) {
        PyObject *callable = numpy_jit_ops[i].known_callable;
        if (callable == NULL) {
            continue;
        }
        NumpyJitOp op = numpy_jit_ops[i].op;
        if (find_candidate(op, NPY_DOUBLE, NPY_DOUBLE,
                           NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE) == NULL) {
            continue;
        }
        if (PyJit_RegisterCallableProvider(callable, &numpy_jit_provider) < 0) {
            PyErr_Clear();
        }
    }
    return 0;
}
