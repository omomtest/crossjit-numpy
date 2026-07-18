#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <opcode.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "jit_kernel.h"

#ifndef PYJIT_EXT_ABI_VERSION
#define PYJIT_EXT_ABI_VERSION 2u

typedef enum {
    PYJIT_EXT_HIT = 1,
    PYJIT_EXT_MISS = 0,
    PYJIT_EXT_ERROR = -1,
} PyJitExtResult;

typedef enum {
    PYJIT_EXT_TRACE_BINARY,
    PYJIT_EXT_TRACE_CALL,
    PYJIT_EXT_TRACE_FUSED_BINARY2,
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
        struct {
            PyJitExtPlan *first_plan;
            PyJitExtPlan *second_plan;
            uint32_t flags;
        } fused_binary2;
    } u;
} PyJitExtTraceRequest;

#define PYJIT_EXT_FUSED_BINARY2_SECOND_CONSUMES_LEFT   (1u << 0)
#define PYJIT_EXT_FUSED_BINARY2_SECOND_CONSUMES_RIGHT  (1u << 1)

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const char *op_name;
    const char *layout_name;
    const char *left_role;
    const char *right_role;
    const char *result_role;
    const char *left_dtype;
    const char *right_dtype;
    const char *result_dtype;
    int itemsize;
    int64_t min_elements;
    const char *fused_op_name;
    const char *fused_pattern;
} PyJitExtPlanInfo;

struct _PyJitExtProvider {
    uint32_t abi_version;
    uint32_t struct_size;
    const char *provider_name;
    uint32_t provider_flags;
    PyJitExtPlan *(*try_plan)(
        const PyJitExtTraceRequest *request,
        PyJitExtSiteState *site);
    void (*free_plan)(PyJitExtPlan *plan);
    int (*describe_plan)(
        const PyJitExtPlan *plan,
        PyJitExtPlanInfo *info);
};

typedef enum {
    PYJIT_EXT_PLAN_BINARY,
    PYJIT_EXT_PLAN_CALL,
    PYJIT_EXT_PLAN_FUSED_BINARY2,
} PyJitExtPlanKind;

typedef PyJitExtResult (*PyJitExtBinaryEvalFunc)(
    void *payload, PyObject *left, PyObject *right, PyObject **result);
typedef PyJitExtResult (*PyJitExtCallEvalFunc)(
    void *payload, PyObject *callable, PyObject *const *args,
    Py_ssize_t nargs, PyObject **result);
typedef PyJitExtResult (*PyJitExtFusedBinary2EvalFunc)(
    void *payload, PyObject *first_left, PyObject *first_right,
    PyObject *second_other, PyObject **result);

typedef enum {
    PYJIT_EXT_STENCIL_PATCH_ABS64,
    PYJIT_EXT_STENCIL_PATCH_REL32,
} PyJitExtStencilPatchKind;

typedef enum {
    PYJIT_EXT_STENCIL_VALUE_PLAN,
    PYJIT_EXT_STENCIL_VALUE_PAYLOAD,
    PYJIT_EXT_STENCIL_VALUE_BINARY_EVAL,
    PYJIT_EXT_STENCIL_VALUE_CALL_EVAL,
    PYJIT_EXT_STENCIL_VALUE_FUSED_BINARY2_EVAL,
    PYJIT_EXT_STENCIL_VALUE_NEXT_INSTRUCTION,
    PYJIT_EXT_STENCIL_VALUE_TARGET,
    PYJIT_EXT_STENCIL_VALUE_ERROR_TARGET,
    PYJIT_EXT_STENCIL_VALUE_RECORD_EXT_HIT,
    PYJIT_EXT_STENCIL_VALUE_RECORD_EXT_MISS,
    PYJIT_EXT_STENCIL_VALUE_CUSTOM0,
    PYJIT_EXT_STENCIL_VALUE_CUSTOM1,
    PYJIT_EXT_STENCIL_VALUE_CUSTOM2,
    PYJIT_EXT_STENCIL_VALUE_CUSTOM3,
} PyJitExtStencilValueKind;

typedef enum {
    PYJIT_EXT_STENCIL_SECTION_CODE,
    PYJIT_EXT_STENCIL_SECTION_DATA,
} PyJitExtStencilSection;

typedef struct {
    uint32_t offset;
    uint16_t section;
    uint16_t patch_kind;
    uint16_t value_kind;
    uint16_t reserved;
    int64_t addend;
} PyJitExtStencilHole;

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const unsigned char *code;
    size_t code_size;
    const unsigned char *data;
    size_t data_size;
    const PyJitExtStencilHole *holes;
    size_t hole_count;
    const void *custom0;
    const void *custom1;
    const void *custom2;
    const void *custom3;
} PyJitExtStencil;

typedef const PyJitExtStencil *(*PyJitExtGetStencilFunc)(
    const PyJitExtPlan *plan,
    int opcode);

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
        PyJitExtFusedBinary2EvalFunc fused_binary2_eval;
    } eval;
    void (*free_payload)(void *payload);
    const char *debug_name;
    PyJitExtSiteState *site;
    const PyJitExtStencil *stencil;
    PyJitExtGetStencilFunc get_stencil;
};

#define PYJIT_EXT_GUARD_FAIL_REASON_COUNT 9

struct _PyJitExtSiteState {
    uint16_t hit_count;
    uint16_t miss_count;
    uint16_t decline_count;
    uint8_t disabled;
    uint8_t retry_counter;
    uint8_t trace_kind;
    uint32_t last_plan_op_id;
    uint32_t site_id;
    PyJitExtProvider *last_provider;
    const char *disable_reason;
    uint64_t optimizer_skip_after_disable;
    uint64_t guard_fail_counts[PYJIT_EXT_GUARD_FAIL_REASON_COUNT];
    int64_t element_count_min;
    int64_t element_count_max;
    int64_t element_count_last;
    uint16_t fused_site_count;
    uint16_t fused_hit_count;
    uint16_t fused_miss_count;
    const char *fused_pattern;
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

static PyJitExtResult numpy_jit_prepare_binary_contig_f64(
    void *payload, PyObject *left, PyObject *right, void *run_state);

#include "jit_stencils_generated.h"

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
    NUMPY_JIT_LAYOUT_C_CONTIG_ARRAY_SCALAR,
    NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY,
} NumpyJitLayoutKind;

typedef enum {
    NUMPY_JIT_FUSED_MUL_SCALAR_ADD_ARRAY,
    NUMPY_JIT_FUSED_ADD_ARRAY_MUL_SCALAR,
    NUMPY_JIT_FUSED_AXPBY,
} NumpyJitFusedOp;

typedef struct {
    NumpyJitOp op;
    int binary_oparg;
    PyObject *known_callable;
    int arity;
    const char *name;
} NumpyJitOpDef;

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
    npy_intp min_elements;
    PyObject *expected_callable;
    const NumpyJitCandidate *candidate;
    PyJitExtSiteState *site;
    int fused_op;
    uint32_t fused_flags;
} NumpyJitPlanPayload;

typedef struct {
    const double *left_data;
    const double *right_data;
    double *out_data;
    npy_intp size;
    PyObject *out;
} NumpyJitBinaryContigF64RunState;

static PyJitExtPlan *numpy_jit_try_plan(
    const PyJitExtTraceRequest *request, PyJitExtSiteState *site);
static void numpy_jit_free_plan(PyJitExtPlan *plan);
static int numpy_jit_describe_plan(
    const PyJitExtPlan *plan, PyJitExtPlanInfo *info);
static const PyJitExtStencil *numpy_jit_get_stencil(
    const PyJitExtPlan *plan, int opcode);

static PyJitExtProvider numpy_jit_provider = {
    .abi_version = PYJIT_EXT_ABI_VERSION,
    .struct_size = sizeof(PyJitExtProvider),
    .provider_name = "numpy",
    .provider_flags = 0,
    .try_plan = numpy_jit_try_plan,
    .free_plan = numpy_jit_free_plan,
    .describe_plan = numpy_jit_describe_plan,
};

static NumpyJitOpDef numpy_jit_ops[] = {
    { NUMPY_JIT_OP_ADD, NB_ADD, NULL, 2, "add" },
    { NUMPY_JIT_OP_SUBTRACT, NB_SUBTRACT, NULL, 2, "subtract" },
    { NUMPY_JIT_OP_MULTIPLY, NB_MULTIPLY, NULL, 2, "multiply" },
    { NUMPY_JIT_OP_TRUE_DIVIDE, NB_TRUE_DIVIDE, NULL, 2, "true_divide" },
    { NUMPY_JIT_OP_MATMUL, NB_MATRIX_MULTIPLY, NULL, 2, "matmul" },
};

static int
numpy_jit_itemsize_for_dtype(int dtype)
{
    switch (dtype) {
        case NPY_FLOAT:
            return (int)sizeof(float);
        case NPY_DOUBLE:
            return (int)sizeof(double);
        default:
            return 0;
    }
}

static PyJitExtResult numpy_elemwise_f64_binary_eval(
    void *payload, PyObject *left, PyObject *right, PyObject **result);
static PyJitExtResult numpy_elemwise_f64_array_scalar_eval(
    void *payload, PyObject *left, PyObject *right, PyObject **result);
static PyJitExtResult numpy_elemwise_f64_scalar_array_eval(
    void *payload, PyObject *left, PyObject *right, PyObject **result);
static PyJitExtResult numpy_ufunc2_f64_call_eval(
    void *payload, PyObject *callable, PyObject *const *args,
    Py_ssize_t nargs, PyObject **result);
static PyJitExtResult numpy_fused_binary2_eval(
    void *payload, PyObject *first_left, PyObject *first_right,
    PyObject *second_other, PyObject **result);
static PyObject *numpy_jit_fused_mul_scalar_add_array(
    PyObject *NPY_UNUSED(self), PyObject *args);
static PyObject *numpy_jit_fused_add_array_mul_scalar(
    PyObject *NPY_UNUSED(self), PyObject *args);
static PyObject *numpy_jit_fused_axpby(
    PyObject *NPY_UNUSED(self), PyObject *args);

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
    {
        NUMPY_JIT_OP_ADD,
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
        NUMPY_JIT_LAYOUT_C_CONTIG_ARRAY_SCALAR,
        0, 0,
        numpy_elemwise_f64_array_scalar_eval,
        NULL,
    },
    {
        NUMPY_JIT_OP_SUBTRACT,
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
        NUMPY_JIT_LAYOUT_C_CONTIG_ARRAY_SCALAR,
        0, 0,
        numpy_elemwise_f64_array_scalar_eval,
        NULL,
    },
    {
        NUMPY_JIT_OP_MULTIPLY,
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
        NUMPY_JIT_LAYOUT_C_CONTIG_ARRAY_SCALAR,
        0, 0,
        numpy_elemwise_f64_array_scalar_eval,
        NULL,
    },
    {
        NUMPY_JIT_OP_TRUE_DIVIDE,
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
        NUMPY_JIT_LAYOUT_C_CONTIG_ARRAY_SCALAR,
        0, 0,
        numpy_elemwise_f64_array_scalar_eval,
        NULL,
    },
    {
        NUMPY_JIT_OP_ADD,
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
        NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY,
        0, 0,
        numpy_elemwise_f64_scalar_array_eval,
        NULL,
    },
    {
        NUMPY_JIT_OP_SUBTRACT,
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
        NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY,
        0, 0,
        numpy_elemwise_f64_scalar_array_eval,
        NULL,
    },
    {
        NUMPY_JIT_OP_MULTIPLY,
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
        NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY,
        0, 0,
        numpy_elemwise_f64_scalar_array_eval,
        NULL,
    },
    {
        NUMPY_JIT_OP_TRUE_DIVIDE,
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
        NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY,
        0, 0,
        numpy_elemwise_f64_scalar_array_eval,
        NULL,
    },
};

static npy_intp numpy_jit_env_min_elements = -1;
static npy_intp numpy_jit_env_strided_min_elements = -1;
static int numpy_jit_env_enable_strided_same_shape = -1;
static int numpy_jit_env_enable_static_quick_disable = -1;
static int numpy_jit_env_enable_array_scalar = -1;
static int numpy_jit_env_enable_scalar_array = -1;
static int numpy_jit_stats_enabled_cache = -1;
static int numpy_jit_atexit_registered = 0;

#define NUMPY_JIT_SIGNATURE_LEN 192
#define NUMPY_JIT_DTYPE_NAME_LEN 32

typedef struct NumpyJitLayoutMissDetail {
    uint32_t site_id;
    NumpyJitOp op;
    NumpyJitLayoutKind entry_kind;
    int left_dtype;
    int right_dtype;
    int left_ndim;
    int right_ndim;
    char left_dtype_name[NUMPY_JIT_DTYPE_NAME_LEN];
    char right_dtype_name[NUMPY_JIT_DTYPE_NAME_LEN];
    char shape_signature[NUMPY_JIT_SIGNATURE_LEN];
    char stride_signature[NUMPY_JIT_SIGNATURE_LEN];
    int same_shape;
    int left_c_contiguous;
    int right_c_contiguous;
    int inner_contiguous;
    int negative_stride;
    uint64_t count;
    struct NumpyJitLayoutMissDetail *next;
} NumpyJitLayoutMissDetail;

typedef struct NumpyJitDtypeMissDetail {
    uint32_t site_id;
    NumpyJitOp op;
    NumpyJitLayoutKind entry_kind;
    int left_dtype;
    int right_dtype;
    char left_dtype_name[NUMPY_JIT_DTYPE_NAME_LEN];
    char right_dtype_name[NUMPY_JIT_DTYPE_NAME_LEN];
    int same_shape;
    int left_c_contiguous;
    int right_c_contiguous;
    uint64_t count;
    struct NumpyJitDtypeMissDetail *next;
} NumpyJitDtypeMissDetail;

typedef enum {
    NUMPY_JIT_GUARD_NOT_EXACT_ARRAY,
    NUMPY_JIT_GUARD_DTYPE_MISMATCH,
    NUMPY_JIT_GUARD_NOT_C_CONTIGUOUS,
    NUMPY_JIT_GUARD_NDIM_MISMATCH,
    NUMPY_JIT_GUARD_SHAPE_MISMATCH,
    NUMPY_JIT_GUARD_MIN_ELEMENTS,
    NUMPY_JIT_GUARD_NOT_SCALAR,
    NUMPY_JIT_GUARD_SCALAR_CONVERT_FAILED,
    NUMPY_JIT_GUARD_STRIDED_MIN_ELEMENTS,
    NUMPY_JIT_GUARD_COUNT
} NumpyJitGuardFailReason;

static uint64_t numpy_jit_guard_fail_counts[NUMPY_JIT_GUARD_COUNT];
static NumpyJitLayoutMissDetail *numpy_jit_layout_miss_details = NULL;
static NumpyJitDtypeMissDetail *numpy_jit_dtype_miss_details = NULL;

static int numpy_jit_stats_enabled(void);

static const char *
numpy_jit_op_name(NumpyJitOp op)
{
    for (Py_ssize_t i = 0; i < Py_ARRAY_LENGTH(numpy_jit_ops); i++) {
        if (numpy_jit_ops[i].op == op) {
            return numpy_jit_ops[i].name;
        }
    }
    return "unknown";
}

static const char *
numpy_jit_layout_name(NumpyJitLayoutKind layout)
{
    switch (layout) {
        case NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE:
            return "c_contig_same_shape";
        case NUMPY_JIT_LAYOUT_C_CONTIG_ARRAY_SCALAR:
            return "c_contig_array_scalar";
        case NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY:
            return "c_contig_scalar_array";
        default:
            return "unknown";
    }
}

static const char *
numpy_jit_dtype_static_name(int dtype)
{
    switch (dtype) {
        case NPY_DOUBLE:
            return "float64";
        case NPY_FLOAT:
            return "float32";
        case NPY_LONG:
            return "int64";
        case NPY_INT:
            return "int32";
        case NPY_BOOL:
            return "bool";
        case NPY_CDOUBLE:
            return "complex128";
        case NPY_CFLOAT:
            return "complex64";
        case -1:
            return "python_scalar";
        case -2:
            return "non_ndarray";
        default:
            return "unknown";
    }
}

static const char *
numpy_jit_fused_op_name(NumpyJitFusedOp op)
{
    switch (op) {
        case NUMPY_JIT_FUSED_MUL_SCALAR_ADD_ARRAY:
            return "mul_scalar_add_array";
        case NUMPY_JIT_FUSED_ADD_ARRAY_MUL_SCALAR:
            return "add_array_mul_scalar";
        case NUMPY_JIT_FUSED_AXPBY:
            return "axpby";
        default:
            return "unknown";
    }
}

static void
numpy_jit_layout_roles(NumpyJitLayoutKind layout,
                       const char **left_role,
                       const char **right_role,
                       const char **result_role)
{
    *result_role = "array";
    switch (layout) {
        case NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE:
            *left_role = "array";
            *right_role = "array";
            break;
        case NUMPY_JIT_LAYOUT_C_CONTIG_ARRAY_SCALAR:
            *left_role = "array";
            *right_role = "scalar";
            break;
        case NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY:
            *left_role = "scalar";
            *right_role = "array";
            break;
        default:
            *left_role = "unknown";
            *right_role = "unknown";
            *result_role = "unknown";
            break;
    }
}

static int
numpy_jit_describe_plan(const PyJitExtPlan *plan, PyJitExtPlanInfo *info)
{
    if (plan == NULL || info == NULL ||
        plan->provider != &numpy_jit_provider ||
        plan->payload == NULL) {
        return 0;
    }
    NumpyJitPlanPayload *payload = (NumpyJitPlanPayload *)plan->payload;
    memset(info, 0, sizeof(*info));
    info->abi_version = PYJIT_EXT_ABI_VERSION;
    info->struct_size = sizeof(*info);
    info->op_name = numpy_jit_op_name(payload->op);
    info->layout_name = numpy_jit_layout_name(payload->layout);
    numpy_jit_layout_roles(payload->layout,
                           &info->left_role,
                           &info->right_role,
                           &info->result_role);
    info->left_dtype = numpy_jit_dtype_static_name(payload->left_dtype);
    info->right_dtype = numpy_jit_dtype_static_name(payload->right_dtype);
    info->result_dtype = numpy_jit_dtype_static_name(payload->result_dtype);
    info->itemsize = payload->itemsize;
    info->min_elements = (int64_t)payload->min_elements;
    if (plan->kind == PYJIT_EXT_PLAN_FUSED_BINARY2) {
        info->fused_op_name = numpy_jit_fused_op_name(
            (NumpyJitFusedOp)payload->fused_op);
        if (payload->fused_flags &
            PYJIT_EXT_FUSED_BINARY2_SECOND_CONSUMES_RIGHT) {
            info->fused_pattern =
                payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY ?
                "a+scalar*b" : "a+b*scalar";
        }
        else {
            info->fused_pattern =
                payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY ?
                "scalar*a+b" : "a*scalar+b";
        }
    }
    return 1;
}

static void
numpy_jit_dtype_name(int dtype, char *buf, size_t size)
{
    const char *name = NULL;
    switch (dtype) {
        case NPY_DOUBLE:
            name = "float64";
            break;
        case NPY_FLOAT:
            name = "float32";
            break;
        case NPY_LONG:
            name = "int64";
            break;
        case NPY_INT:
            name = "int32";
            break;
        case NPY_BOOL:
            name = "bool";
            break;
        case NPY_CDOUBLE:
            name = "complex128";
            break;
        case NPY_CFLOAT:
            name = "complex64";
            break;
        case -1:
            name = "python_scalar";
            break;
        case -2:
            name = "non_ndarray";
            break;
        default:
            break;
    }
    if (name != NULL) {
        snprintf(buf, size, "%s", name);
    }
    else {
        snprintf(buf, size, "dtype_%d", dtype);
    }
}

static int
numpy_jit_object_dtype(PyObject *obj)
{
    if (PyArray_CheckExact(obj)) {
        return PyArray_TYPE((PyArrayObject *)obj);
    }
    if (PyFloat_CheckExact(obj) || PyLong_CheckExact(obj)) {
        return -1;
    }
    return -2;
}

static int
numpy_jit_object_ndim(PyObject *obj)
{
    return PyArray_CheckExact(obj) ? PyArray_NDIM((PyArrayObject *)obj) : 0;
}

static int
numpy_jit_object_c_contiguous(PyObject *obj)
{
    return PyArray_CheckExact(obj) ?
        PyArray_IS_C_CONTIGUOUS((PyArrayObject *)obj) : 1;
}

static void
numpy_jit_append_array_values(char *buf, size_t size, PyArrayObject *array,
                              int strides)
{
    size_t used = strlen(buf);
    if (used < size) {
        snprintf(buf + used, size - used, "[");
    }
    int ndim = PyArray_NDIM(array);
    npy_intp *values = strides ? PyArray_STRIDES(array) : PyArray_DIMS(array);
    for (int i = 0; i < ndim; i++) {
        used = strlen(buf);
        if (used < size) {
            snprintf(buf + used, size - used, "%s%" NPY_INTP_FMT,
                     i == 0 ? "" : ",", values[i]);
        }
    }
    used = strlen(buf);
    if (used < size) {
        snprintf(buf + used, size - used, "]");
    }
}

static void
numpy_jit_shape_signature(PyObject *left, PyObject *right,
                          char *buf, size_t size)
{
    buf[0] = '\0';
    if (PyArray_CheckExact(left)) {
        numpy_jit_append_array_values(buf, size, (PyArrayObject *)left, 0);
    }
    else {
        snprintf(buf, size, "scalar");
    }
    size_t used = strlen(buf);
    if (used < size) {
        snprintf(buf + used, size - used, "|");
    }
    if (PyArray_CheckExact(right)) {
        numpy_jit_append_array_values(buf, size, (PyArrayObject *)right, 0);
    }
    else {
        used = strlen(buf);
        if (used < size) {
            snprintf(buf + used, size - used, "scalar");
        }
    }
}

static void
numpy_jit_stride_signature(PyObject *left, PyObject *right,
                           char *buf, size_t size)
{
    buf[0] = '\0';
    if (PyArray_CheckExact(left)) {
        numpy_jit_append_array_values(buf, size, (PyArrayObject *)left, 1);
    }
    else {
        snprintf(buf, size, "scalar");
    }
    size_t used = strlen(buf);
    if (used < size) {
        snprintf(buf + used, size - used, "|");
    }
    if (PyArray_CheckExact(right)) {
        numpy_jit_append_array_values(buf, size, (PyArrayObject *)right, 1);
    }
    else {
        used = strlen(buf);
        if (used < size) {
            snprintf(buf + used, size - used, "scalar");
        }
    }
}

static int
numpy_jit_same_shape(PyObject *left, PyObject *right)
{
    if (!PyArray_CheckExact(left) || !PyArray_CheckExact(right)) {
        return 0;
    }
    PyArrayObject *a = (PyArrayObject *)left;
    PyArrayObject *b = (PyArrayObject *)right;
    int ndim = PyArray_NDIM(a);
    if (ndim != PyArray_NDIM(b)) {
        return 0;
    }
    npy_intp *adims = PyArray_DIMS(a);
    npy_intp *bdims = PyArray_DIMS(b);
    for (int i = 0; i < ndim; i++) {
        if (adims[i] != bdims[i]) {
            return 0;
        }
    }
    return 1;
}

static int
numpy_jit_negative_stride(PyObject *left, PyObject *right)
{
    PyObject *items[2] = {left, right};
    for (int obj = 0; obj < 2; obj++) {
        if (!PyArray_CheckExact(items[obj])) {
            continue;
        }
        PyArrayObject *array = (PyArrayObject *)items[obj];
        npy_intp *strides = PyArray_STRIDES(array);
        for (int i = 0; i < PyArray_NDIM(array); i++) {
            if (strides[i] < 0) {
                return 1;
            }
        }
    }
    return 0;
}

static int
numpy_jit_inner_contiguous(PyObject *left, PyObject *right)
{
    PyObject *items[2] = {left, right};
    for (int obj = 0; obj < 2; obj++) {
        if (!PyArray_CheckExact(items[obj])) {
            continue;
        }
        PyArrayObject *array = (PyArrayObject *)items[obj];
        int ndim = PyArray_NDIM(array);
        if (ndim <= 0) {
            continue;
        }
        if (PyArray_STRIDES(array)[ndim - 1] != PyArray_ITEMSIZE(array)) {
            return 0;
        }
    }
    return 1;
}

static void
numpy_jit_record_layout_miss(NumpyJitPlanPayload *payload,
                             PyObject *left, PyObject *right)
{
    if (!numpy_jit_stats_enabled() || payload == NULL) {
        return;
    }

    NumpyJitLayoutMissDetail detail;
    memset(&detail, 0, sizeof(detail));
    detail.site_id = payload->site != NULL ? payload->site->site_id : UINT32_MAX;
    detail.op = payload->op;
    detail.entry_kind = payload->layout;
    detail.left_dtype = numpy_jit_object_dtype(left);
    detail.right_dtype = numpy_jit_object_dtype(right);
    numpy_jit_dtype_name(detail.left_dtype, detail.left_dtype_name,
                         sizeof(detail.left_dtype_name));
    numpy_jit_dtype_name(detail.right_dtype, detail.right_dtype_name,
                         sizeof(detail.right_dtype_name));
    detail.left_ndim = numpy_jit_object_ndim(left);
    detail.right_ndim = numpy_jit_object_ndim(right);
    numpy_jit_shape_signature(left, right, detail.shape_signature,
                              sizeof(detail.shape_signature));
    numpy_jit_stride_signature(left, right, detail.stride_signature,
                               sizeof(detail.stride_signature));
    detail.same_shape = numpy_jit_same_shape(left, right);
    detail.left_c_contiguous = numpy_jit_object_c_contiguous(left);
    detail.right_c_contiguous = numpy_jit_object_c_contiguous(right);
    detail.inner_contiguous = numpy_jit_inner_contiguous(left, right);
    detail.negative_stride = numpy_jit_negative_stride(left, right);

    for (NumpyJitLayoutMissDetail *entry = numpy_jit_layout_miss_details;
         entry != NULL; entry = entry->next) {
        if (entry->site_id == detail.site_id &&
            entry->op == detail.op &&
            entry->entry_kind == detail.entry_kind &&
            entry->left_dtype == detail.left_dtype &&
            entry->right_dtype == detail.right_dtype &&
            entry->left_ndim == detail.left_ndim &&
            entry->right_ndim == detail.right_ndim &&
            entry->same_shape == detail.same_shape &&
            entry->left_c_contiguous == detail.left_c_contiguous &&
            entry->right_c_contiguous == detail.right_c_contiguous &&
            entry->inner_contiguous == detail.inner_contiguous &&
            entry->negative_stride == detail.negative_stride &&
            strcmp(entry->shape_signature, detail.shape_signature) == 0 &&
            strcmp(entry->stride_signature, detail.stride_signature) == 0) {
            entry->count++;
            return;
        }
    }

    NumpyJitLayoutMissDetail *entry =
        PyMem_RawMalloc(sizeof(NumpyJitLayoutMissDetail));
    if (entry == NULL) {
        return;
    }
    *entry = detail;
    entry->count = 1;
    entry->next = numpy_jit_layout_miss_details;
    numpy_jit_layout_miss_details = entry;
}

static void
numpy_jit_record_dtype_miss(NumpyJitPlanPayload *payload,
                            PyObject *left, PyObject *right)
{
    if (!numpy_jit_stats_enabled() || payload == NULL) {
        return;
    }

    NumpyJitDtypeMissDetail detail;
    memset(&detail, 0, sizeof(detail));
    detail.site_id = payload->site != NULL ? payload->site->site_id : UINT32_MAX;
    detail.op = payload->op;
    detail.entry_kind = payload->layout;
    detail.left_dtype = numpy_jit_object_dtype(left);
    detail.right_dtype = numpy_jit_object_dtype(right);
    numpy_jit_dtype_name(detail.left_dtype, detail.left_dtype_name,
                         sizeof(detail.left_dtype_name));
    numpy_jit_dtype_name(detail.right_dtype, detail.right_dtype_name,
                         sizeof(detail.right_dtype_name));
    detail.same_shape = numpy_jit_same_shape(left, right);
    detail.left_c_contiguous = numpy_jit_object_c_contiguous(left);
    detail.right_c_contiguous = numpy_jit_object_c_contiguous(right);

    for (NumpyJitDtypeMissDetail *entry = numpy_jit_dtype_miss_details;
         entry != NULL; entry = entry->next) {
        if (entry->site_id == detail.site_id &&
            entry->op == detail.op &&
            entry->entry_kind == detail.entry_kind &&
            entry->left_dtype == detail.left_dtype &&
            entry->right_dtype == detail.right_dtype &&
            entry->same_shape == detail.same_shape &&
            entry->left_c_contiguous == detail.left_c_contiguous &&
            entry->right_c_contiguous == detail.right_c_contiguous) {
            entry->count++;
            return;
        }
    }

    NumpyJitDtypeMissDetail *entry =
        PyMem_RawMalloc(sizeof(NumpyJitDtypeMissDetail));
    if (entry == NULL) {
        return;
    }
    *entry = detail;
    entry->count = 1;
    entry->next = numpy_jit_dtype_miss_details;
    numpy_jit_dtype_miss_details = entry;
}

static int
numpy_jit_stats_enabled(void)
{
    if (numpy_jit_stats_enabled_cache >= 0) {
        return numpy_jit_stats_enabled_cache;
    }
    char *env = getenv("PYTHON_JIT_EXT_STATS");
    numpy_jit_stats_enabled_cache =
        (env != NULL && env[0] != '\0' && env[0] != '0');
    return numpy_jit_stats_enabled_cache;
}

static void
numpy_jit_guard_fail(NumpyJitPlanPayload *payload,
                     NumpyJitGuardFailReason reason)
{
    if (numpy_jit_stats_enabled()) {
        numpy_jit_guard_fail_counts[reason]++;
        if (payload != NULL && payload->site != NULL &&
            reason < PYJIT_EXT_GUARD_FAIL_REASON_COUNT) {
            payload->site->guard_fail_counts[reason]++;
        }
    }
}

static void
numpy_jit_record_element_count(NumpyJitPlanPayload *payload, npy_intp size)
{
    if (payload == NULL || payload->site == NULL || size < 0) {
        return;
    }
    PyJitExtSiteState *site = payload->site;
    int64_t value = (int64_t)size;
    if (site->element_count_min == 0 || value < site->element_count_min) {
        site->element_count_min = value;
    }
    if (value > site->element_count_max) {
        site->element_count_max = value;
    }
    site->element_count_last = value;
}

static int
numpy_jit_site_disable_enabled(void)
{
    char *value = getenv("PYTHON_JIT_EXT_ENABLE_SITE_DISABLE");
    return value == NULL || value[0] != '0';
}

static int
numpy_jit_env_flag(const char *name, int *cache, int default_value)
{
    if (*cache >= 0) {
        return *cache;
    }
    char *value = Py_GETENV(name);
    *cache = value == NULL || value[0] == '\0' ? default_value : value[0] != '0';
    return *cache;
}

static int
numpy_jit_strided_same_shape_enabled(void)
{
    return numpy_jit_env_flag("NUMPY_JIT_EXT_ENABLE_STRIDED_SAME_SHAPE",
                              &numpy_jit_env_enable_strided_same_shape, 1);
}

static int
numpy_jit_static_quick_disable_enabled(void)
{
    return numpy_jit_env_flag("NUMPY_JIT_EXT_ENABLE_STATIC_QUICK_DISABLE",
                              &numpy_jit_env_enable_static_quick_disable, 1);
}

static int
numpy_jit_array_scalar_enabled(void)
{
    return numpy_jit_env_flag("NUMPY_JIT_EXT_ENABLE_ARRAY_SCALAR",
                              &numpy_jit_env_enable_array_scalar, 1);
}

static int
numpy_jit_scalar_array_enabled(void)
{
    return numpy_jit_env_flag("NUMPY_JIT_EXT_ENABLE_SCALAR_ARRAY",
                              &numpy_jit_env_enable_scalar_array, 1);
}

static void
numpy_jit_maybe_disable_static_miss(NumpyJitPlanPayload *payload,
                                    const char *reason)
{
    if (payload == NULL || payload->site == NULL ||
        !numpy_jit_static_quick_disable_enabled() ||
        !numpy_jit_site_disable_enabled()) {
        return;
    }
    PyJitExtSiteState *site = payload->site;
    if (site->disabled || site->hit_count != 0) {
        return;
    }
    if (site->decline_count != UINT16_MAX) {
        site->decline_count++;
    }
    if (site->decline_count >= 2) {
        site->disabled = 1;
        site->disable_reason = reason;
    }
}

static void
numpy_jit_dump_stats(void)
{
    if (!numpy_jit_stats_enabled()) {
        return;
    }
    const char *dir = getenv("PYTHON_JIT_EXT_STATS_DIR");
    if (dir == NULL || dir[0] == '\0') {
        dir = ".";
    }
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir,
             "crossjit_numpy_guard_summary.ndjson");
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return;
    }
    static const char *names[NUMPY_JIT_GUARD_COUNT] = {
        "not_exact_ndarray",
        "dtype_mismatch",
        "not_c_contiguous",
        "ndim_mismatch",
        "shape_mismatch",
        "min_elements",
        "not_scalar",
        "scalar_convert_failed",
        "strided_min_elements",
    };
    for (int i = 0; i < NUMPY_JIT_GUARD_COUNT; i++) {
        fprintf(fp, "{\"reason\":\"%s\",\"count\":%" PRIu64 "}\n",
                names[i], numpy_jit_guard_fail_counts[i]);
    }
    fclose(fp);

    snprintf(path, sizeof(path), "%s/%s", dir,
             "crossjit_numpy_layout_miss_detail.ndjson");
    fp = fopen(path, "w");
    if (fp != NULL) {
        for (NumpyJitLayoutMissDetail *entry = numpy_jit_layout_miss_details;
             entry != NULL; entry = entry->next) {
            fprintf(fp,
                    "{\"site_id\":%" PRIu32 ",\"op\":\"%s\","
                    "\"entry_kind\":\"%s\","
                    "\"left_dtype_num\":%d,\"left_dtype\":\"%s\","
                    "\"right_dtype_num\":%d,\"right_dtype\":\"%s\","
                    "\"left_ndim\":%d,\"right_ndim\":%d,"
                    "\"shape_signature\":\"%s\","
                    "\"stride_signature\":\"%s\","
                    "\"same_shape\":%d,"
                    "\"left_c_contiguous\":%d,"
                    "\"right_c_contiguous\":%d,"
                    "\"inner_contiguous\":%d,"
                    "\"negative_stride\":%d,"
                    "\"count\":%" PRIu64 "}\n",
                    entry->site_id,
                    numpy_jit_op_name(entry->op),
                    numpy_jit_layout_name(entry->entry_kind),
                    entry->left_dtype, entry->left_dtype_name,
                    entry->right_dtype, entry->right_dtype_name,
                    entry->left_ndim, entry->right_ndim,
                    entry->shape_signature,
                    entry->stride_signature,
                    entry->same_shape,
                    entry->left_c_contiguous,
                    entry->right_c_contiguous,
                    entry->inner_contiguous,
                    entry->negative_stride,
                    entry->count);
        }
        fclose(fp);
    }

    snprintf(path, sizeof(path), "%s/%s", dir,
             "crossjit_numpy_dtype_miss_detail.ndjson");
    fp = fopen(path, "w");
    if (fp != NULL) {
        for (NumpyJitDtypeMissDetail *entry = numpy_jit_dtype_miss_details;
             entry != NULL; entry = entry->next) {
            fprintf(fp,
                    "{\"site_id\":%" PRIu32 ",\"op\":\"%s\","
                    "\"entry_kind\":\"%s\","
                    "\"left_dtype_num\":%d,\"left_dtype\":\"%s\","
                    "\"right_dtype_num\":%d,\"right_dtype\":\"%s\","
                    "\"same_shape\":%d,"
                    "\"left_c_contiguous\":%d,"
                    "\"right_c_contiguous\":%d,"
                    "\"count\":%" PRIu64 "}\n",
                    entry->site_id,
                    numpy_jit_op_name(entry->op),
                    numpy_jit_layout_name(entry->entry_kind),
                    entry->left_dtype, entry->left_dtype_name,
                    entry->right_dtype, entry->right_dtype_name,
                    entry->same_shape,
                    entry->left_c_contiguous,
                    entry->right_c_contiguous,
                    entry->count);
        }
        fclose(fp);
    }
}

static int
numpy_jit_fused_refs_debug_enabled(void)
{
    const char *enabled = getenv("PYTHON_JIT_EXT_DEBUG_FUSED_BINARY2_REFS");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

static void
numpy_jit_log_fused_refs(PyObject *first_left, PyObject *first_right,
                         PyObject *second_other, PyObject *result,
                         const char *status)
{
    if (!numpy_jit_fused_refs_debug_enabled()) {
        return;
    }
    const char *dir = getenv("PYTHON_JIT_EXT_STATS_DIR");
    if (dir == NULL || dir[0] == '\0') {
        dir = "/tmp";
    }
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir,
             "crossjit_fused_binary2_refs.ndjson");
    FILE *fp = fopen(path, "a");
    if (fp == NULL) {
        return;
    }
    fprintf(fp,
            "{\"first_left_ptr\":\"%p\",\"first_left_refcnt\":%" PRId64 ","
            "\"first_right_ptr\":\"%p\",\"first_right_refcnt\":%" PRId64 ","
            "\"second_other_ptr\":\"%p\",\"second_other_refcnt\":%" PRId64 ","
            "\"result_ptr\":\"%p\",\"result_refcnt\":%" PRId64 ","
            "\"status\":\"%s\"}\n",
            (void *)first_left,
            first_left != NULL ? (int64_t)Py_REFCNT(first_left) : -1,
            (void *)first_right,
            first_right != NULL ? (int64_t)Py_REFCNT(first_right) : -1,
            (void *)second_other,
            second_other != NULL ? (int64_t)Py_REFCNT(second_other) : -1,
            (void *)result,
            result != NULL ? (int64_t)Py_REFCNT(result) : -1,
            status);
    fclose(fp);
}

static npy_intp
numpy_jit_get_min_elements(void)
{
    if (numpy_jit_env_min_elements >= 0) {
        return numpy_jit_env_min_elements;
    }

    npy_intp min_elements = 16;
    char *env = Py_GETENV("NUMPY_JIT_EXT_MIN_ELEMENTS");
    if (env != NULL && env[0] != '\0') {
        char *end = NULL;
        unsigned long value = strtoul(env, &end, 10);
        if (end != env) {
            min_elements = (npy_intp)value;
        }
    }
    numpy_jit_env_min_elements = min_elements;
    return min_elements;
}

static npy_intp
numpy_jit_get_strided_min_elements(void)
{
    if (numpy_jit_env_strided_min_elements >= 0) {
        return numpy_jit_env_strided_min_elements;
    }

    npy_intp min_elements = 16;
    char *env = Py_GETENV("NUMPY_JIT_EXT_STRIDED_MIN_ELEMENTS");
    if (env != NULL && env[0] != '\0') {
        char *end = NULL;
        unsigned long value = strtoul(env, &end, 10);
        if (end != env) {
            min_elements = (npy_intp)value;
        }
    }
    numpy_jit_env_strided_min_elements = min_elements;
    return min_elements;
}

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

static int
numpy_jit_stencil_site_size_ok(const NumpyJitPlanPayload *payload)
{
    if (payload == NULL || payload->site == NULL) {
        return 0;
    }
    PyJitExtSiteState *site = payload->site;
    if (site->element_count_min != site->element_count_max ||
        site->element_count_last != site->element_count_min) {
        return 0;
    }
    return site->element_count_last >= payload->min_elements;
}

static const PyJitExtStencil *
numpy_jit_get_stencil(const PyJitExtPlan *plan, int NPY_UNUSED(opcode))
{
#if NUMPY_JIT_HAS_EXT_BINARY_R23_STENCIL
    NumpyJitPlanPayload *payload = plan != NULL ?
        (NumpyJitPlanPayload *)plan->payload : NULL;
    if (plan != NULL && plan->kind == PYJIT_EXT_PLAN_BINARY &&
        payload != NULL &&
        payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE &&
        payload->left_dtype == NPY_DOUBLE &&
        payload->right_dtype == NPY_DOUBLE &&
        payload->result_dtype == NPY_DOUBLE &&
        numpy_jit_stencil_site_size_ok(payload)) {
        return &numpy_jit_ext_binary_r23_stencil;
    }
#endif
    return NULL;
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
make_numpy_plan(const NumpyJitCandidate *candidate, PyObject *callable,
                PyJitExtSiteState *site)
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
    payload->itemsize = numpy_jit_itemsize_for_dtype(candidate->result_dtype);
    payload->min_elements = numpy_jit_get_min_elements();
    if (payload->min_elements < (npy_intp)candidate->min_elements) {
        payload->min_elements = (npy_intp)candidate->min_elements;
    }
    payload->expected_callable = callable;
    Py_XINCREF(callable);
    payload->candidate = candidate;
    payload->site = site;

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
        plan->get_stencil = numpy_jit_get_stencil;
    }
    else {
        plan->eval.call_eval = candidate->call_eval;
    }
    plan->free_payload = NULL;
    plan->debug_name = "numpy.elemwise.float.c_contig";
    return plan;
}

static PyJitExtPlan *
numpy_jit_try_binary_plan(const PyJitExtTraceRequest *request,
                          PyJitExtSiteState *site)
{
    NumpyJitOp op;
    if (!resolve_binary_oparg(request->u.binary.binary_oparg, &op)) {
        return NULL;
    }

    NumpyJitLayoutKind layout;
    if (request->u.binary.left_type == &PyArray_Type &&
        request->u.binary.right_type == &PyArray_Type) {
        layout = NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE;
    }
    else if (request->u.binary.left_type == &PyArray_Type &&
             (request->u.binary.right_type == &PyFloat_Type ||
              request->u.binary.right_type == &PyLong_Type)) {
        if (!numpy_jit_array_scalar_enabled()) {
            return NULL;
        }
        layout = NUMPY_JIT_LAYOUT_C_CONTIG_ARRAY_SCALAR;
    }
    else if ((request->u.binary.left_type == &PyFloat_Type ||
              request->u.binary.left_type == &PyLong_Type) &&
             request->u.binary.right_type == &PyArray_Type) {
        if (!numpy_jit_scalar_array_enabled()) {
            return NULL;
        }
        layout = NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY;
    }
    else {
        return NULL;
    }

    const NumpyJitCandidate *candidate = find_candidate(
        op, NPY_DOUBLE, NPY_DOUBLE, layout);
    if (candidate == NULL) {
        return NULL;
    }
    return make_numpy_plan(candidate, NULL, site);
}

static PyJitExtPlan *
numpy_jit_try_call_plan(const PyJitExtTraceRequest *request,
                        PyJitExtSiteState *site)
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
    return make_numpy_plan(candidate, request->u.call.callable, site);
}

static PyJitExtPlan *
numpy_jit_try_fused_binary2_plan(const PyJitExtTraceRequest *request,
                                 PyJitExtSiteState *site)
{
    PyJitExtPlan *first = request->u.fused_binary2.first_plan;
    PyJitExtPlan *second = request->u.fused_binary2.second_plan;
    if (first == NULL || second == NULL ||
        first->provider != &numpy_jit_provider ||
        second->provider != &numpy_jit_provider ||
        first->kind != PYJIT_EXT_PLAN_BINARY ||
        second->kind != PYJIT_EXT_PLAN_BINARY) {
        return NULL;
    }
    NumpyJitPlanPayload *first_payload =
        (NumpyJitPlanPayload *)first->payload;
    NumpyJitPlanPayload *second_payload =
        (NumpyJitPlanPayload *)second->payload;
    if (first_payload == NULL || second_payload == NULL ||
        first_payload->op != NUMPY_JIT_OP_MULTIPLY ||
        second_payload->op != NUMPY_JIT_OP_ADD) {
        return NULL;
    }
    if (first_payload->layout != NUMPY_JIT_LAYOUT_C_CONTIG_ARRAY_SCALAR &&
        first_payload->layout != NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY) {
        return NULL;
    }
    if (second_payload->layout != NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE) {
        return NULL;
    }

    NumpyJitPlanPayload *payload = PyMem_Calloc(1, sizeof(*payload));
    if (payload == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    payload->op = NUMPY_JIT_OP_ADD;
    payload->layout = first_payload->layout;
    payload->left_dtype = NPY_DOUBLE;
    payload->right_dtype = NPY_DOUBLE;
    payload->result_dtype = NPY_DOUBLE;
    payload->itemsize = (int)sizeof(double);
    payload->min_elements = numpy_jit_get_min_elements();
    payload->candidate = first_payload->candidate;
    payload->site = site;
    payload->fused_op = NUMPY_JIT_FUSED_MUL_SCALAR_ADD_ARRAY;
    payload->fused_flags = request->u.fused_binary2.flags;

    PyJitExtPlan *plan = PyMem_Calloc(1, sizeof(*plan));
    if (plan == NULL) {
        PyMem_Free(payload);
        PyErr_NoMemory();
        return NULL;
    }
    plan->abi_version = PYJIT_EXT_ABI_VERSION;
    plan->struct_size = sizeof(*plan);
    plan->kind = PYJIT_EXT_PLAN_FUSED_BINARY2;
    plan->provider = &numpy_jit_provider;
    plan->flags = PYJIT_EXT_PLAN_REQUIRES_GIL |
                  PYJIT_EXT_PLAN_NO_PYTHON_CALLBACK |
                  PYJIT_EXT_PLAN_ALLOCATES_RESULT;
    plan->op_id = 1000u + (uint32_t)payload->fused_op;
    plan->payload = payload;
    plan->eval.fused_binary2_eval = numpy_fused_binary2_eval;
    plan->debug_name = "numpy.fused.binary2.mul_scalar_add";
    if (site != NULL) {
        if (site->fused_site_count != UINT16_MAX) {
            site->fused_site_count++;
        }
        if (payload->fused_flags &
            PYJIT_EXT_FUSED_BINARY2_SECOND_CONSUMES_RIGHT) {
            site->fused_pattern =
                first_payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY ?
                "a+scalar*b" : "a+b*scalar";
        }
        else {
            site->fused_pattern =
                first_payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY ?
                "scalar*a+b" : "a*scalar+b";
        }
    }
    return plan;
}

static PyJitExtPlan *
numpy_jit_try_plan(const PyJitExtTraceRequest *request, PyJitExtSiteState *site)
{
    if (site != NULL) {
        site->last_provider = &numpy_jit_provider;
    }
    switch (request->kind) {
        case PYJIT_EXT_TRACE_BINARY:
            return numpy_jit_try_binary_plan(request, site);
        case PYJIT_EXT_TRACE_CALL:
            return numpy_jit_try_call_plan(request, site);
        case PYJIT_EXT_TRACE_FUSED_BINARY2:
            return numpy_jit_try_fused_binary2_plan(request, site);
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
numpy_jit_is_supported_float_dtype(int dtype)
{
    return dtype == NPY_DOUBLE || dtype == NPY_FLOAT;
}

typedef enum {
    NUMPY_JIT_EVAL_LAYOUT_C_CONTIG,
    NUMPY_JIT_EVAL_LAYOUT_STRIDED_SAME_SHAPE,
} NumpyJitEvalLayout;

static int
numpy_jit_positive_inner_contig_strides(PyArrayObject *a, PyArrayObject *b)
{
    int ndim = PyArray_NDIM(a);
    if (ndim <= 0) {
        return 0;
    }

    npy_intp *astrides = PyArray_STRIDES(a);
    npy_intp *bstrides = PyArray_STRIDES(b);
    for (int i = 0; i < ndim; i++) {
        if (astrides[i] <= 0 || bstrides[i] <= 0) {
            return 0;
        }
    }
    return astrides[ndim - 1] == PyArray_ITEMSIZE(a) &&
           bstrides[ndim - 1] == PyArray_ITEMSIZE(b);
}

static int
same_shape_float_layout(NumpyJitPlanPayload *payload,
                        PyObject *left, PyObject *right,
                        PyArrayObject **a_out, PyArrayObject **b_out,
                        npy_intp *size_out, int *dtype_out,
                        NumpyJitEvalLayout *layout_out)
{
    if (!PyArray_CheckExact(left) || !PyArray_CheckExact(right)) {
        numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_NOT_EXACT_ARRAY);
        return 0;
    }
    PyArrayObject *a = (PyArrayObject *)left;
    PyArrayObject *b = (PyArrayObject *)right;
    int dtype = PyArray_TYPE(a);
    if (dtype != PyArray_TYPE(b) ||
        !numpy_jit_is_supported_float_dtype(dtype)) {
        numpy_jit_record_dtype_miss(payload, left, right);
        numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_DTYPE_MISMATCH);
        return 0;
    }
    int ndim = PyArray_NDIM(a);
    if (ndim != PyArray_NDIM(b)) {
        numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_NDIM_MISMATCH);
        return 0;
    }
    npy_intp size = 1;
    npy_intp *adims = PyArray_DIMS(a);
    npy_intp *bdims = PyArray_DIMS(b);
    for (int i = 0; i < ndim; i++) {
        if (adims[i] != bdims[i]) {
            numpy_jit_maybe_disable_static_miss(
                payload, "numpy_shape_mismatch");
            numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_SHAPE_MISMATCH);
            return 0;
        }
        size *= adims[i];
    }

    NumpyJitEvalLayout layout = NUMPY_JIT_EVAL_LAYOUT_C_CONTIG;
    if (!PyArray_IS_C_CONTIGUOUS(a) || !PyArray_IS_C_CONTIGUOUS(b)) {
        if (!numpy_jit_strided_same_shape_enabled() ||
            !numpy_jit_positive_inner_contig_strides(a, b)) {
            numpy_jit_record_layout_miss(payload, left, right);
            numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_NOT_C_CONTIGUOUS);
            return 0;
        }
        layout = NUMPY_JIT_EVAL_LAYOUT_STRIDED_SAME_SHAPE;
    }

    *a_out = a;
    *b_out = b;
    *size_out = size;
    *dtype_out = dtype;
    *layout_out = layout;
    return 1;
}

static int
c_contig_float_array(NumpyJitPlanPayload *payload, PyObject *array,
                     PyObject *detail_left, PyObject *detail_right,
                     PyArrayObject **array_out, npy_intp *size_out,
                     int *dtype_out)
{
    if (!PyArray_CheckExact(array)) {
        numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_NOT_EXACT_ARRAY);
        return 0;
    }
    PyArrayObject *a = (PyArrayObject *)array;
    int dtype = PyArray_TYPE(a);
    if (!numpy_jit_is_supported_float_dtype(dtype)) {
        numpy_jit_record_dtype_miss(payload, detail_left, detail_right);
        numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_DTYPE_MISMATCH);
        return 0;
    }
    if (!PyArray_IS_C_CONTIGUOUS(a)) {
        numpy_jit_record_layout_miss(payload, detail_left, detail_right);
        numpy_jit_maybe_disable_static_miss(
            payload, "numpy_scalar_array_not_c_contiguous");
        numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_NOT_C_CONTIGUOUS);
        return 0;
    }
    int ndim = PyArray_NDIM(a);
    if (ndim == 0) {
        numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_NDIM_MISMATCH);
        return 0;
    }
    npy_intp size = 1;
    npy_intp *dims = PyArray_DIMS(a);
    for (int i = 0; i < ndim; i++) {
        size *= dims[i];
    }
    *array_out = a;
    *size_out = size;
    *dtype_out = dtype;
    return 1;
}

static int
exact_python_scalar_as_double(NumpyJitPlanPayload *payload, PyObject *scalar,
                              double *value_out)
{
    if (!PyFloat_CheckExact(scalar) && !PyLong_CheckExact(scalar)) {
        numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_NOT_SCALAR);
        return 0;
    }
    double value = PyFloat_AsDouble(scalar);
    if (value == -1.0 && PyErr_Occurred()) {
        PyErr_Clear();
        numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_SCALAR_CONVERT_FAILED);
        return 0;
    }
    *value_out = value;
    return 1;
}

static PyJitExtResult
numpy_elemwise_strided_same_shape_loop(NumpyJitPlanPayload *payload,
                                       PyArrayObject *a, PyArrayObject *b,
                                       PyArrayObject *out,
                                       npy_intp size, int dtype)
{
    return numpy_jit_kernel_elemwise_strided_same_shape(
        (NumpyJitKernelOp)payload->op, dtype, PyArray_NDIM(a), size,
        PyArray_DIMS(a), PyArray_STRIDES(a), PyArray_STRIDES(b),
        PyArray_BYTES(a), PyArray_BYTES(b), PyArray_BYTES(out)) ?
        PYJIT_EXT_HIT : PYJIT_EXT_MISS;
}

static PyJitExtResult
numpy_jit_prepare_binary_contig_f64(void *payload_raw, PyObject *left,
                                    PyObject *right, void *run_state_raw)
{
    NumpyJitPlanPayload *payload = (NumpyJitPlanPayload *)payload_raw;
    NumpyJitBinaryContigF64RunState *run =
        (NumpyJitBinaryContigF64RunState *)run_state_raw;
    if (payload == NULL || run == NULL) {
        return PYJIT_EXT_MISS;
    }
    if (payload->layout != NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE ||
        payload->left_dtype != NPY_DOUBLE ||
        payload->right_dtype != NPY_DOUBLE ||
        payload->result_dtype != NPY_DOUBLE) {
        return PYJIT_EXT_MISS;
    }
    if (!PyArray_CheckExact(left) || !PyArray_CheckExact(right)) {
        numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_NOT_EXACT_ARRAY);
        return PYJIT_EXT_MISS;
    }

    PyArrayObject *a = (PyArrayObject *)left;
    PyArrayObject *b = (PyArrayObject *)right;
    if (PyArray_TYPE(a) != NPY_DOUBLE || PyArray_TYPE(b) != NPY_DOUBLE) {
        numpy_jit_record_dtype_miss(payload, left, right);
        numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_DTYPE_MISMATCH);
        return PYJIT_EXT_MISS;
    }
    int ndim = PyArray_NDIM(a);
    if (ndim == 0 || ndim != PyArray_NDIM(b)) {
        numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_NDIM_MISMATCH);
        return PYJIT_EXT_MISS;
    }

    npy_intp size = 1;
    npy_intp *adims = PyArray_DIMS(a);
    npy_intp *bdims = PyArray_DIMS(b);
    for (int i = 0; i < ndim; i++) {
        if (adims[i] != bdims[i]) {
            numpy_jit_maybe_disable_static_miss(
                payload, "numpy_shape_mismatch");
            numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_SHAPE_MISMATCH);
            return PYJIT_EXT_MISS;
        }
        size *= adims[i];
    }
    if (!PyArray_IS_C_CONTIGUOUS(a) || !PyArray_IS_C_CONTIGUOUS(b)) {
        numpy_jit_record_layout_miss(payload, left, right);
        numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_NOT_C_CONTIGUOUS);
        return PYJIT_EXT_MISS;
    }
    numpy_jit_record_element_count(payload, size);
    if (size < payload->min_elements) {
        numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_MIN_ELEMENTS);
        return PYJIT_EXT_MISS;
    }

    PyArrayObject *out = (PyArrayObject *)PyArray_EMPTY(ndim, adims,
                                                        NPY_DOUBLE, 0);
    if (out == NULL) {
        return PYJIT_EXT_ERROR;
    }

    run->left_data = (const double *)PyArray_BYTES(a);
    run->right_data = (const double *)PyArray_BYTES(b);
    run->out_data = (double *)PyArray_BYTES(out);
    run->size = size;
    run->out = (PyObject *)out;
    return PYJIT_EXT_HIT;
}

static PyJitExtResult
numpy_elemwise_eval_common(NumpyJitPlanPayload *payload,
                           PyObject *left, PyObject *right,
                           PyObject **result)
{
    PyArrayObject *a;
    PyArrayObject *b;
    npy_intp size;
    int dtype;
    NumpyJitEvalLayout layout;
    if (!same_shape_float_layout(payload, left, right, &a, &b, &size,
                                 &dtype, &layout)) {
        return PYJIT_EXT_MISS;
    }
    numpy_jit_record_element_count(payload, size);
    if (layout == NUMPY_JIT_EVAL_LAYOUT_STRIDED_SAME_SHAPE) {
        npy_intp strided_min_elements = numpy_jit_get_strided_min_elements();
        if (size < strided_min_elements) {
            numpy_jit_maybe_disable_static_miss(payload, "strided_min_elements");
            numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_STRIDED_MIN_ELEMENTS);
            return PYJIT_EXT_MISS;
        }
    }
    else if (size < payload->min_elements) {
        numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_MIN_ELEMENTS);
        return PYJIT_EXT_MISS;
    }

    PyArrayObject *out = (PyArrayObject *)PyArray_EMPTY(
        PyArray_NDIM(a), PyArray_DIMS(a), dtype, 0);
    if (out == NULL) {
        return PYJIT_EXT_ERROR;
    }

    if (layout == NUMPY_JIT_EVAL_LAYOUT_STRIDED_SAME_SHAPE) {
        PyJitExtResult strided_result =
            numpy_elemwise_strided_same_shape_loop(payload, a, b, out,
                                                   size, dtype);
        if (strided_result != PYJIT_EXT_HIT) {
            Py_DECREF(out);
            return strided_result;
        }
        *result = (PyObject *)out;
        return PYJIT_EXT_HIT;
    }

    if (!numpy_jit_kernel_elemwise_contig(
            (NumpyJitKernelOp)payload->op, dtype, PyArray_BYTES(a),
            PyArray_BYTES(b), PyArray_BYTES(out), size)) {
        Py_DECREF(out);
        return PYJIT_EXT_MISS;
    }

    *result = (PyObject *)out;
    return PYJIT_EXT_HIT;
}

static PyJitExtResult
numpy_elemwise_array_scalar_eval_common(NumpyJitPlanPayload *payload,
                                        PyObject *array, PyObject *scalar,
                                        int scalar_first,
                                        PyObject **result)
{
    PyArrayObject *a;
    npy_intp size;
    double scalar_value;
    int dtype;
    PyObject *detail_left = scalar_first ? scalar : array;
    PyObject *detail_right = scalar_first ? array : scalar;
    if (!c_contig_float_array(payload, array, detail_left, detail_right,
                              &a, &size, &dtype)) {
        return PYJIT_EXT_MISS;
    }
    numpy_jit_record_element_count(payload, size);
    if (!exact_python_scalar_as_double(payload, scalar, &scalar_value)) {
        return PYJIT_EXT_MISS;
    }
    if (size < payload->min_elements) {
        numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_MIN_ELEMENTS);
        return PYJIT_EXT_MISS;
    }

    PyArrayObject *out = (PyArrayObject *)PyArray_EMPTY(
        PyArray_NDIM(a), PyArray_DIMS(a), dtype, 0);
    if (out == NULL) {
        return PYJIT_EXT_ERROR;
    }

    if (!numpy_jit_kernel_array_scalar_contig(
            (NumpyJitKernelOp)payload->op, dtype, PyArray_BYTES(a),
            scalar_value, scalar_first, PyArray_BYTES(out), size)) {
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
numpy_elemwise_f64_array_scalar_eval(void *payload, PyObject *left,
                                     PyObject *right, PyObject **result)
{
    return numpy_elemwise_array_scalar_eval_common(
        (NumpyJitPlanPayload *)payload, left, right, 0, result);
}

static PyJitExtResult
numpy_elemwise_f64_scalar_array_eval(void *payload, PyObject *left,
                                     PyObject *right, PyObject **result)
{
    return numpy_elemwise_array_scalar_eval_common(
        (NumpyJitPlanPayload *)payload, right, left, 1, result);
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

static int
numpy_jit_fused_same_shape(PyObject *left, PyObject *right,
                           PyArrayObject **a_out, PyArrayObject **b_out,
                           npy_intp *size_out, int *dtype_out,
                           NumpyJitEvalLayout *layout_out)
{
    if (!PyArray_CheckExact(left) || !PyArray_CheckExact(right)) {
        PyErr_SetString(PyExc_TypeError, "expected exact ndarray operands");
        return 0;
    }
    PyArrayObject *a = (PyArrayObject *)left;
    PyArrayObject *b = (PyArrayObject *)right;
    int dtype = PyArray_TYPE(a);
    if (dtype != PyArray_TYPE(b) || !numpy_jit_is_supported_float_dtype(dtype)) {
        PyErr_SetString(PyExc_TypeError, "expected matching float32 or float64 arrays");
        return 0;
    }
    int ndim = PyArray_NDIM(a);
    if (ndim != PyArray_NDIM(b)) {
        PyErr_SetString(PyExc_ValueError, "expected same ndim");
        return 0;
    }
    npy_intp size = 1;
    npy_intp *adims = PyArray_DIMS(a);
    npy_intp *bdims = PyArray_DIMS(b);
    for (int i = 0; i < ndim; i++) {
        if (adims[i] != bdims[i]) {
            PyErr_SetString(PyExc_ValueError, "expected same shape");
            return 0;
        }
        size *= adims[i];
    }
    NumpyJitEvalLayout layout = NUMPY_JIT_EVAL_LAYOUT_C_CONTIG;
    if (!PyArray_IS_C_CONTIGUOUS(a) || !PyArray_IS_C_CONTIGUOUS(b)) {
        if (!numpy_jit_positive_inner_contig_strides(a, b)) {
            PyErr_SetString(PyExc_ValueError, "expected C-contiguous or positive inner-contiguous strides");
            return 0;
        }
        layout = NUMPY_JIT_EVAL_LAYOUT_STRIDED_SAME_SHAPE;
    }
    *a_out = a;
    *b_out = b;
    *size_out = size;
    *dtype_out = dtype;
    *layout_out = layout;
    return 1;
}

static PyObject *
numpy_jit_fused_eval(PyObject *a_obj, double alpha, PyObject *b_obj,
                     double beta, NumpyJitFusedOp op)
{
    PyArrayObject *a;
    PyArrayObject *b;
    npy_intp size;
    int dtype;
    NumpyJitEvalLayout layout;
    if (!numpy_jit_fused_same_shape(a_obj, b_obj, &a, &b, &size, &dtype,
                                    &layout)) {
        return NULL;
    }
    if (size < 16) {
        PyErr_SetString(PyExc_ValueError, "controlled fused prototype threshold is 16 elements");
        return NULL;
    }

    PyArrayObject *out = (PyArrayObject *)PyArray_EMPTY(
        PyArray_NDIM(a), PyArray_DIMS(a), dtype, 0);
    if (out == NULL) {
        return NULL;
    }

    if (!numpy_jit_kernel_fused_contig_or_strided(
            (int)op, dtype,
            layout == NUMPY_JIT_EVAL_LAYOUT_STRIDED_SAME_SHAPE,
            PyArray_NDIM(a), size, PyArray_DIMS(a), PyArray_STRIDES(a),
            PyArray_STRIDES(b), PyArray_BYTES(a), PyArray_BYTES(b),
            alpha, beta, PyArray_BYTES(out))) {
        Py_DECREF(out);
        PyErr_SetString(PyExc_ValueError, "unsupported fused JIT kernel");
        return NULL;
    }
    return (PyObject *)out;
}

static PyJitExtResult
numpy_fused_binary2_eval(void *payload_obj, PyObject *first_left,
                         PyObject *first_right, PyObject *second_other,
                         PyObject **result)
{
    NumpyJitPlanPayload *payload = (NumpyJitPlanPayload *)payload_obj;
    PyObject *array = first_left;
    PyObject *scalar = first_right;
    if (payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY) {
        array = first_right;
        scalar = first_left;
    }

    double scalar_value;
    if (!exact_python_scalar_as_double(payload, scalar, &scalar_value)) {
        numpy_jit_log_fused_refs(first_left, first_right, second_other,
                                 NULL, "MISS");
        return PYJIT_EXT_MISS;
    }

    PyObject *out = numpy_jit_fused_eval(
        array, scalar_value, second_other, 0.0,
        (NumpyJitFusedOp)payload->fused_op);
    if (out == NULL) {
        if (PyErr_ExceptionMatches(PyExc_TypeError) ||
            PyErr_ExceptionMatches(PyExc_ValueError)) {
            PyErr_Clear();
            numpy_jit_log_fused_refs(first_left, first_right, second_other,
                                     NULL, "MISS");
            return PYJIT_EXT_MISS;
        }
        numpy_jit_log_fused_refs(first_left, first_right, second_other,
                                 NULL, "ERROR");
        return PYJIT_EXT_ERROR;
    }
    *result = out;
    numpy_jit_log_fused_refs(first_left, first_right, second_other,
                             out, "HIT");
    return PYJIT_EXT_HIT;
}

static PyObject *
numpy_jit_fused_mul_scalar_add_array(PyObject *NPY_UNUSED(self), PyObject *args)
{
    PyObject *a;
    PyObject *b;
    double scalar;
    if (!PyArg_ParseTuple(args, "OdO:_crossjit_fused_mul_scalar_add_array",
                          &a, &scalar, &b)) {
        return NULL;
    }
    return numpy_jit_fused_eval(
        a, scalar, b, 0.0, NUMPY_JIT_FUSED_MUL_SCALAR_ADD_ARRAY);
}

static PyObject *
numpy_jit_fused_add_array_mul_scalar(PyObject *NPY_UNUSED(self), PyObject *args)
{
    PyObject *a;
    PyObject *b;
    double scalar;
    if (!PyArg_ParseTuple(args, "OOd:_crossjit_fused_add_array_mul_scalar",
                          &a, &b, &scalar)) {
        return NULL;
    }
    return numpy_jit_fused_eval(
        a, scalar, b, 0.0, NUMPY_JIT_FUSED_ADD_ARRAY_MUL_SCALAR);
}

static PyObject *
numpy_jit_fused_axpby(PyObject *NPY_UNUSED(self), PyObject *args)
{
    PyObject *a;
    PyObject *b;
    double alpha;
    double beta;
    if (!PyArg_ParseTuple(args, "dOdO:_crossjit_fused_axpby",
                          &alpha, &a, &beta, &b)) {
        return NULL;
    }
    return numpy_jit_fused_eval(a, alpha, b, beta, NUMPY_JIT_FUSED_AXPBY);
}

static PyMethodDef numpy_jit_fused_methods[] = {
    {
        "_crossjit_fused_mul_scalar_add_array",
        numpy_jit_fused_mul_scalar_add_array,
        METH_VARARGS,
        "Controlled CrossJIT prototype: out = a * scalar + b.",
    },
    {
        "_crossjit_fused_add_array_mul_scalar",
        numpy_jit_fused_add_array_mul_scalar,
        METH_VARARGS,
        "Controlled CrossJIT prototype: out = (a + b) * scalar.",
    },
    {
        "_crossjit_fused_axpby",
        numpy_jit_fused_axpby,
        METH_VARARGS,
        "Controlled CrossJIT prototype: out = alpha * a + beta * b.",
    },
    {NULL, NULL, 0, NULL},
};

NPY_NO_EXPORT int
jit_adapter_init(PyObject *umath_module)
{
    if (!numpy_jit_atexit_registered) {
        if (Py_AtExit(numpy_jit_dump_stats) == 0) {
            numpy_jit_atexit_registered = 1;
        }
    }
    if (PyModule_AddFunctions(umath_module, numpy_jit_fused_methods) < 0) {
        return -1;
    }

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
        if (oparg >= 0) {
            PyTypeObject *left_type = &PyArray_Type;
            PyTypeObject *right_type = &PyArray_Type;
            switch (numpy_jit_candidates[i].layout) {
                case NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE:
                    left_type = &PyArray_Type;
                    right_type = &PyArray_Type;
                    if (PyJit_RegisterBinaryProvider(
                            left_type, right_type, oparg,
                            &numpy_jit_provider) < 0) {
                        PyErr_Clear();
                    }
                    break;
                case NUMPY_JIT_LAYOUT_C_CONTIG_ARRAY_SCALAR:
                    if (PyJit_RegisterBinaryProvider(
                            &PyArray_Type, &PyFloat_Type, oparg,
                            &numpy_jit_provider) < 0) {
                        PyErr_Clear();
                    }
                    if (PyJit_RegisterBinaryProvider(
                            &PyArray_Type, &PyLong_Type, oparg,
                            &numpy_jit_provider) < 0) {
                        PyErr_Clear();
                    }
                    break;
                case NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY:
                    if (PyJit_RegisterBinaryProvider(
                            &PyFloat_Type, &PyArray_Type, oparg,
                            &numpy_jit_provider) < 0) {
                        PyErr_Clear();
                    }
                    if (PyJit_RegisterBinaryProvider(
                            &PyLong_Type, &PyArray_Type, oparg,
                            &numpy_jit_provider) < 0) {
                        PyErr_Clear();
                    }
                    break;
            }
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
