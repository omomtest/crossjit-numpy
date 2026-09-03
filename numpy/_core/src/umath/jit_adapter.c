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
#include "numpy/arrayscalars.h"
#include "numpy/npy_math.h"
#include "numpy/ufuncobject.h"
#include "extobj.h"
#include "mem_overlap.h"
#include "jit_kernel.h"

#ifndef PYJIT_EXT_ABI_VERSION
#define PYJIT_EXT_ABI_VERSION 5u
#define PYJIT_EXT_REGION_MAX_NODES 4u
#define PYJIT_EXT_REGION_MAX_INPUTS 6u
#define PYJIT_EXT_REGION_MAX_NODE_INPUTS 3u
#define PYJIT_EXT_REGION_PIC_SIZE 4u
#define PYJIT_EXT_UNARY_NEGATIVE 1

typedef enum {
    PYJIT_EXT_COLD = 2,
    PYJIT_EXT_HIT = 1,
    PYJIT_EXT_MISS = 0,
    PYJIT_EXT_ERROR = -1,
} PyJitExtResult;

typedef enum {
    PYJIT_EXT_TRACE_BINARY,
    PYJIT_EXT_TRACE_CALL,
    PYJIT_EXT_TRACE_FUSED_BINARY2,
    PYJIT_EXT_TRACE_REGION,
    PYJIT_EXT_TRACE_UNARY,
} PyJitExtTraceKind;

typedef struct _PyJitExtProvider PyJitExtProvider;
typedef struct _PyJitExtPlan PyJitExtPlan;
typedef struct _PyJitExtSiteState PyJitExtSiteState;
typedef struct _PyJitExtLoweredRegion PyJitExtLoweredRegion;
typedef uint16_t PyJitExtValueId;

typedef struct {
    PyJitExtValueId value_id;
    PyTypeObject *type;
    uint32_t flags;
} PyJitExtRegionInput;

typedef struct {
    const PyJitExtPlan *atomic_plan;
    uint32_t semantic_id;
    uint32_t effect_flags;
    PyJitExtValueId inputs[PYJIT_EXT_REGION_MAX_NODE_INPUTS];
    uint8_t input_count;
    PyJitExtValueId output;
    uint32_t bytecode_target;
} PyJitExtRegionNode;

typedef struct {
    PyJitExtProvider *provider;
    const PyJitExtRegionInput *inputs;
    uint8_t input_count;
    const PyJitExtRegionNode *nodes;
    uint8_t node_count;
    PyJitExtValueId output;
    uint32_t first_target;
    uint32_t last_target;
    uint64_t graph_hash;
    uint32_t flags;
} PyJitExtRegionRequest;

typedef PyJitExtResult (*PyJitExtRegionEvalFunc)(
    void *payload, PyObject *const *args, Py_ssize_t nargs,
    PyObject **result);

struct _PyJitExtLoweredRegion {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t property_signature;
    uint32_t flags;
    void *payload;
    PyJitExtRegionEvalFunc guard_eval;
    const char *debug_name;
};

typedef struct {
    PyJitExtTraceKind kind;
    union {
        struct {
            int binary_oparg;
            PyTypeObject *left_type;
            PyTypeObject *right_type;
        } binary;
        struct {
            int unary_opcode;
            PyTypeObject *operand_type;
        } unary;
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
        PyJitExtRegionRequest region;
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
    PyJitExtPlan *(*prepare_region)(
        const PyJitExtRegionRequest *request,
        PyJitExtSiteState *site);
    PyJitExtResult (*region_property_signature)(
        const PyJitExtPlan *region, PyObject *const *args,
        Py_ssize_t nargs, uint64_t *signature);
    PyJitExtLoweredRegion *(*specialize_region)(
        const PyJitExtPlan *region, PyObject *const *args,
        Py_ssize_t nargs, uint64_t signature);
    void (*free_lowered_region)(PyJitExtLoweredRegion *lowered);
};

typedef enum {
    PYJIT_EXT_PLAN_BINARY,
    PYJIT_EXT_PLAN_CALL,
    PYJIT_EXT_PLAN_FUSED_BINARY2,
    PYJIT_EXT_PLAN_REGION,
    PYJIT_EXT_PLAN_UNARY,
} PyJitExtPlanKind;

typedef PyJitExtResult (*PyJitExtBinaryEvalFunc)(
    void *payload, PyObject *left, PyObject *right, PyObject **result);
typedef PyJitExtResult (*PyJitExtUnaryEvalFunc)(
    void *payload, PyObject *operand, PyObject **result);
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
    PYJIT_EXT_STENCIL_VALUE_REGION_EVAL,
    PYJIT_EXT_STENCIL_VALUE_REGION_FALLBACK_EVAL,
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
        PyJitExtRegionEvalFunc region_eval;
        PyJitExtUnaryEvalFunc unary_eval;
    } eval;
    void (*free_payload)(void *payload);
    const char *debug_name;
    PyJitExtSiteState *site;
    const PyJitExtStencil *stencil;
    PyJitExtGetStencilFunc get_stencil;
    PyJitExtRegionEvalFunc region_fallback_eval;
    uint32_t effect_flags;
    int8_t mutation_input;
    uint8_t region_input_count;
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
    uint16_t region_site_count;
    uint16_t region_hit_count;
    uint16_t region_fallback_count;
    uint16_t region_error_count;
    uint8_t region_node_count;
    uint8_t region_input_count;
    uint64_t region_graph_hash;
    uint8_t region_disabled;
    const char *region_disable_reason;
    uint16_t region_pic_hit_count;
    uint16_t region_pic_miss_count;
    uint16_t region_specialize_count;
    uint16_t region_specialize_decline_count;
    uint16_t region_negative_count;
    uint16_t region_saturated_count;
    const char *region_lowering_name;
    const char *region_guard_fail_reason;
};

#define PYJIT_EXT_PLAN_REQUIRES_GIL          (1u << 0)
#define PYJIT_EXT_PLAN_NO_PYTHON_CALLBACK    (1u << 2)
#define PYJIT_EXT_PLAN_REGION_ANCHOR_ONLY    (1u << 7)
#define PYJIT_EXT_PLAN_ALLOCATES_RESULT      (1u << 3)
#define PYJIT_CALL_HAS_KWARGS                (1u << 0)
#define PYJIT_EXT_EFFECT_PURE                (1u << 0)
#define PYJIT_EXT_EFFECT_NO_PYTHON_CALLBACK  (1u << 1)
#define PYJIT_EXT_EFFECT_MUTATES_INPUT       (1u << 2)
#define PYJIT_EXT_EFFECT_FAST_MATH           (1u << 3)
#define PYJIT_EXT_EFFECT_PRESERVES_FP_POLICY (1u << 4)
#define PYJIT_EXT_EFFECT_OWNS_ATOMIC_PLANS   (1u << 5)

PyAPI_FUNC(int) PyJit_RegisterBinaryProvider(
    PyTypeObject *left_type,
    PyTypeObject *right_type,
    int binary_oparg,
    PyJitExtProvider *provider);
PyAPI_FUNC(int) PyJit_RegisterUnaryProvider(
    PyTypeObject *operand_type, int unary_opcode,
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
    NUMPY_JIT_OP_POWER,
    NUMPY_JIT_OP_NEGATIVE,
    NUMPY_JIT_OP_MATMUL,
} NumpyJitOp;

typedef enum {
    NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE,
    NUMPY_JIT_LAYOUT_C_CONTIG_ARRAY_SCALAR,
    NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY,
} NumpyJitLayoutKind;

typedef enum {
    NUMPY_JIT_FUSED_MUL_SCALAR_ADD_ARRAY,
    NUMPY_JIT_FUSED_MUL_SCALAR_SUB_ARRAY,
    NUMPY_JIT_FUSED_ARRAY_SUB_MUL_SCALAR,
    NUMPY_JIT_FUSED_ADD_ARRAY_MUL_SCALAR,
    NUMPY_JIT_FUSED_SUB_ARRAY_MUL_SCALAR,
    NUMPY_JIT_FUSED_ADD_ARRAY_DIV_SCALAR,
    NUMPY_JIT_FUSED_SUB_ARRAY_DIV_SCALAR,
    NUMPY_JIT_FUSED_MUL_TWO_SCALARS,
    NUMPY_JIT_FUSED_AXPBY,
    NUMPY_JIT_FUSED_ADD_THREE_ARRAYS,
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
    PyArrayObject *result;
    uint64_t bytes;
    uint64_t hit;
    uint64_t miss;
    uint64_t refcnt_miss;
    uint64_t shape_miss;
    int disabled;
    const char *disable_reason;
} NumpyJitResultCache;

typedef struct {
    uint64_t signature;
    PyJitExtLoweredRegion *lowered;
    uint8_t negative;
    uint8_t unsupported_layout_mask;
} NumpyJitRegionPICEntry;

typedef struct NumpyJitPlanPayload {
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
    int inplace;
    uint32_t fused_flags;
    const char *fused_pattern;
    NumpyJitOp region_first_op;
    NumpyJitOp region_second_op;
    int region_second_consumes_right;
    int region_runtime_other_first;
    uint8_t region_generic;
    uint8_t region_matmul_epilogue;
    uint8_t region_node_count;
    uint8_t region_input_count;
    uint8_t region_output_slot;
    int8_t region_mutation_input;
    PyJitExtValueId region_input_ids[PYJIT_EXT_REGION_MAX_INPUTS];
    NumpyJitOp region_ops[PYJIT_EXT_REGION_MAX_NODES];
    uint8_t region_left[PYJIT_EXT_REGION_MAX_NODES];
    uint8_t region_right[PYJIT_EXT_REGION_MAX_NODES];
    PyJitExtPlan *region_atomic_plans[PYJIT_EXT_REGION_MAX_NODES];
    NumpyJitKernelRegionNode region_nodes[PYJIT_EXT_REGION_MAX_NODES];
    PyJitExtPlan *region_owner_plan;
    uint32_t region_execution_count;
    uint8_t region_pic_count;
    uint8_t region_pic_saturated;
    NumpyJitRegionPICEntry region_pic[PYJIT_EXT_REGION_PIC_SIZE];
    NumpyJitResultCache result_cache;
    struct NumpyJitPlanPayload *stats_next;
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
static PyJitExtPlan *numpy_jit_prepare_region(
    const PyJitExtRegionRequest *request, PyJitExtSiteState *site);
static PyJitExtResult numpy_jit_region_property_signature(
    const PyJitExtPlan *region, PyObject *const *args,
    Py_ssize_t nargs, uint64_t *signature);
static PyJitExtLoweredRegion *numpy_jit_specialize_region(
    const PyJitExtPlan *region, PyObject *const *args,
    Py_ssize_t nargs, uint64_t signature);
static void numpy_jit_free_lowered_region(PyJitExtLoweredRegion *lowered);
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
    .prepare_region = numpy_jit_prepare_region,
    .region_property_signature = numpy_jit_region_property_signature,
    .specialize_region = numpy_jit_specialize_region,
    .free_lowered_region = numpy_jit_free_lowered_region,
};

static NumpyJitOpDef numpy_jit_ops[] = {
    { NUMPY_JIT_OP_ADD, NB_ADD, NULL, 2, "add" },
    { NUMPY_JIT_OP_SUBTRACT, NB_SUBTRACT, NULL, 2, "subtract" },
    { NUMPY_JIT_OP_MULTIPLY, NB_MULTIPLY, NULL, 2, "multiply" },
    { NUMPY_JIT_OP_TRUE_DIVIDE, NB_TRUE_DIVIDE, NULL, 2, "true_divide" },
    { NUMPY_JIT_OP_POWER, NB_POWER, NULL, 2, "power" },
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

static int
numpy_jit_is_exact_power2_exponent(PyObject *value)
{
    if (PyLong_CheckExact(value)) {
        int overflow = 0;
        long exponent = PyLong_AsLongAndOverflow(value, &overflow);
        if (exponent == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            return 0;
        }
        return overflow == 0 && exponent == 2;
    }
    return 0;
}

static PyJitExtResult numpy_elemwise_f64_binary_eval(
    void *payload, PyObject *left, PyObject *right, PyObject **result);
static PyJitExtResult numpy_elemwise_f64_array_scalar_eval(
    void *payload, PyObject *left, PyObject *right, PyObject **result);
static PyJitExtResult numpy_elemwise_f64_scalar_array_eval(
    void *payload, PyObject *left, PyObject *right, PyObject **result);
static PyJitExtResult numpy_power2_array_scalar_eval(
    void *payload, PyObject *left, PyObject *right, PyObject **result);
static PyJitExtResult numpy_matmul_binary_eval(
    void *payload, PyObject *left, PyObject *right, PyObject **result);
static PyJitExtResult numpy_negative_unary_eval(
    void *payload, PyObject *operand, PyObject **result);
static PyJitExtResult numpy_ufunc2_f64_call_eval(
    void *payload, PyObject *callable, PyObject *const *args,
    Py_ssize_t nargs, PyObject **result);
static PyJitExtResult numpy_fused_binary2_eval(
    void *payload, PyObject *first_left, PyObject *first_right,
    PyObject *second_other, PyObject **result);
static PyJitExtResult numpy_region_eval(
    void *payload, PyObject *const *args, Py_ssize_t nargs,
    PyObject **result);
static PyJitExtResult numpy_region_fallback_eval(
    void *payload, PyObject *const *args, Py_ssize_t nargs,
    PyObject **result);
static int numpy_jit_region_layout_class(PyArrayObject *array);
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
        NUMPY_JIT_OP_MATMUL,
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
        NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE,
        0, 0,
        numpy_matmul_binary_eval,
        NULL,
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
        NUMPY_JIT_OP_POWER,
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
        NUMPY_JIT_LAYOUT_C_CONTIG_ARRAY_SCALAR,
        0, 0,
        numpy_power2_array_scalar_eval,
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
static npy_intp numpy_jit_env_strided_inner_scalar_max_elements = -1;
static int numpy_jit_env_enable_strided_same_shape = -1;
static int numpy_jit_env_enable_strided_1d_same_shape = -1;
static int numpy_jit_env_enable_static_quick_disable = -1;
static int numpy_jit_env_enable_array_scalar = -1;
static int numpy_jit_env_enable_scalar_array = -1;
static int numpy_jit_env_enable_strided_1d_scalar = -1;
static int numpy_jit_env_enable_strided_inner_scalar = -1;
static int numpy_jit_env_enable_result_cache = -1;
static uint64_t numpy_jit_env_result_cache_max_bytes = UINT64_MAX;
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

typedef struct NumpyJitResultCacheStats {
    uint32_t site_id;
    NumpyJitOp op;
    NumpyJitLayoutKind layout;
    int result_dtype;
    uint64_t hit;
    uint64_t miss;
    uint64_t refcnt_miss;
    uint64_t shape_miss;
    uint64_t cached_bytes;
    int disabled;
    const char *disable_reason;
    struct NumpyJitResultCacheStats *next;
} NumpyJitResultCacheStats;

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
static NumpyJitResultCacheStats *numpy_jit_result_cache_stats = NULL;
static NumpyJitPlanPayload *numpy_jit_plan_stats_list = NULL;

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
        case NUMPY_JIT_FUSED_MUL_SCALAR_SUB_ARRAY:
            return "mul_scalar_sub_array";
        case NUMPY_JIT_FUSED_ARRAY_SUB_MUL_SCALAR:
            return "array_sub_mul_scalar";
        case NUMPY_JIT_FUSED_ADD_ARRAY_MUL_SCALAR:
            return "add_array_mul_scalar";
        case NUMPY_JIT_FUSED_SUB_ARRAY_MUL_SCALAR:
            return "sub_array_mul_scalar";
        case NUMPY_JIT_FUSED_ADD_ARRAY_DIV_SCALAR:
            return "add_array_div_scalar";
        case NUMPY_JIT_FUSED_SUB_ARRAY_DIV_SCALAR:
            return "sub_array_div_scalar";
        case NUMPY_JIT_FUSED_MUL_TWO_SCALARS:
            return "mul_two_scalars";
        case NUMPY_JIT_FUSED_AXPBY:
            return "axpby";
        case NUMPY_JIT_FUSED_ADD_THREE_ARRAYS:
            return "add_three_arrays";
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
        info->fused_pattern = payload->fused_pattern;
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
    if (!numpy_jit_stats_enabled() || payload == NULL ||
        payload->site == NULL || size < 0) {
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
numpy_jit_strided_1d_same_shape_enabled(void)
{
    return numpy_jit_env_flag("NUMPY_JIT_EXT_ENABLE_STRIDED_1D_SAME_SHAPE",
                              &numpy_jit_env_enable_strided_1d_same_shape, 1);
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

static int
numpy_jit_strided_1d_scalar_enabled(void)
{
    return numpy_jit_env_flag("NUMPY_JIT_EXT_ENABLE_STRIDED_1D_SCALAR",
                              &numpy_jit_env_enable_strided_1d_scalar, 1);
}

static int
numpy_jit_strided_inner_scalar_enabled(void)
{
    return numpy_jit_env_flag("NUMPY_JIT_EXT_ENABLE_STRIDED_INNER_SCALAR",
                              &numpy_jit_env_enable_strided_inner_scalar, 1);
}

static int
numpy_jit_result_cache_enabled(void)
{
    return numpy_jit_env_flag("NUMPY_JIT_EXT_ENABLE_RESULT_CACHE",
                              &numpy_jit_env_enable_result_cache, 1);
}

static uint64_t
numpy_jit_result_cache_max_bytes(void)
{
    if (numpy_jit_env_result_cache_max_bytes != UINT64_MAX) {
        return numpy_jit_env_result_cache_max_bytes;
    }
    uint64_t max_bytes = 256ull * 1024ull;
    char *env = Py_GETENV("NUMPY_JIT_EXT_RESULT_CACHE_MAX_BYTES");
    if (env != NULL && env[0] != '\0') {
        char *end = NULL;
        unsigned long long value = strtoull(env, &end, 10);
        if (end != env) {
            max_bytes = (uint64_t)value;
        }
    }
    numpy_jit_env_result_cache_max_bytes = max_bytes;
    return max_bytes;
}

static void
numpy_jit_link_plan_stats(NumpyJitPlanPayload *payload)
{
    payload->stats_next = numpy_jit_plan_stats_list;
    numpy_jit_plan_stats_list = payload;
}

static void
numpy_jit_unlink_plan_stats(NumpyJitPlanPayload *payload)
{
    NumpyJitPlanPayload **cursor = &numpy_jit_plan_stats_list;
    while (*cursor != NULL) {
        if (*cursor == payload) {
            *cursor = payload->stats_next;
            payload->stats_next = NULL;
            return;
        }
        cursor = &(*cursor)->stats_next;
    }
}

static void
numpy_jit_record_result_cache_stats(NumpyJitPlanPayload *payload)
{
    if (payload == NULL) {
        return;
    }
    NumpyJitResultCache *cache = &payload->result_cache;
    if (cache->hit == 0 && cache->miss == 0 && cache->refcnt_miss == 0 &&
        cache->shape_miss == 0 && !cache->disabled && cache->bytes == 0) {
        return;
    }

    uint32_t site_id = payload->site != NULL ?
        payload->site->site_id : UINT32_MAX;
    for (NumpyJitResultCacheStats *entry = numpy_jit_result_cache_stats;
         entry != NULL; entry = entry->next) {
        if (entry->site_id == site_id &&
            entry->op == payload->op &&
            entry->layout == payload->layout &&
            entry->result_dtype == payload->result_dtype) {
            entry->hit += cache->hit;
            entry->miss += cache->miss;
            entry->refcnt_miss += cache->refcnt_miss;
            entry->shape_miss += cache->shape_miss;
            if (cache->bytes > entry->cached_bytes) {
                entry->cached_bytes = cache->bytes;
            }
            if (cache->disabled) {
                entry->disabled = 1;
                entry->disable_reason = cache->disable_reason;
            }
            return;
        }
    }

    NumpyJitResultCacheStats *entry =
        PyMem_RawMalloc(sizeof(NumpyJitResultCacheStats));
    if (entry == NULL) {
        return;
    }
    memset(entry, 0, sizeof(*entry));
    entry->site_id = site_id;
    entry->op = payload->op;
    entry->layout = payload->layout;
    entry->result_dtype = payload->result_dtype;
    entry->hit = cache->hit;
    entry->miss = cache->miss;
    entry->refcnt_miss = cache->refcnt_miss;
    entry->shape_miss = cache->shape_miss;
    entry->cached_bytes = cache->bytes;
    entry->disabled = cache->disabled;
    entry->disable_reason = cache->disable_reason;
    entry->next = numpy_jit_result_cache_stats;
    numpy_jit_result_cache_stats = entry;
}

static int
numpy_jit_result_cache_shape_matches(PyArrayObject *cached, int ndim,
                                     npy_intp *dims, int dtype)
{
    if (PyArray_TYPE(cached) != dtype || PyArray_NDIM(cached) != ndim) {
        return 0;
    }
    npy_intp *cached_dims = PyArray_DIMS(cached);
    for (int i = 0; i < ndim; i++) {
        if (cached_dims[i] != dims[i]) {
            return 0;
        }
    }
    return 1;
}

static void
numpy_jit_result_cache_maybe_disable(NumpyJitPlanPayload *payload)
{
    NumpyJitResultCache *cache = &payload->result_cache;
    if (!cache->disabled && cache->miss >= 8 && cache->hit < 4) {
        cache->disabled = 1;
        cache->disable_reason = "low_hit_rate";
        Py_CLEAR(cache->result);
        cache->bytes = 0;
    }
}

static PyArrayObject *
numpy_jit_get_or_alloc_result(NumpyJitPlanPayload *payload, int ndim,
                              npy_intp *dims, int dtype, npy_intp size)
{
    if (payload == NULL || !numpy_jit_result_cache_enabled() ||
        payload->result_cache.disabled || size < 0) {
        return (PyArrayObject *)PyArray_EMPTY(ndim, dims, dtype, 0);
    }

    if (payload->itemsize <= 0 ||
        (uint64_t)size > UINT64_MAX / (uint64_t)payload->itemsize) {
        return (PyArrayObject *)PyArray_EMPTY(ndim, dims, dtype, 0);
    }
    uint64_t bytes = (uint64_t)size * (uint64_t)payload->itemsize;
    if (bytes > numpy_jit_result_cache_max_bytes()) {
        return (PyArrayObject *)PyArray_EMPTY(ndim, dims, dtype, 0);
    }

    NumpyJitResultCache *cache = &payload->result_cache;
    if (cache->result != NULL) {
        if (Py_REFCNT(cache->result) != 1) {
            cache->miss++;
            cache->refcnt_miss++;
            Py_CLEAR(cache->result);
            cache->bytes = 0;
            numpy_jit_result_cache_maybe_disable(payload);
            if (cache->disabled) {
                return (PyArrayObject *)PyArray_EMPTY(ndim, dims, dtype, 0);
            }
        }
        else if (numpy_jit_result_cache_shape_matches(
                cache->result, ndim, dims, dtype)) {
            cache->hit++;
            Py_INCREF(cache->result);
            return cache->result;
        }
        else {
            cache->miss++;
            cache->shape_miss++;
            Py_CLEAR(cache->result);
            cache->bytes = 0;
            numpy_jit_result_cache_maybe_disable(payload);
            if (cache->disabled) {
                return (PyArrayObject *)PyArray_EMPTY(ndim, dims, dtype, 0);
            }
        }
    }
    else {
        cache->miss++;
        numpy_jit_result_cache_maybe_disable(payload);
        if (cache->disabled) {
            return (PyArrayObject *)PyArray_EMPTY(ndim, dims, dtype, 0);
        }
    }

    PyArrayObject *out = (PyArrayObject *)PyArray_EMPTY(ndim, dims, dtype, 0);
    if (out == NULL) {
        return NULL;
    }
    cache->result = out;
    cache->bytes = bytes;
    Py_INCREF(out);
    return out;
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
numpy_jit_dump_result_cache_row(FILE *fp, uint32_t site_id, NumpyJitOp op,
                                NumpyJitLayoutKind layout, int result_dtype,
                                NumpyJitResultCache *cache,
                                uint64_t cached_bytes)
{
    uint64_t total = cache->hit + cache->miss;
    double hit_rate = total == 0 ? 0.0 : (double)cache->hit / (double)total;
    fprintf(fp,
            "{\"site_id\":%" PRIu32 ",\"op\":\"%s\","
            "\"layout\":\"%s\","
            "\"result_dtype\":\"%s\","
            "\"hit\":%" PRIu64 ","
            "\"miss\":%" PRIu64 ","
            "\"refcnt_miss\":%" PRIu64 ","
            "\"shape_miss\":%" PRIu64 ","
            "\"hit_rate\":%.6f,"
            "\"disabled\":%d,"
            "\"disable_reason\":\"%s\","
            "\"cached_bytes\":%" PRIu64 "}\n",
            site_id,
            numpy_jit_op_name(op),
            numpy_jit_layout_name(layout),
            numpy_jit_dtype_static_name(result_dtype),
            cache->hit,
            cache->miss,
            cache->refcnt_miss,
            cache->shape_miss,
            hit_rate,
            cache->disabled,
            cache->disable_reason != NULL ? cache->disable_reason : "",
            cached_bytes);
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

    snprintf(path, sizeof(path), "%s/%s", dir,
             "crossjit_numpy_result_cache.ndjson");
    fp = fopen(path, "w");
    if (fp != NULL) {
        for (NumpyJitResultCacheStats *entry = numpy_jit_result_cache_stats;
             entry != NULL; entry = entry->next) {
            NumpyJitResultCache cache;
            memset(&cache, 0, sizeof(cache));
            cache.hit = entry->hit;
            cache.miss = entry->miss;
            cache.refcnt_miss = entry->refcnt_miss;
            cache.shape_miss = entry->shape_miss;
            cache.disabled = entry->disabled;
            cache.disable_reason = entry->disable_reason;
            numpy_jit_dump_result_cache_row(
                fp, entry->site_id, entry->op, entry->layout,
                entry->result_dtype, &cache, entry->cached_bytes);
        }
        for (NumpyJitPlanPayload *payload = numpy_jit_plan_stats_list;
             payload != NULL; payload = payload->stats_next) {
            NumpyJitResultCache *cache = &payload->result_cache;
            if (cache->hit == 0 && cache->miss == 0 &&
                cache->refcnt_miss == 0 && cache->shape_miss == 0 &&
                !cache->disabled && cache->bytes == 0) {
                continue;
            }
            numpy_jit_dump_result_cache_row(
                fp,
                payload->site != NULL ? payload->site->site_id : UINT32_MAX,
                payload->op, payload->layout, payload->result_dtype,
                cache, cache->bytes);
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

static npy_intp
numpy_jit_get_strided_inner_scalar_max_elements(void)
{
    if (numpy_jit_env_strided_inner_scalar_max_elements >= 0) {
        return numpy_jit_env_strided_inner_scalar_max_elements;
    }

    npy_intp max_elements = 32768;
    char *env = Py_GETENV(
        "NUMPY_JIT_EXT_STRIDED_INNER_SCALAR_MAX_ELEMENTS");
    if (env != NULL && env[0] != '\0') {
        char *end = NULL;
        unsigned long value = strtoul(env, &end, 10);
        if (end != env) {
            max_elements = (npy_intp)value;
        }
    }
    numpy_jit_env_strided_inner_scalar_max_elements = max_elements;
    return max_elements;
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
    switch (binary_oparg) {
        case NB_INPLACE_ADD:
            *op = NUMPY_JIT_OP_ADD;
            return 1;
        case NB_INPLACE_SUBTRACT:
            *op = NUMPY_JIT_OP_SUBTRACT;
            return 1;
        case NB_INPLACE_MULTIPLY:
            *op = NUMPY_JIT_OP_MULTIPLY;
            return 1;
        case NB_INPLACE_TRUE_DIVIDE:
            *op = NUMPY_JIT_OP_TRUE_DIVIDE;
            return 1;
        default:
            return 0;
    }
}

static int
numpy_jit_binary_oparg_is_inplace(int binary_oparg)
{
    return binary_oparg == NB_INPLACE_ADD ||
           binary_oparg == NB_INPLACE_SUBTRACT ||
           binary_oparg == NB_INPLACE_MULTIPLY ||
           binary_oparg == NB_INPLACE_TRUE_DIVIDE;
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

static PyObject *
numpy_jit_known_ufunc(NumpyJitOp op)
{
    for (Py_ssize_t i = 0; i < Py_ARRAY_LENGTH(numpy_jit_ops); i++) {
        if (numpy_jit_ops[i].op == op) {
            return numpy_jit_ops[i].known_callable;
        }
    }
    return NULL;
}

static const PyJitExtStencil *
numpy_jit_get_stencil(const PyJitExtPlan *plan, int NPY_UNUSED(opcode))
{
#if NUMPY_JIT_HAS_EXT_BINARY_R23_STENCIL
    NumpyJitPlanPayload *payload = plan != NULL ?
        (NumpyJitPlanPayload *)plan->payload : NULL;
    if (plan != NULL && plan->kind == PYJIT_EXT_PLAN_BINARY &&
        payload != NULL &&
        !payload->inplace &&
        payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE &&
        payload->left_dtype == NPY_DOUBLE &&
        payload->right_dtype == NPY_DOUBLE &&
        payload->result_dtype == NPY_DOUBLE) {
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
    numpy_jit_link_plan_stats(payload);

    PyJitExtPlan *plan = PyMem_Calloc(1, sizeof(*plan));
    if (plan == NULL) {
        numpy_jit_unlink_plan_stats(payload);
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
    plan->effect_flags = PYJIT_EXT_EFFECT_PURE |
                         PYJIT_EXT_EFFECT_NO_PYTHON_CALLBACK;
    plan->mutation_input = -1;
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
              request->u.binary.right_type == &PyLong_Type ||
              request->u.binary.right_type == &PyFloatArrType_Type ||
              request->u.binary.right_type == &PyDoubleArrType_Type)) {
        if (!numpy_jit_array_scalar_enabled()) {
            return NULL;
        }
        layout = NUMPY_JIT_LAYOUT_C_CONTIG_ARRAY_SCALAR;
    }
    else if ((request->u.binary.left_type == &PyFloat_Type ||
              request->u.binary.left_type == &PyLong_Type ||
              request->u.binary.left_type == &PyFloatArrType_Type ||
              request->u.binary.left_type == &PyDoubleArrType_Type) &&
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
    PyJitExtPlan *plan = make_numpy_plan(candidate, NULL, site);
    if (plan != NULL && op == NUMPY_JIT_OP_MATMUL) {
        /* Matmul is an anchor plan executed through NumPy's normal ufunc;
           it has no elementwise copy-and-patch stencil. */
        plan->get_stencil = NULL;
        plan->flags |= PYJIT_EXT_PLAN_REGION_ANCHOR_ONLY;
        plan->op_id = NB_MATRIX_MULTIPLY;
        plan->debug_name = "numpy.matmul.anchor";
    }
    if (plan != NULL &&
        numpy_jit_binary_oparg_is_inplace(
            request->u.binary.binary_oparg)) {
        NumpyJitPlanPayload *payload =
            (NumpyJitPlanPayload *)plan->payload;
        payload->inplace = 1;
        payload->min_elements = 1;
        plan->flags &= ~PYJIT_EXT_PLAN_ALLOCATES_RESULT;
        plan->effect_flags &= ~PYJIT_EXT_EFFECT_PURE;
        plan->effect_flags |= PYJIT_EXT_EFFECT_MUTATES_INPUT;
        plan->mutation_input = 0;
        plan->op_id = 100u + (uint32_t)candidate->op;
        plan->get_stencil = NULL;
        plan->debug_name = "numpy.elemwise.float.inplace";
    }
    return plan;
}

static PyJitExtPlan *
numpy_jit_try_unary_plan(const PyJitExtTraceRequest *request,
                         PyJitExtSiteState *site)
{
    if (request->u.unary.unary_opcode != PYJIT_EXT_UNARY_NEGATIVE ||
        (request->u.unary.operand_type != NULL &&
         request->u.unary.operand_type != &PyArray_Type)) {
        return NULL;
    }
    NumpyJitPlanPayload *payload = PyMem_Calloc(1, sizeof(*payload));
    PyJitExtPlan *plan = PyMem_Calloc(1, sizeof(*plan));
    if (payload == NULL || plan == NULL) {
        PyMem_Free(payload);
        PyMem_Free(plan);
        PyErr_NoMemory();
        return NULL;
    }
    payload->op = NUMPY_JIT_OP_NEGATIVE;
    payload->layout = NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE;
    payload->min_elements = numpy_jit_get_min_elements();
    payload->site = site;
    numpy_jit_link_plan_stats(payload);
    plan->abi_version = PYJIT_EXT_ABI_VERSION;
    plan->struct_size = sizeof(*plan);
    plan->kind = PYJIT_EXT_PLAN_UNARY;
    plan->provider = &numpy_jit_provider;
    plan->flags = PYJIT_EXT_PLAN_REQUIRES_GIL |
                  PYJIT_EXT_PLAN_NO_PYTHON_CALLBACK |
                  PYJIT_EXT_PLAN_ALLOCATES_RESULT |
                  PYJIT_EXT_PLAN_REGION_ANCHOR_ONLY;
    plan->op_id = NUMPY_JIT_OP_NEGATIVE;
    plan->payload = payload;
    plan->eval.unary_eval = numpy_negative_unary_eval;
    plan->effect_flags = PYJIT_EXT_EFFECT_PURE |
                         PYJIT_EXT_EFFECT_NO_PYTHON_CALLBACK;
    plan->mutation_input = -1;
    plan->debug_name = "numpy.negative.region_anchor";
    return plan;
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
    if (first_payload == NULL || second_payload == NULL) {
        return NULL;
    }
    if (first_payload->inplace) {
        return NULL;
    }

    int fused_op = -1;
    const char *fused_pattern = NULL;
    int second_consumes_right =
        (request->u.fused_binary2.flags &
         PYJIT_EXT_FUSED_BINARY2_SECOND_CONSUMES_RIGHT) != 0;
    int supported_scaled_inplace_second =
        second_payload->inplace && second_consumes_right &&
        first_payload->op == NUMPY_JIT_OP_MULTIPLY &&
        (first_payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_ARRAY_SCALAR ||
         first_payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY) &&
        second_payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE &&
        (second_payload->op == NUMPY_JIT_OP_ADD ||
         second_payload->op == NUMPY_JIT_OP_SUBTRACT);
    int supported_add3_inplace_second =
        second_payload->inplace && second_consumes_right &&
        first_payload->op == NUMPY_JIT_OP_ADD &&
        first_payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE &&
        second_payload->op == NUMPY_JIT_OP_ADD &&
        second_payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE;
    int supported_inplace_second = supported_scaled_inplace_second ||
                                   supported_add3_inplace_second;
    if (second_payload->inplace && !supported_inplace_second) {
        return NULL;
    }

    /* (array + array) + array, including out += array + array. */
    if (first_payload->op == NUMPY_JIT_OP_ADD &&
        first_payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE &&
        second_payload->op == NUMPY_JIT_OP_ADD &&
        second_payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE) {
        fused_op = NUMPY_JIT_FUSED_ADD_THREE_ARRAYS;
        fused_pattern = second_payload->inplace ?
            "out+=a+b" : "(a+b)+c";
    }

    /* (array * scalar) +/- array, in either second-operand direction. */
    if (fused_op < 0 &&
        first_payload->op == NUMPY_JIT_OP_MULTIPLY &&
        (first_payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_ARRAY_SCALAR ||
         first_payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY) &&
        second_payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE) {
        if (second_payload->op == NUMPY_JIT_OP_ADD) {
            fused_op = NUMPY_JIT_FUSED_MUL_SCALAR_ADD_ARRAY;
            fused_pattern = second_consumes_right ?
                "a+b*scalar" : "a*scalar+b";
        }
        else if (second_payload->op == NUMPY_JIT_OP_SUBTRACT) {
            fused_op = second_consumes_right ?
                NUMPY_JIT_FUSED_ARRAY_SUB_MUL_SCALAR :
                NUMPY_JIT_FUSED_MUL_SCALAR_SUB_ARRAY;
            fused_pattern = second_consumes_right ?
                "a-b*scalar" : "a*scalar-b";
        }
    }

    /* (array +/- array) * scalar and (array +/- array) / scalar. */
    if (fused_op < 0 &&
        (first_payload->op == NUMPY_JIT_OP_ADD ||
         first_payload->op == NUMPY_JIT_OP_SUBTRACT) &&
        first_payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE) {
        NumpyJitLayoutKind expected_second_layout = second_consumes_right ?
            NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY :
            NUMPY_JIT_LAYOUT_C_CONTIG_ARRAY_SCALAR;
        if (second_payload->layout == expected_second_layout &&
            second_payload->op == NUMPY_JIT_OP_MULTIPLY) {
            fused_op = first_payload->op == NUMPY_JIT_OP_ADD ?
                NUMPY_JIT_FUSED_ADD_ARRAY_MUL_SCALAR :
                NUMPY_JIT_FUSED_SUB_ARRAY_MUL_SCALAR;
            if (first_payload->op == NUMPY_JIT_OP_ADD) {
                fused_pattern = second_consumes_right ?
                    "scalar*(a+b)" : "(a+b)*scalar";
            }
            else {
                fused_pattern = second_consumes_right ?
                    "scalar*(a-b)" : "(a-b)*scalar";
            }
        }
        else if (!second_consumes_right &&
                 second_payload->layout == expected_second_layout &&
                 second_payload->op == NUMPY_JIT_OP_TRUE_DIVIDE) {
            fused_op = first_payload->op == NUMPY_JIT_OP_ADD ?
                NUMPY_JIT_FUSED_ADD_ARRAY_DIV_SCALAR :
                NUMPY_JIT_FUSED_SUB_ARRAY_DIV_SCALAR;
            fused_pattern = first_payload->op == NUMPY_JIT_OP_ADD ?
                "(a+b)/scalar" : "(a-b)/scalar";
        }
    }

    /* (array * scalar) * scalar, in either second-operand direction. */
    if (fused_op < 0 &&
        first_payload->op == NUMPY_JIT_OP_MULTIPLY &&
        (first_payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_ARRAY_SCALAR ||
         first_payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY) &&
        second_payload->op == NUMPY_JIT_OP_MULTIPLY) {
        NumpyJitLayoutKind expected_second_layout = second_consumes_right ?
            NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY :
            NUMPY_JIT_LAYOUT_C_CONTIG_ARRAY_SCALAR;
        if (second_payload->layout == expected_second_layout) {
            fused_op = NUMPY_JIT_FUSED_MUL_TWO_SCALARS;
            fused_pattern = second_consumes_right ?
                "scalar2*(array*scalar1)" :
                "(array*scalar1)*scalar2";
        }
    }
    if (fused_op < 0) {
        return NULL;
    }

    NumpyJitPlanPayload *payload = PyMem_Calloc(1, sizeof(*payload));
    if (payload == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    payload->op = second_payload->op;
    payload->layout = first_payload->layout;
    payload->left_dtype = NPY_DOUBLE;
    payload->right_dtype = NPY_DOUBLE;
    payload->result_dtype = NPY_DOUBLE;
    payload->itemsize = (int)sizeof(double);
    payload->min_elements = numpy_jit_get_min_elements();
    payload->candidate = first_payload->candidate;
    payload->site = site;
    payload->fused_op = fused_op;
    payload->inplace = second_payload->inplace;
    payload->fused_flags = request->u.fused_binary2.flags;
    payload->fused_pattern = fused_pattern;
    numpy_jit_link_plan_stats(payload);

    PyJitExtPlan *plan = PyMem_Calloc(1, sizeof(*plan));
    if (plan == NULL) {
        numpy_jit_unlink_plan_stats(payload);
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
    if (payload->inplace) {
        plan->flags &= ~PYJIT_EXT_PLAN_ALLOCATES_RESULT;
    }
    plan->op_id = 1000u + (uint32_t)payload->fused_op;
    plan->payload = payload;
    plan->eval.fused_binary2_eval = numpy_fused_binary2_eval;
    plan->debug_name = "numpy.fused.binary2.arithmetic_chain";
    if (site != NULL) {
        if (site->fused_site_count != UINT16_MAX) {
            site->fused_site_count++;
        }
        site->fused_pattern = payload->fused_pattern;
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
        case PYJIT_EXT_TRACE_UNARY:
            return numpy_jit_try_unary_plan(request, site);
        case PYJIT_EXT_TRACE_FUSED_BINARY2:
            return numpy_jit_try_fused_binary2_plan(request, site);
        default:
            return NULL;
    }
}

static int
numpy_region_value_slot(const NumpyJitPlanPayload *payload,
                        const PyJitExtRegionRequest *request,
                        PyJitExtValueId value_id, int nodes_built)
{
    for (int i = 0; i < payload->region_input_count; i++) {
        if (payload->region_input_ids[i] == value_id) {
            return i;
        }
    }
    for (int i = 0; i < nodes_built; i++) {
        if (request->nodes[i].output == value_id) {
            return payload->region_input_count + i;
        }
    }
    return -1;
}

static int
numpy_jit_region_kernel_op(NumpyJitOp op)
{
    switch (op) {
        case NUMPY_JIT_OP_ADD:
            return NUMPY_JIT_KERNEL_REGION_ADD;
        case NUMPY_JIT_OP_SUBTRACT:
            return NUMPY_JIT_KERNEL_REGION_SUB;
        case NUMPY_JIT_OP_MULTIPLY:
            return NUMPY_JIT_KERNEL_REGION_MUL;
        case NUMPY_JIT_OP_TRUE_DIVIDE:
            return NUMPY_JIT_KERNEL_REGION_DIV;
        case NUMPY_JIT_OP_POWER:
            return NUMPY_JIT_KERNEL_REGION_SQUARE;
        case NUMPY_JIT_OP_NEGATIVE:
            return NUMPY_JIT_KERNEL_REGION_NEGATIVE;
        default:
            return -1;
    }
}

/* Build a heterogeneous schedule for ``matmul -> elementwise``.  The
   contraction remains a NumPy ufunc call, but writes directly into the final
   Region result; the epilogue then updates that buffer in place.  This avoids
   materializing the normal matmul temporary while keeping the structural DAG
   independent of NumPy layout policy. */
static PyJitExtPlan *
numpy_jit_build_matmul_epilogue_region(
        const PyJitExtRegionRequest *request, PyJitExtSiteState *site)
{
    if (request->node_count != 2 || request->input_count != 3) {
        return NULL;
    }
    const PyJitExtRegionNode *anchor = &request->nodes[0];
    const PyJitExtRegionNode *epilogue = &request->nodes[1];
    PyJitExtPlan *anchor_plan = (PyJitExtPlan *)anchor->atomic_plan;
    PyJitExtPlan *epilogue_plan = (PyJitExtPlan *)epilogue->atomic_plan;
    if (anchor->input_count != 2 || epilogue->input_count != 2 ||
        anchor_plan == NULL || epilogue_plan == NULL ||
        anchor_plan->provider != &numpy_jit_provider ||
        epilogue_plan->provider != &numpy_jit_provider ||
        anchor_plan->kind != PYJIT_EXT_PLAN_BINARY ||
        epilogue_plan->kind != PYJIT_EXT_PLAN_BINARY ||
        anchor_plan->payload == NULL || epilogue_plan->payload == NULL ||
        ((NumpyJitPlanPayload *)anchor_plan->payload)->op !=
            NUMPY_JIT_OP_MATMUL ||
        (epilogue_plan->effect_flags & PYJIT_EXT_EFFECT_MUTATES_INPUT)) {
        return NULL;
    }
    int consumes_left = epilogue->inputs[0] == anchor->output;
    int consumes_right = epilogue->inputs[1] == anchor->output;
    if (consumes_left == consumes_right) {
        return NULL;
    }
    NumpyJitPlanPayload *epilogue_payload =
        (NumpyJitPlanPayload *)epilogue_plan->payload;
    if (numpy_jit_region_kernel_op(epilogue_payload->op) < 0 ||
        epilogue_payload->op == NUMPY_JIT_OP_POWER) {
        return NULL;
    }

    NumpyJitPlanPayload *payload = PyMem_Calloc(1, sizeof(*payload));
    PyJitExtPlan *plan = PyMem_Calloc(1, sizeof(*plan));
    if (payload == NULL || plan == NULL) {
        PyMem_Free(payload);
        PyMem_Free(plan);
        PyErr_NoMemory();
        return NULL;
    }
    payload->region_matmul_epilogue = 1;
    payload->region_node_count = 2;
    payload->region_input_count = 3;
    payload->region_output_slot = 4;
    payload->region_mutation_input = -1;
    payload->min_elements = 1;
    payload->site = site;
    for (int i = 0; i < request->input_count; i++) {
        payload->region_input_ids[i] = request->inputs[i].value_id;
    }
    for (int i = 0; i < 2; i++) {
        int left = numpy_region_value_slot(
            payload, request, request->nodes[i].inputs[0], i);
        int right = numpy_region_value_slot(
            payload, request, request->nodes[i].inputs[1], i);
        if (left < 0 || right < 0) {
            PyMem_Free(payload);
            PyMem_Free(plan);
            return NULL;
        }
        payload->region_left[i] = (uint8_t)left;
        payload->region_right[i] = (uint8_t)right;
        payload->region_atomic_plans[i] =
            (PyJitExtPlan *)request->nodes[i].atomic_plan;
    }
    payload->region_ops[0] = NUMPY_JIT_OP_MATMUL;
    payload->region_ops[1] = epilogue_payload->op;
    payload->region_owner_plan = plan;
    numpy_jit_link_plan_stats(payload);

    plan->abi_version = PYJIT_EXT_ABI_VERSION;
    plan->struct_size = sizeof(*plan);
    plan->kind = PYJIT_EXT_PLAN_REGION;
    plan->provider = &numpy_jit_provider;
    plan->flags = PYJIT_EXT_PLAN_REQUIRES_GIL |
                  PYJIT_EXT_PLAN_NO_PYTHON_CALLBACK |
                  PYJIT_EXT_PLAN_ALLOCATES_RESULT;
    plan->op_id = 4000u + (uint32_t)epilogue_payload->op;
    plan->payload = payload;
    plan->eval.region_eval = numpy_region_eval;
    plan->region_fallback_eval = numpy_region_fallback_eval;
    plan->effect_flags = PYJIT_EXT_EFFECT_PURE |
                         PYJIT_EXT_EFFECT_NO_PYTHON_CALLBACK |
                         PYJIT_EXT_EFFECT_PRESERVES_FP_POLICY |
                         PYJIT_EXT_EFFECT_OWNS_ATOMIC_PLANS;
    plan->mutation_input = -1;
    plan->region_input_count = 3;
    plan->debug_name = "numpy.region.matmul_epilogue";
    return plan;
}

static PyJitExtPlan *
numpy_jit_build_generic_region(const PyJitExtRegionRequest *request,
                               PyJitExtSiteState *site)
{
    const char *generic_enabled =
        getenv("NUMPY_JIT_ENABLE_GENERIC_REGION_PROGRAM");
    if (generic_enabled != NULL && generic_enabled[0] == '0') {
        return NULL;
    }
    if (request->node_count < 2 ||
        request->node_count > PYJIT_EXT_REGION_MAX_NODES ||
        request->input_count < 2 ||
        request->input_count > PYJIT_EXT_REGION_MAX_INPUTS) {
        return NULL;
    }
    NumpyJitPlanPayload *payload = PyMem_Calloc(1, sizeof(*payload));
    if (payload == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    payload->region_generic = 1;
    payload->region_node_count = request->node_count;
    payload->region_input_count = request->input_count;
    payload->region_output_slot = request->input_count + request->node_count - 1;
    payload->region_mutation_input = -1;
    payload->itemsize = (int)sizeof(double);
    payload->min_elements = numpy_jit_get_min_elements();
    payload->site = site;
    for (int i = 0; i < request->input_count; i++) {
        payload->region_input_ids[i] = request->inputs[i].value_id;
    }

    int mutation_input = -1;
    for (int i = 0; i < request->node_count; i++) {
        const PyJitExtRegionNode *node = &request->nodes[i];
        PyJitExtPlan *atomic = (PyJitExtPlan *)node->atomic_plan;
        int expected_arity = atomic != NULL &&
            atomic->kind == PYJIT_EXT_PLAN_UNARY ? 1 : 2;
        if (node->input_count != expected_arity || atomic == NULL ||
            atomic->provider != &numpy_jit_provider ||
            (atomic->kind != PYJIT_EXT_PLAN_BINARY &&
             atomic->kind != PYJIT_EXT_PLAN_UNARY) || atomic->payload == NULL ||
            !(atomic->effect_flags & PYJIT_EXT_EFFECT_NO_PYTHON_CALLBACK) ||
            (i + 1 < request->node_count &&
             (!(atomic->effect_flags & PYJIT_EXT_EFFECT_PURE) ||
              (atomic->effect_flags & PYJIT_EXT_EFFECT_MUTATES_INPUT)))) {
            PyMem_Free(payload);
            return NULL;
        }
        NumpyJitPlanPayload *atomic_payload =
            (NumpyJitPlanPayload *)atomic->payload;
        int kernel_op = numpy_jit_region_kernel_op(atomic_payload->op);
        if (kernel_op < 0) {
            PyMem_Free(payload);
            return NULL;
        }
        int left = numpy_region_value_slot(
            payload, request, node->inputs[0], i);
        int right = expected_arity == 1 ? left : numpy_region_value_slot(
            payload, request, node->inputs[1], i);
        if (left < 0 || right < 0) {
            PyMem_Free(payload);
            return NULL;
        }
        payload->region_ops[i] = atomic_payload->op;
        payload->region_left[i] = (uint8_t)left;
        payload->region_right[i] = (uint8_t)right;
        payload->region_nodes[i] = (NumpyJitKernelRegionNode){
            .op = (uint8_t)kernel_op,
            .left = (uint8_t)left,
            .right = (uint8_t)right,
            .third = 0,
            .out = (uint8_t)(request->input_count + i),
        };
        payload->region_atomic_plans[i] = atomic;
        if (atomic->effect_flags & PYJIT_EXT_EFFECT_MUTATES_INPUT) {
            if (i + 1 != request->node_count || atomic->mutation_input != 0) {
                PyMem_Free(payload);
                return NULL;
            }
            mutation_input = numpy_region_value_slot(
                payload, request, node->inputs[0], i);
            if (mutation_input < 0 || mutation_input >= request->input_count) {
                PyMem_Free(payload);
                return NULL;
            }
            payload->inplace = 1;
            payload->region_mutation_input = (int8_t)mutation_input;
        }
    }

    int defer_to_atomic_prefix = 0;
    if (payload->region_node_count == 2) {
        int op0 = payload->region_nodes[0].op;
        int op1 = payload->region_nodes[1].op;
        defer_to_atomic_prefix =
            (op0 == NUMPY_JIT_KERNEL_REGION_SUB &&
             op1 == NUMPY_JIT_KERNEL_REGION_MUL) ||
            (op0 == NUMPY_JIT_KERNEL_REGION_MUL &&
             op1 == NUMPY_JIT_KERNEL_REGION_ADD) ||
            (op0 == NUMPY_JIT_KERNEL_REGION_MUL &&
             op1 == NUMPY_JIT_KERNEL_REGION_SUB) ||
            (op0 == NUMPY_JIT_KERNEL_REGION_ADD &&
             op1 == NUMPY_JIT_KERNEL_REGION_DIV) ||
            (op0 == NUMPY_JIT_KERNEL_REGION_MUL &&
             op1 == NUMPY_JIT_KERNEL_REGION_DIV);
    }
    else if (payload->region_node_count == 3) {
        int op0 = payload->region_nodes[0].op;
        int op1 = payload->region_nodes[1].op;
        int op2 = payload->region_nodes[2].op;
        defer_to_atomic_prefix =
            (op0 == NUMPY_JIT_KERNEL_REGION_MUL &&
             op1 == NUMPY_JIT_KERNEL_REGION_ADD &&
             op2 == NUMPY_JIT_KERNEL_REGION_DIV) ||
            (op0 == NUMPY_JIT_KERNEL_REGION_SUB &&
             op1 == NUMPY_JIT_KERNEL_REGION_MUL &&
             op2 == NUMPY_JIT_KERNEL_REGION_SUB);
    }
    if (defer_to_atomic_prefix) {
        /* These short, usually strided programs are cheaper as the existing
           vectorized atomic/dedicated prefix sequence than in the generic
           per-element interpreter.  Returning NULL lets the structural
           builder retry a shorter prefix without teaching CPython any NumPy
           semantic pattern. */
        PyMem_Free(payload);
        return NULL;
    }

    numpy_jit_link_plan_stats(payload);
    PyJitExtPlan *plan = PyMem_Calloc(1, sizeof(*plan));
    if (plan == NULL) {
        numpy_jit_unlink_plan_stats(payload);
        PyMem_Free(payload);
        PyErr_NoMemory();
        return NULL;
    }
    plan->abi_version = PYJIT_EXT_ABI_VERSION;
    plan->struct_size = sizeof(*plan);
    plan->kind = PYJIT_EXT_PLAN_REGION;
    plan->provider = &numpy_jit_provider;
    plan->flags = PYJIT_EXT_PLAN_REQUIRES_GIL |
                  PYJIT_EXT_PLAN_NO_PYTHON_CALLBACK |
                  PYJIT_EXT_PLAN_ALLOCATES_RESULT;
    plan->op_id = 3000u + request->node_count;
    plan->payload = payload;
    payload->region_owner_plan = plan;
    plan->eval.region_eval = numpy_region_eval;
    plan->region_fallback_eval = numpy_region_fallback_eval;
    plan->effect_flags = PYJIT_EXT_EFFECT_NO_PYTHON_CALLBACK |
                         PYJIT_EXT_EFFECT_PRESERVES_FP_POLICY |
                         PYJIT_EXT_EFFECT_OWNS_ATOMIC_PLANS;
    plan->mutation_input = mutation_input;
    plan->region_input_count = request->input_count;
    if (payload->inplace) {
        plan->flags &= ~PYJIT_EXT_PLAN_ALLOCATES_RESULT;
        plan->effect_flags |= PYJIT_EXT_EFFECT_MUTATES_INPUT;
    }
    else {
        plan->effect_flags |= PYJIT_EXT_EFFECT_PURE;
    }
    plan->debug_name = "numpy.region.elementwise.program.v1";
    return plan;
}

static PyJitExtPlan *
numpy_jit_prepare_region(const PyJitExtRegionRequest *request,
                     PyJitExtSiteState *site)
{
    if (request == NULL || request->provider != &numpy_jit_provider ||
        request->nodes == NULL || request->inputs == NULL) {
        return NULL;
    }
    PyJitExtPlan *matmul_epilogue =
        numpy_jit_build_matmul_epilogue_region(request, site);
    if (matmul_epilogue != NULL || PyErr_Occurred()) {
        return matmul_epilogue;
    }
    const char *legacy_binary2 =
        getenv("NUMPY_JIT_REGION_USE_LEGACY_BINARY2");
    if (request->node_count > 2 &&
        !(legacy_binary2 != NULL && legacy_binary2[0] == '0')) {
        for (int pair = 0; pair + 1 < request->node_count; pair++) {
            const PyJitExtRegionNode *first_node = &request->nodes[pair];
            const PyJitExtRegionNode *second_node =
                &request->nodes[pair + 1];
            PyJitExtPlan *first = (PyJitExtPlan *)first_node->atomic_plan;
            PyJitExtPlan *second = (PyJitExtPlan *)second_node->atomic_plan;
            int consumes_right =
                second_node->inputs[1] == first_node->output;
            if ((!consumes_right &&
                 second_node->inputs[0] != first_node->output) ||
                first == NULL || second == NULL ||
                first_node->input_count != 2 ||
                second_node->input_count != 2 ||
                first->provider != &numpy_jit_provider ||
                second->provider != &numpy_jit_provider) {
                continue;
            }
            PyJitExtTraceRequest prefix_request = {
                .kind = PYJIT_EXT_TRACE_FUSED_BINARY2,
                .u.fused_binary2 = {
                    .first_plan = first,
                    .second_plan = second,
                    .flags = consumes_right ?
                        PYJIT_EXT_FUSED_BINARY2_SECOND_CONSUMES_RIGHT :
                        PYJIT_EXT_FUSED_BINARY2_SECOND_CONSUMES_LEFT,
                },
            };
            PyJitExtPlan *prefix = numpy_jit_try_fused_binary2_plan(
                &prefix_request, NULL);
            if (prefix != NULL) {
                /* The dedicated prefix is materially faster than the
                   interpreted expression program for short vectors.  Ask
                   the structural builder to retry a shorter prefix; DAGs
                   without a dedicated prefix still use the 2-4 node generic
                   lowering below. */
                numpy_jit_free_plan(prefix);
                return NULL;
            }
            if (PyErr_Occurred()) {
                PyErr_Clear();
            }
        }
    }
    if ((legacy_binary2 != NULL && legacy_binary2[0] == '0') ||
        request->node_count != 2 ||
        request->input_count != 3) {
        return numpy_jit_build_generic_region(request, site);
    }
    const PyJitExtRegionNode *first_node = &request->nodes[0];
    const PyJitExtRegionNode *second_node = &request->nodes[1];
    PyJitExtPlan *first = (PyJitExtPlan *)first_node->atomic_plan;
    PyJitExtPlan *second = (PyJitExtPlan *)second_node->atomic_plan;
    if (first == NULL || second == NULL || first_node->input_count != 2 ||
        second_node->input_count != 2 ||
        first->provider != &numpy_jit_provider ||
        second->provider != &numpy_jit_provider ||
        (first->effect_flags & (PYJIT_EXT_EFFECT_PURE |
                                PYJIT_EXT_EFFECT_NO_PYTHON_CALLBACK)) !=
            (PYJIT_EXT_EFFECT_PURE |
             PYJIT_EXT_EFFECT_NO_PYTHON_CALLBACK) ||
        !(second->effect_flags & PYJIT_EXT_EFFECT_NO_PYTHON_CALLBACK) ||
        (((second->effect_flags & PYJIT_EXT_EFFECT_MUTATES_INPUT) != 0) !=
         (((NumpyJitPlanPayload *)second->payload)->inplace != 0))) {
        return NULL;
    }
    int consumes_right = second_node->inputs[1] == first_node->output;
    if (!consumes_right && second_node->inputs[0] != first_node->output) {
        return NULL;
    }

    PyJitExtTraceRequest legacy = {
        .kind = PYJIT_EXT_TRACE_FUSED_BINARY2,
        .u.fused_binary2 = {
            .first_plan = first,
            .second_plan = second,
            .flags = consumes_right ?
                PYJIT_EXT_FUSED_BINARY2_SECOND_CONSUMES_RIGHT :
                PYJIT_EXT_FUSED_BINARY2_SECOND_CONSUMES_LEFT,
        },
    };
    PyJitExtPlan *plan = numpy_jit_try_fused_binary2_plan(&legacy, site);
    if (plan == NULL) {
        return numpy_jit_build_generic_region(request, site);
    }
    NumpyJitPlanPayload *payload = (NumpyJitPlanPayload *)plan->payload;
    NumpyJitPlanPayload *first_payload =
        (NumpyJitPlanPayload *)first->payload;
    NumpyJitPlanPayload *second_payload =
        (NumpyJitPlanPayload *)second->payload;
    payload->region_first_op = first_payload->op;
    payload->region_second_op = second_payload->op;
    payload->region_second_consumes_right = consumes_right;
    payload->region_runtime_other_first =
        request->inputs[0].value_id != first_node->inputs[0];
    payload->region_node_count = request->node_count;
    payload->region_input_count = request->input_count;
    payload->region_output_slot = request->input_count + request->node_count - 1;
    for (int i = 0; i < request->input_count; i++) {
        payload->region_input_ids[i] = request->inputs[i].value_id;
    }
    for (int i = 0; i < request->node_count; i++) {
        int left = numpy_region_value_slot(
            payload, request, request->nodes[i].inputs[0], i);
        int right = numpy_region_value_slot(
            payload, request, request->nodes[i].inputs[1], i);
        if (left < 0 || right < 0) {
            memset(payload->region_atomic_plans, 0,
                   sizeof(payload->region_atomic_plans));
            payload->region_node_count = 0;
            numpy_jit_free_plan(plan);
            return NULL;
        }
        payload->region_left[i] = (uint8_t)left;
        payload->region_right[i] = (uint8_t)right;
        payload->region_atomic_plans[i] =
            (PyJitExtPlan *)request->nodes[i].atomic_plan;
    }
    plan->kind = PYJIT_EXT_PLAN_REGION;
    plan->eval.region_eval = numpy_region_eval;
    plan->region_fallback_eval = numpy_region_fallback_eval;
    plan->region_input_count = request->input_count;
    plan->effect_flags = PYJIT_EXT_EFFECT_NO_PYTHON_CALLBACK |
                         PYJIT_EXT_EFFECT_PRESERVES_FP_POLICY |
                         PYJIT_EXT_EFFECT_OWNS_ATOMIC_PLANS;
    if (payload->inplace) {
        plan->effect_flags |= PYJIT_EXT_EFFECT_MUTATES_INPUT;
        plan->mutation_input = consumes_right ? 0 : 2;
    }
    else {
        plan->effect_flags |= PYJIT_EXT_EFFECT_PURE;
        plan->mutation_input = -1;
    }
    plan->op_id = 2000u + (uint32_t)payload->fused_op;
    plan->debug_name = "numpy.region.elementwise.v1";
    payload->region_owner_plan = plan;
    return plan;
}

static void
numpy_jit_free_plan(PyJitExtPlan *plan)
{
    if (plan == NULL) {
        return;
    }
    NumpyJitPlanPayload *payload = (NumpyJitPlanPayload *)plan->payload;
    if (payload != NULL) {
        for (int i = 0; i < payload->region_pic_count; i++) {
            if (payload->region_pic[i].lowered != NULL) {
                numpy_jit_free_lowered_region(
                    payload->region_pic[i].lowered);
                payload->region_pic[i].lowered = NULL;
            }
        }
        for (int i = 0; i < payload->region_node_count; i++) {
            PyJitExtPlan *atomic = payload->region_atomic_plans[i];
            if (atomic != NULL && atomic != plan) {
                payload->region_atomic_plans[i] = NULL;
                numpy_jit_free_plan(atomic);
            }
        }
        numpy_jit_record_result_cache_stats(payload);
        numpy_jit_unlink_plan_stats(payload);
        Py_XDECREF(payload->result_cache.result);
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
    NUMPY_JIT_EVAL_LAYOUT_STRIDED_1D,
} NumpyJitEvalLayout;

typedef enum {
    NUMPY_JIT_SCALAR_LAYOUT_C_CONTIG,
    NUMPY_JIT_SCALAR_LAYOUT_STRIDED_1D,
    NUMPY_JIT_SCALAR_LAYOUT_STRIDED_INNER,
} NumpyJitScalarLayout;

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
        if (ndim == 1 &&
            numpy_jit_strided_1d_same_shape_enabled() &&
            PyArray_ISALIGNED(a) && PyArray_ISALIGNED(b) &&
            PyArray_STRIDE(a, 0) > 0 && PyArray_STRIDE(b, 0) > 0 &&
            PyArray_STRIDE(a, 0) % PyArray_ITEMSIZE(a) == 0 &&
            PyArray_STRIDE(b, 0) % PyArray_ITEMSIZE(b) == 0) {
            layout = NUMPY_JIT_EVAL_LAYOUT_STRIDED_1D;
        }
        else if (numpy_jit_strided_same_shape_enabled() &&
                 numpy_jit_positive_inner_contig_strides(a, b)) {
            layout = NUMPY_JIT_EVAL_LAYOUT_STRIDED_SAME_SHAPE;
        }
        else {
            numpy_jit_record_layout_miss(payload, left, right);
            numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_NOT_C_CONTIGUOUS);
            return 0;
        }
    }

    *a_out = a;
    *b_out = b;
    *size_out = size;
    *dtype_out = dtype;
    *layout_out = layout;
    return 1;
}

static int
contig_or_strided_float_array(
    NumpyJitPlanPayload *payload, PyObject *array,
    PyObject *detail_left, PyObject *detail_right,
    PyArrayObject **array_out, npy_intp *size_out,
    int *dtype_out, npy_intp *stride_out,
    NumpyJitScalarLayout *layout_out)
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
    int ndim = PyArray_NDIM(a);
    if (ndim == 0) {
        numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_NDIM_MISMATCH);
        return 0;
    }
    npy_intp stride = PyArray_ITEMSIZE(a);
    NumpyJitScalarLayout layout = NUMPY_JIT_SCALAR_LAYOUT_C_CONTIG;
    if (!PyArray_IS_C_CONTIGUOUS(a)) {
        if (!PyArray_ISALIGNED(a)) {
            goto unsupported_layout;
        }
        if (ndim == 1) {
            if (!numpy_jit_strided_1d_scalar_enabled() ||
                PyArray_STRIDE(a, 0) <= 0 ||
                PyArray_STRIDE(a, 0) % PyArray_ITEMSIZE(a) != 0) {
                goto unsupported_layout;
            }
            stride = PyArray_STRIDE(a, 0);
            layout = NUMPY_JIT_SCALAR_LAYOUT_STRIDED_1D;
        }
        else {
            npy_intp *strides = PyArray_STRIDES(a);
            if (!numpy_jit_strided_inner_scalar_enabled() ||
                strides[ndim - 1] != PyArray_ITEMSIZE(a)) {
                goto unsupported_layout;
            }
            for (int i = 0; i < ndim; i++) {
                if (strides[i] <= 0) {
                    goto unsupported_layout;
                }
            }
            layout = NUMPY_JIT_SCALAR_LAYOUT_STRIDED_INNER;
        }
    }
    npy_intp size = 1;
    npy_intp *dims = PyArray_DIMS(a);
    for (int i = 0; i < ndim; i++) {
        size *= dims[i];
    }
    if (layout == NUMPY_JIT_SCALAR_LAYOUT_STRIDED_INNER &&
        size > numpy_jit_get_strided_inner_scalar_max_elements()) {
        goto unsupported_layout;
    }
    *array_out = a;
    *size_out = size;
    *dtype_out = dtype;
    *stride_out = stride;
    *layout_out = layout;
    return 1;

unsupported_layout:
    numpy_jit_record_layout_miss(payload, detail_left, detail_right);
    numpy_jit_maybe_disable_static_miss(
        payload, "numpy_scalar_array_not_c_contiguous");
    numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_NOT_C_CONTIGUOUS);
    return 0;
}

static int
exact_python_scalar_as_double(NumpyJitPlanPayload *payload, PyObject *scalar,
                              double *value_out)
{
    double value;
    if (PyFloat_CheckExact(scalar) || PyLong_CheckExact(scalar)) {
        value = PyFloat_AsDouble(scalar);
    }
    else if (Py_TYPE(scalar) == &PyFloatArrType_Type) {
        value = (double)PyArrayScalar_VAL(scalar, Float);
    }
    else if (Py_TYPE(scalar) == &PyDoubleArrType_Type) {
        value = (double)PyArrayScalar_VAL(scalar, Double);
    }
    else {
        numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_NOT_SCALAR);
        return 0;
    }
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

    PyArrayObject *a;
    PyArrayObject *b;
    npy_intp size;
    int dtype;
    NumpyJitEvalLayout layout;
    if (!same_shape_float_layout(
            payload, left, right, &a, &b, &size, &dtype, &layout)) {
        return PYJIT_EXT_MISS;
    }
    if (dtype != NPY_DOUBLE) {
        numpy_jit_record_dtype_miss(payload, left, right);
        numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_DTYPE_MISMATCH);
        return PYJIT_EXT_MISS;
    }
    numpy_jit_record_element_count(payload, size);
    if (layout != NUMPY_JIT_EVAL_LAYOUT_C_CONTIG &&
        size < numpy_jit_get_strided_min_elements()) {
        numpy_jit_maybe_disable_static_miss(payload, "strided_min_elements");
        numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_STRIDED_MIN_ELEMENTS);
        return PYJIT_EXT_MISS;
    }
    if (layout == NUMPY_JIT_EVAL_LAYOUT_C_CONTIG &&
        size < payload->min_elements) {
        numpy_jit_guard_fail(payload, NUMPY_JIT_GUARD_MIN_ELEMENTS);
        return PYJIT_EXT_MISS;
    }

    PyArrayObject *out = numpy_jit_get_or_alloc_result(
        payload, PyArray_NDIM(a), PyArray_DIMS(a), NPY_DOUBLE, size);
    if (out == NULL) {
        return PYJIT_EXT_ERROR;
    }

    run->left_data = (const double *)PyArray_BYTES(a);
    run->right_data = (const double *)PyArray_BYTES(b);
    run->out_data = (double *)PyArray_BYTES(out);
    run->out = (PyObject *)out;

    if (layout == NUMPY_JIT_EVAL_LAYOUT_STRIDED_1D) {
        if (!numpy_jit_kernel_elemwise_strided_1d(
                (NumpyJitKernelOp)payload->op, NPY_DOUBLE, size,
                PyArray_BYTES(a), PyArray_STRIDE(a, 0),
                PyArray_BYTES(b), PyArray_STRIDE(b, 0),
                PyArray_BYTES(out))) {
            Py_DECREF(out);
            return PYJIT_EXT_MISS;
        }
        run->size = 0;
    }
    else if (layout == NUMPY_JIT_EVAL_LAYOUT_STRIDED_SAME_SHAPE) {
        if (!numpy_jit_kernel_elemwise_strided_same_shape(
                (NumpyJitKernelOp)payload->op, NPY_DOUBLE,
                PyArray_NDIM(a), size, PyArray_DIMS(a),
                PyArray_STRIDES(a), PyArray_STRIDES(b),
                PyArray_BYTES(a), PyArray_BYTES(b), PyArray_BYTES(out))) {
            Py_DECREF(out);
            return PYJIT_EXT_MISS;
        }
        run->size = 0;
    }
    else {
        run->size = size;
    }
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
    if (layout != NUMPY_JIT_EVAL_LAYOUT_C_CONTIG) {
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

    PyArrayObject *out;
    if (payload->inplace) {
        if (!PyArray_ISWRITEABLE(a) ||
            solve_may_share_memory(a, b, NPY_MAY_SHARE_BOUNDS) != 0) {
            return PYJIT_EXT_MISS;
        }
        out = a;
    }
    else {
        out = numpy_jit_get_or_alloc_result(
            payload, PyArray_NDIM(a), PyArray_DIMS(a), dtype, size);
        if (out == NULL) {
            return PYJIT_EXT_ERROR;
        }
    }

    if (layout == NUMPY_JIT_EVAL_LAYOUT_STRIDED_SAME_SHAPE) {
        PyJitExtResult strided_result =
            numpy_elemwise_strided_same_shape_loop(payload, a, b, out,
                                                   size, dtype);
        if (strided_result != PYJIT_EXT_HIT) {
            if (!payload->inplace) {
                Py_DECREF(out);
            }
            return strided_result;
        }
        if (payload->inplace) {
            Py_INCREF(out);
        }
        *result = (PyObject *)out;
        return PYJIT_EXT_HIT;
    }
    if (layout == NUMPY_JIT_EVAL_LAYOUT_STRIDED_1D) {
        if (!numpy_jit_kernel_elemwise_strided_1d(
                (NumpyJitKernelOp)payload->op, dtype, size,
                PyArray_BYTES(a), PyArray_STRIDE(a, 0),
                PyArray_BYTES(b), PyArray_STRIDE(b, 0),
                PyArray_BYTES(out))) {
            if (!payload->inplace) {
                Py_DECREF(out);
            }
            return PYJIT_EXT_MISS;
        }
        if (payload->inplace) {
            Py_INCREF(out);
        }
        *result = (PyObject *)out;
        return PYJIT_EXT_HIT;
    }
    if (!numpy_jit_kernel_elemwise_contig(
            (NumpyJitKernelOp)payload->op, dtype, PyArray_BYTES(a),
            PyArray_BYTES(b), PyArray_BYTES(out), size)) {
        if (!payload->inplace) {
            Py_DECREF(out);
        }
        return PYJIT_EXT_MISS;
    }

    if (payload->inplace) {
        Py_INCREF(out);
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
    npy_intp stride;
    NumpyJitScalarLayout layout;
    double scalar_value;
    int dtype;
    PyObject *detail_left = scalar_first ? scalar : array;
    PyObject *detail_right = scalar_first ? array : scalar;
    if (!contig_or_strided_float_array(
            payload, array, detail_left, detail_right,
            &a, &size, &dtype, &stride, &layout)) {
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

    PyArrayObject *out;
    if (payload->inplace) {
        if (scalar_first || !PyArray_ISWRITEABLE(a)) {
            return PYJIT_EXT_MISS;
        }
        out = a;
    }
    else {
        out = numpy_jit_get_or_alloc_result(
            payload, PyArray_NDIM(a), PyArray_DIMS(a), dtype, size);
        if (out == NULL) {
            return PYJIT_EXT_ERROR;
        }
    }

    int kernel_ok;
    if (layout == NUMPY_JIT_SCALAR_LAYOUT_C_CONTIG) {
        kernel_ok = numpy_jit_kernel_array_scalar_contig(
            (NumpyJitKernelOp)payload->op, dtype, PyArray_BYTES(a),
            scalar_value, scalar_first, PyArray_BYTES(out), size);
    }
    else if (layout == NUMPY_JIT_SCALAR_LAYOUT_STRIDED_1D) {
        kernel_ok = numpy_jit_kernel_array_scalar_strided_1d(
            (NumpyJitKernelOp)payload->op, dtype, PyArray_BYTES(a), stride,
            scalar_value, scalar_first, PyArray_BYTES(out), size);
    }
    else {
        kernel_ok = numpy_jit_kernel_array_scalar_strided_inner(
            (NumpyJitKernelOp)payload->op, dtype, PyArray_NDIM(a), size,
            PyArray_DIMS(a), PyArray_STRIDES(a), PyArray_BYTES(a),
            scalar_value, scalar_first, PyArray_BYTES(out));
    }
    if (!kernel_ok) {
        if (!payload->inplace) {
            Py_DECREF(out);
        }
        return PYJIT_EXT_MISS;
    }

    if (payload->inplace) {
        Py_INCREF(out);
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
numpy_power2_array_scalar_eval(void *payload, PyObject *left,
                               PyObject *right, PyObject **result)
{
    if (!numpy_jit_is_exact_power2_exponent(right)) {
        PyObject *fallback = PyNumber_Power(left, right, Py_None);
        if (fallback == NULL) {
            return PYJIT_EXT_ERROR;
        }
        *result = fallback;
        return PYJIT_EXT_HIT;
    }
    return numpy_elemwise_array_scalar_eval_common(
        (NumpyJitPlanPayload *)payload, left, right, 0, result);
}

static PyJitExtResult
numpy_matmul_binary_eval(void *NPY_UNUSED(payload), PyObject *left,
                         PyObject *right, PyObject **result)
{
    PyObject *value = PyNumber_MatrixMultiply(left, right);
    if (value == NULL) {
        return PYJIT_EXT_ERROR;
    }
    *result = value;
    return PYJIT_EXT_HIT;
}

static PyJitExtResult
numpy_negative_unary_eval(void *NPY_UNUSED(payload), PyObject *operand,
                          PyObject **result)
{
    PyObject *value = PyNumber_Negative(operand);
    if (value == NULL) {
        return PYJIT_EXT_ERROR;
    }
    *result = value;
    return PYJIT_EXT_HIT;
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
    if (PyArray_CheckExact(args[0]) && PyArray_CheckExact(args[1])) {
        return numpy_elemwise_eval_common(plan, args[0], args[1], result);
    }
    if (PyArray_CheckExact(args[0]) &&
        (PyFloat_CheckExact(args[1]) || PyLong_CheckExact(args[1])) &&
        numpy_jit_array_scalar_enabled()) {
        NumpyJitLayoutKind saved_layout = plan->layout;
        plan->layout = NUMPY_JIT_LAYOUT_C_CONTIG_ARRAY_SCALAR;
        PyJitExtResult eval_result = numpy_elemwise_array_scalar_eval_common(
            plan, args[0], args[1], 0, result);
        plan->layout = saved_layout;
        return eval_result;
    }
    if ((PyFloat_CheckExact(args[0]) || PyLong_CheckExact(args[0])) &&
        PyArray_CheckExact(args[1]) &&
        numpy_jit_scalar_array_enabled()) {
        NumpyJitLayoutKind saved_layout = plan->layout;
        plan->layout = NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY;
        PyJitExtResult eval_result = numpy_elemwise_array_scalar_eval_common(
            plan, args[1], args[0], 1, result);
        plan->layout = saved_layout;
        return eval_result;
    }
    return PYJIT_EXT_MISS;
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
        if (ndim == 1 && PyArray_ISALIGNED(a) && PyArray_ISALIGNED(b) &&
            PyArray_STRIDE(a, 0) > 0 && PyArray_STRIDE(b, 0) > 0 &&
            PyArray_STRIDE(a, 0) % PyArray_ITEMSIZE(a) == 0 &&
            PyArray_STRIDE(b, 0) % PyArray_ITEMSIZE(b) == 0) {
            layout = NUMPY_JIT_EVAL_LAYOUT_STRIDED_1D;
        }
        else if (numpy_jit_positive_inner_contig_strides(a, b)) {
            layout = NUMPY_JIT_EVAL_LAYOUT_STRIDED_SAME_SHAPE;
        }
        else {
            PyErr_SetString(PyExc_ValueError, "expected C-contiguous or positive inner-contiguous strides");
            return 0;
        }
    }
    *a_out = a;
    *b_out = b;
    *size_out = size;
    *dtype_out = dtype;
    *layout_out = layout;
    return 1;
}

static PyObject *
numpy_jit_fused_eval(NumpyJitPlanPayload *payload, PyObject *a_obj,
                     double alpha, PyObject *b_obj, double beta,
                     NumpyJitFusedOp op)
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

    PyArrayObject *out = numpy_jit_get_or_alloc_result(
        payload, PyArray_NDIM(a), PyArray_DIMS(a), dtype, size);
    if (out == NULL) {
        return NULL;
    }

    if (!numpy_jit_kernel_fused_contig_or_strided(
            (int)op, dtype,
            layout != NUMPY_JIT_EVAL_LAYOUT_C_CONTIG,
            PyArray_NDIM(a), size, PyArray_DIMS(a), PyArray_STRIDES(a),
            PyArray_STRIDES(b), PyArray_BYTES(a), PyArray_BYTES(b),
            alpha, beta, PyArray_BYTES(out))) {
        Py_DECREF(out);
        PyErr_SetString(PyExc_ValueError, "unsupported fused JIT kernel");
        return NULL;
    }
    return (PyObject *)out;
}

static int
numpy_jit_contig_disjoint(PyArrayObject *left, PyArrayObject *right,
                          npy_intp size)
{
    if (size < 0 || (uint64_t)size > UINTPTR_MAX /
            (uint64_t)PyArray_ITEMSIZE(left)) {
        return 0;
    }
    uintptr_t left_start = (uintptr_t)PyArray_BYTES(left);
    uintptr_t right_start = (uintptr_t)PyArray_BYTES(right);
    uintptr_t bytes = (uintptr_t)size * (uintptr_t)PyArray_ITEMSIZE(left);
    if (left_start > UINTPTR_MAX - bytes || right_start > UINTPTR_MAX - bytes) {
        return 0;
    }
    return left_start + bytes <= right_start ||
           right_start + bytes <= left_start;
}

static PyJitExtResult
numpy_fused_add3_eval(NumpyJitPlanPayload *payload, PyObject *first_left,
                      PyObject *first_right, PyObject *second_other,
                      PyObject **result)
{
    PyArrayObject *a;
    PyArrayObject *b;
    npy_intp size;
    int dtype;
    NumpyJitEvalLayout layout;
    if (!numpy_jit_fused_same_shape(
            first_left, first_right, &a, &b, &size, &dtype, &layout)) {
        PyErr_Clear();
        return PYJIT_EXT_MISS;
    }
    if (!PyArray_CheckExact(second_other) ||
        layout != NUMPY_JIT_EVAL_LAYOUT_C_CONTIG ||
        size < payload->min_elements) {
        return PYJIT_EXT_MISS;
    }

    PyArrayObject *c = (PyArrayObject *)second_other;
    int ndim = PyArray_NDIM(a);
    if (PyArray_TYPE(c) != dtype || PyArray_NDIM(c) != ndim ||
        !PyArray_IS_C_CONTIGUOUS(c)) {
        return PYJIT_EXT_MISS;
    }
    for (int i = 0; i < ndim; i++) {
        if (PyArray_DIM(c, i) != PyArray_DIM(a, i)) {
            return PYJIT_EXT_MISS;
        }
    }

    if (payload->inplace) {
        if (!PyArray_ISWRITEABLE(c) ||
            !numpy_jit_contig_disjoint(a, c, size) ||
            !numpy_jit_contig_disjoint(b, c, size)) {
            return PYJIT_EXT_MISS;
        }
        if (!numpy_jit_kernel_add3_contig(
                dtype, PyArray_BYTES(a), PyArray_BYTES(b), PyArray_BYTES(c),
                PyArray_BYTES(c), size, 1)) {
            return PYJIT_EXT_MISS;
        }
        Py_INCREF(c);
        *result = (PyObject *)c;
        return PYJIT_EXT_HIT;
    }

    PyArrayObject *out = numpy_jit_get_or_alloc_result(
        payload, ndim, PyArray_DIMS(a), dtype, size);
    if (out == NULL) {
        return PYJIT_EXT_ERROR;
    }
    if (!numpy_jit_kernel_add3_contig(
            dtype, PyArray_BYTES(a), PyArray_BYTES(b), PyArray_BYTES(c),
            PyArray_BYTES(out), size,
            (payload->fused_flags &
             PYJIT_EXT_FUSED_BINARY2_SECOND_CONSUMES_RIGHT) ? 2 : 0)) {
        Py_DECREF(out);
        return PYJIT_EXT_MISS;
    }
    *result = (PyObject *)out;
    return PYJIT_EXT_HIT;
}

static PyJitExtResult
numpy_fused_binary2_eval(void *payload_obj, PyObject *first_left,
                         PyObject *first_right, PyObject *second_other,
                         PyObject **result)
{
    NumpyJitPlanPayload *payload = (NumpyJitPlanPayload *)payload_obj;
    if (payload->fused_op == NUMPY_JIT_FUSED_ADD_THREE_ARRAYS) {
        return numpy_fused_add3_eval(
            payload, first_left, first_right, second_other, result);
    }
    if (payload->fused_op == NUMPY_JIT_FUSED_MUL_TWO_SCALARS) {
        PyObject *array = first_left;
        PyObject *first_scalar = first_right;
        if (payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY) {
            array = first_right;
            first_scalar = first_left;
        }
        double first_value;
        double second_value;
        if (!exact_python_scalar_as_double(
                payload, first_scalar, &first_value) ||
            !exact_python_scalar_as_double(
                payload, second_other, &second_value)) {
            return PYJIT_EXT_MISS;
        }
        PyObject *out = numpy_jit_fused_eval(
            payload, array, first_value, array, second_value,
            NUMPY_JIT_FUSED_MUL_TWO_SCALARS);
        if (out == NULL) {
            if (PyErr_ExceptionMatches(PyExc_TypeError) ||
                PyErr_ExceptionMatches(PyExc_ValueError)) {
                PyErr_Clear();
                return PYJIT_EXT_MISS;
            }
            return PYJIT_EXT_ERROR;
        }
        *result = out;
        return PYJIT_EXT_HIT;
    }
    PyObject *array_a;
    PyObject *array_b;
    PyObject *scalar;
    if (payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE) {
        array_a = first_left;
        array_b = first_right;
        scalar = second_other;
    }
    else {
        array_a = first_left;
        scalar = first_right;
        if (payload->layout == NUMPY_JIT_LAYOUT_C_CONTIG_SCALAR_ARRAY) {
            array_a = first_right;
            scalar = first_left;
        }
        array_b = second_other;
    }

    double scalar_value;
    if (!exact_python_scalar_as_double(payload, scalar, &scalar_value)) {
        numpy_jit_log_fused_refs(first_left, first_right, second_other,
                                 NULL, "MISS");
        return PYJIT_EXT_MISS;
    }

    if (payload->inplace) {
        PyArrayObject *a;
        PyArrayObject *out;
        npy_intp size;
        int dtype;
        NumpyJitEvalLayout layout;
        if (!numpy_jit_fused_same_shape(
                array_a, array_b, &a, &out, &size, &dtype, &layout)) {
            PyErr_Clear();
            return PYJIT_EXT_MISS;
        }
        if (size < payload->min_elements || !PyArray_ISWRITEABLE(out) ||
            solve_may_share_memory(a, out, NPY_MAY_SHARE_BOUNDS) != 0) {
            return PYJIT_EXT_MISS;
        }
        if (!numpy_jit_kernel_fused_contig_or_strided(
                payload->fused_op, dtype,
                layout != NUMPY_JIT_EVAL_LAYOUT_C_CONTIG,
                PyArray_NDIM(a), size, PyArray_DIMS(a), PyArray_STRIDES(a),
                PyArray_STRIDES(out), PyArray_BYTES(a), PyArray_BYTES(out),
                scalar_value, 0.0, PyArray_BYTES(out))) {
            return PYJIT_EXT_MISS;
        }
        Py_INCREF(out);
        *result = (PyObject *)out;
        return PYJIT_EXT_HIT;
    }

    PyObject *out = numpy_jit_fused_eval(
        payload, array_a, scalar_value, array_b, 0.0,
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

static void
numpy_region_logical_args(NumpyJitPlanPayload *payload,
                          PyObject *const *args,
                          PyObject **first_left,
                          PyObject **first_right,
                          PyObject **second_other)
{
    if (payload->region_runtime_other_first) {
        *second_other = args[0];
        *first_left = args[1];
        *first_right = args[2];
    }
    else {
        *first_left = args[0];
        *first_right = args[1];
        *second_other = args[2];
    }
}

static int
numpy_jit_region_power2_args_valid(const NumpyJitPlanPayload *payload,
                                   PyObject *const *args,
                                   Py_ssize_t nargs)
{
    for (int i = 0; i < payload->region_node_count; i++) {
        if (payload->region_ops[i] != NUMPY_JIT_OP_POWER) {
            continue;
        }
        int exponent_slot = payload->region_right[i];
        if (exponent_slot < 0 || exponent_slot >= nargs ||
            !numpy_jit_is_exact_power2_exponent(args[exponent_slot])) {
            return 0;
        }
    }
    return 1;
}

static PyJitExtResult
numpy_region_matmul_epilogue_eval(NumpyJitPlanPayload *payload,
                                  PyObject *const *args, Py_ssize_t nargs,
                                  PyObject **result)
{
    if (nargs != 3) {
        return PYJIT_EXT_MISS;
    }
    int a_slot = payload->region_left[0];
    int b_slot = payload->region_right[0];
    int internal_slot = payload->region_input_count;
    int second_left = payload->region_left[1];
    int second_right = payload->region_right[1];
    int internal_is_left = second_left == internal_slot;
    int other_slot = internal_is_left ? second_right : second_left;
    if (a_slot < 0 || a_slot >= nargs || b_slot < 0 || b_slot >= nargs ||
        other_slot < 0 || other_slot >= nargs ||
        (internal_is_left == (second_right == internal_slot)) ||
        !PyArray_CheckExact(args[a_slot]) ||
        !PyArray_CheckExact(args[b_slot])) {
        return PYJIT_EXT_MISS;
    }
    PyArrayObject *a = (PyArrayObject *)args[a_slot];
    PyArrayObject *b = (PyArrayObject *)args[b_slot];
    int dtype = PyArray_TYPE(a);
    if (PyArray_NDIM(a) != 2 || PyArray_NDIM(b) != 2 ||
        dtype != PyArray_TYPE(b) ||
        !numpy_jit_is_supported_float_dtype(dtype) ||
        !PyArray_IS_C_CONTIGUOUS(a) || !PyArray_IS_C_CONTIGUOUS(b) ||
        PyArray_DIM(a, 1) != PyArray_DIM(b, 0)) {
        return PYJIT_EXT_MISS;
    }
    npy_intp dims[2] = {PyArray_DIM(a, 0), PyArray_DIM(b, 1)};
    npy_intp size = dims[0] * dims[1];
    PyObject *other = args[other_slot];
    PyArrayObject *other_array = NULL;
    double scalar = 0.0;
    int other_is_scalar = !PyArray_CheckExact(other);
    if (other_is_scalar) {
        if (!exact_python_scalar_as_double(payload, other, &scalar)) {
            return PYJIT_EXT_MISS;
        }
    }
    else {
        other_array = (PyArrayObject *)other;
        int same_shape = PyArray_NDIM(other_array) == 2 &&
            PyArray_DIM(other_array, 0) == dims[0] &&
            PyArray_DIM(other_array, 1) == dims[1];
        int row_bias = PyArray_NDIM(other_array) == 1 &&
            PyArray_DIM(other_array, 0) == dims[1];
        if (PyArray_TYPE(other_array) != dtype ||
            (!same_shape && !row_bias) ||
            !PyArray_IS_C_CONTIGUOUS(other_array)) {
            return PYJIT_EXT_MISS;
        }
    }

    PyObject *matmul = numpy_jit_known_ufunc(NUMPY_JIT_OP_MATMUL);
    if (matmul == NULL) {
        return PYJIT_EXT_MISS;
    }
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(2, dims, dtype);
    if (out == NULL) {
        return PYJIT_EXT_ERROR;
    }
    PyObject *matmul_result = PyObject_CallFunctionObjArgs(
        matmul, (PyObject *)a, (PyObject *)b, (PyObject *)out, NULL);
    if (matmul_result == NULL) {
        Py_DECREF(out);
        return PYJIT_EXT_ERROR;
    }
    Py_DECREF(matmul_result);

    /* For small outputs, keeping the epilogue in this C frame avoids a
       second ufunc dispatch and dominates the benefit.  Once the output no
       longer fits comfortably in L1, NumPy's vectorized ufunc loop is faster
       than the scalar schedule while ``out=out`` still removes the temporary
       allocation and extra write. */
    if (size > 2048) {
        PyObject *epilogue = numpy_jit_known_ufunc(payload->region_ops[1]);
        if (epilogue == NULL) {
            Py_DECREF(out);
            return PYJIT_EXT_MISS;
        }
        PyObject *left = internal_is_left ? (PyObject *)out : other;
        PyObject *right = internal_is_left ? other : (PyObject *)out;
        PyObject *epilogue_result = PyObject_CallFunctionObjArgs(
            epilogue, left, right, (PyObject *)out, NULL);
        if (epilogue_result == NULL) {
            Py_DECREF(out);
            return PYJIT_EXT_ERROR;
        }
        Py_DECREF(epilogue_result);
        *result = (PyObject *)out;
        return PYJIT_EXT_HIT;
    }

    int kernel_ok;
    NumpyJitKernelOp op = (NumpyJitKernelOp)payload->region_ops[1];
    if (other_is_scalar) {
        kernel_ok = numpy_jit_kernel_array_scalar_contig(
            op, dtype, PyArray_BYTES(out), scalar, !internal_is_left,
            PyArray_BYTES(out), size);
    }
    else {
        if (PyArray_NDIM(other_array) == 1) {
            kernel_ok = numpy_jit_kernel_row_bias_contig(
                op, dtype, PyArray_BYTES(out), PyArray_BYTES(other_array),
                dims[0], dims[1], internal_is_left);
        }
        else {
            const void *left_data = internal_is_left ?
                (const void *)PyArray_BYTES(out) :
                (const void *)PyArray_BYTES(other_array);
            const void *right_data = internal_is_left ?
                (const void *)PyArray_BYTES(other_array) :
                (const void *)PyArray_BYTES(out);
            kernel_ok = numpy_jit_kernel_elemwise_contig(
                op, dtype, left_data, right_data, PyArray_BYTES(out), size);
        }
    }
    if (!kernel_ok) {
        Py_DECREF(out);
        return PYJIT_EXT_MISS;
    }
    *result = (PyObject *)out;
    return PYJIT_EXT_HIT;
}

static PyJitExtResult
numpy_region_generic_eval(NumpyJitPlanPayload *payload,
                          PyObject *const *args, Py_ssize_t nargs,
                          PyObject **result)
{
    if (nargs != payload->region_input_count) {
        return PYJIT_EXT_MISS;
    }
    if (!numpy_jit_region_power2_args_valid(payload, args, nargs)) {
        return PYJIT_EXT_MISS;
    }
    const void *input_data[PYJIT_EXT_REGION_MAX_INPUTS] = {0};
    const npy_intp *input_strides[PYJIT_EXT_REGION_MAX_INPUTS] = {0};
    double scalar_values[PYJIT_EXT_REGION_MAX_INPUTS] = {0.0};
    uint8_t input_is_scalar[PYJIT_EXT_REGION_MAX_INPUTS] = {0};
    PyArrayObject *shape_source = NULL;
    int dtype = -1;
    npy_intp size = -1;
    int all_c_contiguous = 1;
    for (Py_ssize_t i = 0; i < nargs; i++) {
        if (PyArray_CheckExact(args[i])) {
            PyArrayObject *array = (PyArrayObject *)args[i];
            int current_dtype = PyArray_TYPE(array);
            if ((current_dtype != NPY_FLOAT && current_dtype != NPY_DOUBLE) ||
                numpy_jit_region_layout_class(array) == 0) {
                return PYJIT_EXT_MISS;
            }
            if (!PyArray_IS_C_CONTIGUOUS(array)) {
                all_c_contiguous = 0;
            }
            if (shape_source == NULL) {
                shape_source = array;
                dtype = current_dtype;
                size = PyArray_SIZE(array);
            }
            else {
                if (current_dtype != dtype ||
                    PyArray_NDIM(array) != PyArray_NDIM(shape_source)) {
                    return PYJIT_EXT_MISS;
                }
                for (int dim = 0; dim < PyArray_NDIM(array); dim++) {
                    if (PyArray_DIM(array, dim) !=
                        PyArray_DIM(shape_source, dim)) {
                        return PYJIT_EXT_MISS;
                    }
                }
            }
            input_data[i] = PyArray_BYTES(array);
            input_strides[i] = PyArray_STRIDES(array);
        }
        else {
            input_is_scalar[i] = 1;
            if (!exact_python_scalar_as_double(
                    payload, args[i], &scalar_values[i])) {
                return PYJIT_EXT_MISS;
            }
        }
    }
    if (shape_source == NULL || size < payload->min_elements) {
        return PYJIT_EXT_MISS;
    }
    payload->itemsize = numpy_jit_itemsize_for_dtype(dtype);

    PyArrayObject *out;
    if (payload->inplace) {
        int target_index = payload->region_mutation_input;
        if (target_index < 0 || target_index >= nargs ||
            !PyArray_CheckExact(args[target_index])) {
            return PYJIT_EXT_MISS;
        }
        out = (PyArrayObject *)args[target_index];
        if (!PyArray_ISWRITEABLE(out)) {
            return PYJIT_EXT_MISS;
        }
        for (Py_ssize_t i = 0; i < nargs; i++) {
            if (!PyArray_CheckExact(args[i]) || args[i] == (PyObject *)out) {
                continue;
            }
            if (solve_may_share_memory(
                    out, (PyArrayObject *)args[i], NPY_MAY_SHARE_BOUNDS) != 0) {
                return PYJIT_EXT_MISS;
            }
        }
    }
    else {
        out = numpy_jit_get_or_alloc_result(
            payload, PyArray_NDIM(shape_source), PyArray_DIMS(shape_source),
            dtype, size);
        if (out == NULL) {
            return PYJIT_EXT_ERROR;
        }
    }
    int kernel_ok;
    if (all_c_contiguous && PyArray_IS_C_CONTIGUOUS(out)) {
        kernel_ok = numpy_jit_kernel_region_contig(
            payload->region_nodes, payload->region_node_count,
            payload->region_input_count, payload->region_output_slot, dtype,
            input_data, scalar_values, input_is_scalar,
            PyArray_BYTES(out), size);
    }
    else {
        kernel_ok = numpy_jit_kernel_region_strided(
            payload->region_nodes, payload->region_node_count,
            payload->region_input_count, payload->region_output_slot, dtype,
            PyArray_NDIM(shape_source), PyArray_DIMS(shape_source),
            input_data, input_strides, scalar_values, input_is_scalar,
            PyArray_BYTES(out), PyArray_STRIDES(out), size);
    }
    if (!kernel_ok) {
        if (!payload->inplace) {
            Py_DECREF(out);
        }
        return PYJIT_EXT_MISS;
    }
    if (payload->inplace) {
        Py_INCREF(out);
    }
    *result = (PyObject *)out;
    return PYJIT_EXT_HIT;
}

static PyJitExtResult
numpy_region_lowered_body(void *payload_obj, PyObject *const *args,
                          Py_ssize_t nargs, PyObject **result)
{
    if (payload_obj == NULL || args == NULL) {
        return PYJIT_EXT_MISS;
    }
    NumpyJitPlanPayload *payload = (NumpyJitPlanPayload *)payload_obj;
    if ((!payload->region_generic && nargs != 3) ||
        (payload->region_generic && nargs != payload->region_input_count)) {
        return PYJIT_EXT_MISS;
    }
    PyObject *first_left = NULL;
    PyObject *first_right = NULL;
    PyObject *second_other = NULL;
    if (!payload->region_generic && !payload->region_matmul_epilogue) {
        numpy_region_logical_args(payload, args, &first_left, &first_right,
                                  &second_other);
    }
    int errmask;
    if (_get_errmask_fast(&errmask) < 0) {
        return PYJIT_EXT_ERROR;
    }
    /* NumPy encodes four independent policies in three-bit fields.  Only
       ignore and warn are safe for speculative execution: warn can replay
       the exact Python recipe if the fused pass raises an FP status bit. */
    for (int shift = 0; shift <= 9; shift += 3) {
        int mode = (errmask >> shift) & 7;
        if (mode > 1) {
            return PYJIT_EXT_MISS;
        }
    }
    int transactional_inplace = payload->inplace && errmask != 0;
    PyArrayObject *mutation_target = NULL;
    if (transactional_inplace) {
        int target_index = payload->region_owner_plan != NULL ?
            payload->region_owner_plan->mutation_input : -1;
        if (target_index < 0 || target_index >= nargs ||
            !PyArray_CheckExact(args[target_index]) ||
            !PyArray_ISWRITEABLE((PyArrayObject *)args[target_index])) {
            return PYJIT_EXT_MISS;
        }
        mutation_target = (PyArrayObject *)args[target_index];
    }
    char fp_barrier;
    if (errmask != 0) {
        npy_clear_floatstatus_barrier(&fp_barrier);
    }
    uint8_t saved_inplace = payload->inplace;
    if (transactional_inplace) {
        /* Do not expose a partial mutation if the fused pass raises an FP
           status bit.  Build into the reusable result buffer first; only a
           clean pass is committed to the original ndarray. */
        payload->inplace = 0;
    }
    PyJitExtResult status;
    if (payload->region_matmul_epilogue) {
        status = numpy_region_matmul_epilogue_eval(
            payload, args, nargs, result);
    }
    else if (payload->region_generic) {
        status = numpy_region_generic_eval(payload, args, nargs, result);
    }
    else {
        status = numpy_fused_binary2_eval(
            payload, first_left, first_right, second_other, result);
    }
    payload->inplace = saved_inplace;
    if ((status == PYJIT_EXT_HIT || status == PYJIT_EXT_COLD) &&
        errmask != 0 &&
        npy_get_floatstatus_barrier(&fp_barrier) != 0) {
        Py_CLEAR(*result);
        npy_clear_floatstatus_barrier(&fp_barrier);
        return PYJIT_EXT_MISS;
    }
    if ((status == PYJIT_EXT_HIT || status == PYJIT_EXT_COLD) &&
        transactional_inplace) {
        PyObject *temporary = *result;
        if (!PyArray_CheckExact(temporary) ||
            PyArray_CopyInto(mutation_target,
                             (PyArrayObject *)temporary) < 0) {
            Py_CLEAR(*result);
            return PYJIT_EXT_ERROR;
        }
        Py_DECREF(temporary);
        Py_INCREF(mutation_target);
        *result = (PyObject *)mutation_target;
    }
    return status;
}

typedef struct {
    NumpyJitPlanPayload *owner;
    uint64_t signature;
    uint8_t nargs;
    uint8_t array_mask;
    uint8_t writable_mask;
    int8_t layouts[PYJIT_EXT_REGION_MAX_INPUTS];
    int8_t dtypes[PYJIT_EXT_REGION_MAX_INPUTS];
    int8_t ndims[PYJIT_EXT_REGION_MAX_INPUTS];
    PyTypeObject *types[PYJIT_EXT_REGION_MAX_INPUTS];
} NumpyJitRegionLowering;

static void
numpy_jit_region_inc_u16(uint16_t *value)
{
    if (value != NULL && *value != UINT16_MAX) {
        (*value)++;
    }
}

static uint64_t
numpy_jit_region_hash_u64(uint64_t hash, uint64_t value)
{
    return (hash ^ value) * 1099511628211ULL;
}

static int
numpy_jit_region_layout_class(PyArrayObject *array)
{
    if (PyArray_IS_C_CONTIGUOUS(array)) {
        return 1;
    }
    int ndim = PyArray_NDIM(array);
    if (ndim == 1 && PyArray_ISALIGNED(array) &&
        PyArray_STRIDE(array, 0) > 0 &&
        PyArray_STRIDE(array, 0) % PyArray_ITEMSIZE(array) == 0) {
        return 2;
    }
    if (ndim > 0 && PyArray_ISALIGNED(array)) {
        for (int dim = 0; dim < ndim; dim++) {
            if (PyArray_STRIDE(array, dim) <= 0) {
                return 0;
            }
        }
        if (PyArray_STRIDE(array, ndim - 1) == PyArray_ITEMSIZE(array)) {
            return 3;
        }
    }
    return 0;
}

static int
numpy_jit_matmul_epilogue_inspect(
        NumpyJitPlanPayload *payload, PyObject *const *args,
        Py_ssize_t nargs, int require_lowerable, uint64_t *signature)
{
    if (nargs != 3) {
        return 0;
    }
    uint64_t hash = 1469598103934665603ULL;
    hash = numpy_jit_region_hash_u64(hash, 0x4d41544d554cULL);
    int supported = 1;
    for (Py_ssize_t i = 0; i < nargs; i++) {
        hash = numpy_jit_region_hash_u64(
            hash, (uint64_t)(uintptr_t)Py_TYPE(args[i]));
        if (PyArray_CheckExact(args[i])) {
            PyArrayObject *array = (PyArrayObject *)args[i];
            hash = numpy_jit_region_hash_u64(
                hash, (uint64_t)PyArray_TYPE(array));
            hash = numpy_jit_region_hash_u64(
                hash, (uint64_t)PyArray_NDIM(array));
            hash = numpy_jit_region_hash_u64(
                hash, (uint64_t)numpy_jit_region_layout_class(array));
        }
        else {
            double ignored;
            if (!exact_python_scalar_as_double(payload, args[i], &ignored)) {
                PyErr_Clear();
                supported = 0;
            }
        }
    }
    int a_slot = payload->region_left[0];
    int b_slot = payload->region_right[0];
    int internal_slot = payload->region_input_count;
    int second_left = payload->region_left[1];
    int second_right = payload->region_right[1];
    int internal_is_left = second_left == internal_slot;
    int other_slot = internal_is_left ? second_right : second_left;
    if (a_slot < 0 || a_slot >= nargs || b_slot < 0 || b_slot >= nargs ||
        other_slot < 0 || other_slot >= nargs ||
        (internal_is_left == (second_right == internal_slot)) ||
        !PyArray_CheckExact(args[a_slot]) ||
        !PyArray_CheckExact(args[b_slot])) {
        supported = 0;
    }
    npy_intp output_elements = 0;
    int dtype = -1;
    npy_intp m = 0;
    npy_intp n = 0;
    if (supported) {
        PyArrayObject *a = (PyArrayObject *)args[a_slot];
        PyArrayObject *b = (PyArrayObject *)args[b_slot];
        dtype = PyArray_TYPE(a);
        supported = PyArray_NDIM(a) == 2 && PyArray_NDIM(b) == 2 &&
                    dtype == PyArray_TYPE(b) &&
                    numpy_jit_is_supported_float_dtype(dtype) &&
                    PyArray_IS_C_CONTIGUOUS(a) &&
                    PyArray_IS_C_CONTIGUOUS(b) &&
                    PyArray_DIM(a, 1) == PyArray_DIM(b, 0);
        if (supported) {
            m = PyArray_DIM(a, 0);
            n = PyArray_DIM(b, 1);
            output_elements = m * n;
            if (PyArray_CheckExact(args[other_slot])) {
                PyArrayObject *other = (PyArrayObject *)args[other_slot];
                int same_shape = PyArray_NDIM(other) == 2 &&
                    PyArray_DIM(other, 0) == m &&
                    PyArray_DIM(other, 1) == n;
                int row_bias = PyArray_NDIM(other) == 1 &&
                    PyArray_DIM(other, 0) == n;
                supported = PyArray_TYPE(other) == dtype &&
                            (same_shape || row_bias) &&
                            PyArray_IS_C_CONTIGUOUS(other);
            }
        }
    }
    int errmask;
    if (_get_errmask_fast(&errmask) < 0) {
        return -1;
    }
    hash = numpy_jit_region_hash_u64(hash, (uint64_t)(uint32_t)errmask);
    for (int shift = 0; shift <= 9; shift += 3) {
        if (((errmask >> shift) & 7) > 1) {
            supported = 0;
        }
    }
    *signature = hash;
    if (!require_lowerable) {
        return 1;
    }
    /* The schedule saves one full-sized temporary allocation and its write
       traffic.  Keep very small outputs on the exact local recipe. */
    return supported && output_elements >= 64;
}

/* Extract a provider-owned property vector without baking runtime properties
   into the structural ExternalRegion.  `require_lowerable` additionally
   applies shape, alias and profitability checks used when installing a PIC
   entry. */
static int
numpy_jit_region_inspect(const PyJitExtPlan *region,
                         PyObject *const *args, Py_ssize_t nargs,
                         int require_lowerable, uint64_t *signature)
{
    if (region == NULL || region->payload == NULL || args == NULL ||
        signature == NULL) {
        return 0;
    }
    NumpyJitPlanPayload *payload = (NumpyJitPlanPayload *)region->payload;
    if (nargs != payload->region_input_count) {
        return 0;
    }
    if (payload->region_matmul_epilogue) {
        return numpy_jit_matmul_epilogue_inspect(
            payload, args, nargs, require_lowerable, signature);
    }
    uint64_t hash = 1469598103934665603ULL;
    hash = numpy_jit_region_hash_u64(hash, (uint64_t)nargs);
    hash = numpy_jit_region_hash_u64(hash, payload->region_node_count);
    PyArrayObject *shape_source = NULL;
    npy_intp element_count = -1;
    int dtype = -1;
    int itemsize = 0;
    int supported = 1;
    int shape_compatible = 1;
    for (Py_ssize_t i = 0; i < nargs; i++) {
        if (PyArray_CheckExact(args[i])) {
            PyArrayObject *array = (PyArrayObject *)args[i];
            int current_dtype = PyArray_TYPE(array);
            int layout = numpy_jit_region_layout_class(array);
            hash = numpy_jit_region_hash_u64(hash, 0x100u + (uint64_t)i);
            hash = numpy_jit_region_hash_u64(hash, (uint64_t)current_dtype);
            hash = numpy_jit_region_hash_u64(
                hash, (uint64_t)PyArray_NDIM(array));
            hash = numpy_jit_region_hash_u64(hash, (uint64_t)layout);
            hash = numpy_jit_region_hash_u64(
                hash, PyArray_ISWRITEABLE(array) ? 1u : 0u);
            if (!numpy_jit_is_supported_float_dtype(current_dtype) ||
                layout == 0) {
                supported = 0;
            }
            if (shape_source == NULL) {
                shape_source = array;
                element_count = PyArray_SIZE(array);
                dtype = current_dtype;
                itemsize = PyArray_ITEMSIZE(array);
            }
            else {
                if (current_dtype != dtype ||
                    PyArray_NDIM(array) != PyArray_NDIM(shape_source)) {
                    supported = 0;
                    shape_compatible = 0;
                }
                else {
                    for (int dim = 0; dim < PyArray_NDIM(array); dim++) {
                        if (PyArray_DIM(array, dim) !=
                            PyArray_DIM(shape_source, dim)) {
                            supported = 0;
                            shape_compatible = 0;
                            break;
                        }
                    }
                }
            }
        }
        else {
            double ignored;
            int scalar_ok = exact_python_scalar_as_double(
                payload, args[i], &ignored);
            hash = numpy_jit_region_hash_u64(hash, 0x200u + (uint64_t)i);
            hash = numpy_jit_region_hash_u64(
                hash, (uint64_t)(uintptr_t)Py_TYPE(args[i]));
            if (!scalar_ok) {
                PyErr_Clear();
                supported = 0;
            }
        }
    }
    for (int i = 0; i < payload->region_node_count; i++) {
        if (payload->region_ops[i] != NUMPY_JIT_OP_POWER) {
            continue;
        }
        int exponent_slot = payload->region_right[i];
        int is_power2 = exponent_slot >= 0 && exponent_slot < nargs &&
            numpy_jit_is_exact_power2_exponent(args[exponent_slot]);
        hash = numpy_jit_region_hash_u64(
            hash, 0x300u + (uint64_t)i);
        hash = numpy_jit_region_hash_u64(hash, is_power2 ? 2u : 0u);
        if (!is_power2) {
            supported = 0;
        }
    }
    hash = numpy_jit_region_hash_u64(hash, shape_compatible ? 1u : 0u);
    int errmask;
    if (_get_errmask_fast(&errmask) < 0) {
        return -1;
    }
    hash = numpy_jit_region_hash_u64(hash, (uint64_t)(uint32_t)errmask);
    int fp_supported = 1;
    for (int shift = 0; shift <= 9; shift += 3) {
        if (((errmask >> shift) & 7) > 1) {
            fp_supported = 0;
        }
    }
    supported &= fp_supported;

    int profitable = 0;
    if (element_count >= 0 && itemsize > 0) {
        uint64_t saved = payload->region_node_count > 1 ?
            (uint64_t)payload->region_node_count - 1u : 0u;
        uint64_t bytes = (uint64_t)element_count;
        if (bytes <= UINT64_MAX / (uint64_t)itemsize) {
            bytes *= (uint64_t)itemsize;
            if (saved == 0 || bytes > UINT64_MAX / saved) {
                bytes = UINT64_MAX;
            }
            else {
                bytes *= saved;
            }
            /* Dedicated two-op kernels have a very small fixed cost and are
               already profitable at the provider's normal 16-element
               admission threshold.  The generic expression program pays an
               extra per-node dispatch cost, so require at least two cache
               lines of eliminated traffic.  This keeps profitability tied
               to the selected lowering instead of imposing one 4 KiB cutoff
               on every region shape. */
            uint64_t lowering_cost = payload->region_generic ? 256u : 128u;
            profitable = element_count >= payload->min_elements &&
                         bytes >= lowering_cost;
        }
    }
    hash = numpy_jit_region_hash_u64(hash, profitable ? 1u : 0u);

    if (payload->inplace) {
        int target_index = payload->region_mutation_input;
        if (target_index < 0 || target_index >= nargs ||
            !PyArray_CheckExact(args[target_index]) ||
            !PyArray_ISWRITEABLE((PyArrayObject *)args[target_index])) {
            supported = 0;
        }
        else {
            PyArrayObject *target = (PyArrayObject *)args[target_index];
            for (Py_ssize_t i = 0; i < nargs; i++) {
                if (!PyArray_CheckExact(args[i]) || args[i] == (PyObject *)target) {
                    continue;
                }
                int overlap = solve_may_share_memory(
                    target, (PyArrayObject *)args[i], NPY_MAY_SHARE_BOUNDS);
                hash = numpy_jit_region_hash_u64(
                    hash, overlap == 0 ? 0u : 1u);
                if (overlap != 0) {
                    supported = 0;
                }
            }
        }
    }
    *signature = hash;
    if (!require_lowerable) {
        return 1;
    }
    return supported && profitable && shape_source != NULL;
}

static PyJitExtResult
numpy_jit_region_property_signature(const PyJitExtPlan *region,
                                    PyObject *const *args,
                                    Py_ssize_t nargs, uint64_t *signature)
{
    int inspected = numpy_jit_region_inspect(
        region, args, nargs, 0, signature);
    return inspected < 0 ? PYJIT_EXT_ERROR :
           inspected ? PYJIT_EXT_HIT : PYJIT_EXT_MISS;
}

static PyJitExtResult
numpy_jit_lowered_region_eval(void *payload_obj, PyObject *const *args,
                              Py_ssize_t nargs, PyObject **result)
{
    NumpyJitRegionLowering *lowering =
        (NumpyJitRegionLowering *)payload_obj;
    if (lowering == NULL || lowering->owner == NULL ||
        lowering->owner->region_owner_plan == NULL) {
        return PYJIT_EXT_MISS;
    }
    if (nargs != lowering->nargs) {
        return PYJIT_EXT_MISS;
    }
    for (Py_ssize_t i = 0; i < nargs; i++) {
        if (Py_TYPE(args[i]) != lowering->types[i]) {
            return PYJIT_EXT_MISS;
        }
        if ((lowering->array_mask & (1u << i)) != 0) {
            PyArrayObject *array = (PyArrayObject *)args[i];
            if (PyArray_TYPE(array) != lowering->dtypes[i] ||
                PyArray_NDIM(array) != lowering->ndims[i] ||
                numpy_jit_region_layout_class(array) != lowering->layouts[i] ||
                (PyArray_ISWRITEABLE(array) != 0) !=
                    ((lowering->writable_mask & (1u << i)) != 0)) {
                return PYJIT_EXT_MISS;
            }
        }
    }
    /* The current lowered payload is property-independent: dtype, layout,
       shape, scalar roles, aliasing, writability and FP policy are all
       validated again by the evaluator below.  Recomputing the complete
       property signature here duplicated those guards on every PIC hit.
       Keep signatures for PIC installation/negative caching, while using the
       evaluator itself as the per-entry dynamic guard. */
    return numpy_region_lowered_body(
        lowering->owner, args, nargs, result);
}

static PyJitExtLoweredRegion *
numpy_jit_specialize_region(const PyJitExtPlan *region,
                            PyObject *const *args, Py_ssize_t nargs,
                            uint64_t signature)
{
    uint64_t checked_signature;
    int inspected = numpy_jit_region_inspect(
        region, args, nargs, 1, &checked_signature);
    if (inspected <= 0 || checked_signature != signature) {
        return NULL;
    }
    NumpyJitRegionLowering *payload = PyMem_Calloc(1, sizeof(*payload));
    PyJitExtLoweredRegion *lowered = PyMem_Calloc(1, sizeof(*lowered));
    if (payload == NULL || lowered == NULL) {
        PyMem_Free(payload);
        PyMem_Free(lowered);
        PyErr_NoMemory();
        return NULL;
    }
    payload->owner = (NumpyJitPlanPayload *)region->payload;
    payload->signature = signature;
    payload->nargs = (uint8_t)nargs;
    for (Py_ssize_t i = 0; i < nargs; i++) {
        payload->types[i] = Py_TYPE(args[i]);
        if (PyArray_CheckExact(args[i])) {
            PyArrayObject *array = (PyArrayObject *)args[i];
            payload->array_mask |= (uint8_t)(1u << i);
            payload->dtypes[i] = (int8_t)PyArray_TYPE(array);
            payload->ndims[i] = (int8_t)PyArray_NDIM(array);
            payload->layouts[i] =
                (int8_t)numpy_jit_region_layout_class(array);
            if (PyArray_ISWRITEABLE(array)) {
                payload->writable_mask |= (uint8_t)(1u << i);
            }
        }
    }
    lowered->abi_version = PYJIT_EXT_ABI_VERSION;
    lowered->struct_size = sizeof(*lowered);
    lowered->property_signature = signature;
    lowered->payload = payload;
    lowered->guard_eval = numpy_jit_lowered_region_eval;
    lowered->debug_name = payload->owner->region_generic ?
        "numpy.region.program.pic" : "numpy.region.specialized.pic";
    return lowered;
}

static void
numpy_jit_free_lowered_region(PyJitExtLoweredRegion *lowered)
{
    if (lowered == NULL) {
        return;
    }
    PyMem_Free(lowered->payload);
    PyMem_Free(lowered);
}

static PyJitExtResult
numpy_region_local_fallback(NumpyJitPlanPayload *payload,
                            PyObject *const *args, Py_ssize_t nargs,
                            PyObject **result)
{
    PyJitExtResult fallback = numpy_region_fallback_eval(
        payload, args, nargs, result);
    return fallback == PYJIT_EXT_HIT ? PYJIT_EXT_COLD : fallback;
}

static PyJitExtResult
numpy_region_eval(void *payload_obj, PyObject *const *args,
                  Py_ssize_t nargs, PyObject **result)
{
    if (payload_obj == NULL || args == NULL) {
        return PYJIT_EXT_MISS;
    }
    NumpyJitPlanPayload *payload = (NumpyJitPlanPayload *)payload_obj;
    PyJitExtPlan *region = payload->region_owner_plan;
    PyJitExtProvider *provider = region != NULL ? region->provider : NULL;
    if (region == NULL || provider == NULL ||
        provider->region_property_signature == NULL ||
        provider->specialize_region == NULL) {
        return numpy_region_local_fallback(payload, args, nargs, result);
    }
    if (payload->region_execution_count != UINT32_MAX) {
        payload->region_execution_count++;
    }
    if (payload->region_execution_count <= 2) {
        return numpy_region_local_fallback(payload, args, nargs, result);
    }

    PyJitExtSiteState *site = payload->site;
    for (int i = 0; i < payload->region_pic_count; i++) {
        NumpyJitRegionPICEntry *entry = &payload->region_pic[i];
        if (entry->negative) {
            continue;
        }
        PyJitExtResult evaluated = entry->lowered->guard_eval(
            entry->lowered->payload, args, nargs, result);
        if (evaluated == PYJIT_EXT_HIT) {
            numpy_jit_region_inc_u16(
                site != NULL ? &site->region_pic_hit_count : NULL);
            if (site != NULL) {
                site->region_lowering_name = entry->lowered->debug_name;
            }
            return evaluated;
        }
        if (evaluated == PYJIT_EXT_ERROR) {
            return evaluated;
        }
    }

    /* A negative entry caused by an unsupported layout can be recognized
       much more cheaply than rebuilding the complete property vector.  Keep
       this as a guard, not a site disable: if the argument later becomes a
       supported positive-stride layout, normal PIC probing resumes. */
    for (int i = 0; i < payload->region_pic_count; i++) {
        NumpyJitRegionPICEntry *entry = &payload->region_pic[i];
        uint8_t mask = entry->unsupported_layout_mask;
        if (!entry->negative || mask == 0 || nargs > 8) {
            continue;
        }
        int same_unsupported_layout = 1;
        for (Py_ssize_t arg = 0; arg < nargs; arg++) {
            if ((mask & (1u << arg)) != 0 &&
                (!PyArray_CheckExact(args[arg]) ||
                 numpy_jit_region_layout_class(
                     (PyArrayObject *)args[arg]) != 0)) {
                same_unsupported_layout = 0;
                break;
            }
        }
        if (same_unsupported_layout) {
            numpy_jit_region_inc_u16(
                payload->site != NULL ?
                    &payload->site->region_pic_miss_count : NULL);
            return numpy_region_local_fallback(payload, args, nargs, result);
        }
    }

    uint64_t signature;
    PyJitExtResult signature_result = provider->region_property_signature(
        region, args, nargs, &signature);
    if (signature_result == PYJIT_EXT_ERROR) {
        return PYJIT_EXT_ERROR;
    }
    if (signature_result != PYJIT_EXT_HIT) {
        return numpy_region_local_fallback(payload, args, nargs, result);
    }
    for (int i = 0; i < payload->region_pic_count; i++) {
        NumpyJitRegionPICEntry *entry = &payload->region_pic[i];
        if (entry->signature != signature) {
            continue;
        }
        if (entry->negative) {
            numpy_jit_region_inc_u16(
                site != NULL ? &site->region_pic_miss_count : NULL);
            return numpy_region_local_fallback(payload, args, nargs, result);
        }
        /* A positive entry with this signature was already probed above.
           Its dynamic shape/alias/FP guard missed, so use the exact local
           recipe without attempting another specialization. */
        numpy_jit_region_inc_u16(
            site != NULL ? &site->region_pic_miss_count : NULL);
        if (site != NULL) {
            site->region_guard_fail_reason = "lowered_guard";
        }
        return numpy_region_local_fallback(payload, args, nargs, result);
    }

    numpy_jit_region_inc_u16(
        site != NULL ? &site->region_pic_miss_count : NULL);
    if (payload->region_pic_count >= PYJIT_EXT_REGION_PIC_SIZE) {
        if (!payload->region_pic_saturated) {
            payload->region_pic_saturated = 1;
            numpy_jit_region_inc_u16(
                site != NULL ? &site->region_saturated_count : NULL);
        }
        return numpy_region_local_fallback(payload, args, nargs, result);
    }

    numpy_jit_region_inc_u16(
        site != NULL ? &site->region_specialize_count : NULL);
    PyJitExtLoweredRegion *lowered = provider->specialize_region(
        region, args, nargs, signature);
    if (lowered == NULL && PyErr_Occurred()) {
        return PYJIT_EXT_ERROR;
    }
    NumpyJitRegionPICEntry *entry =
        &payload->region_pic[payload->region_pic_count++];
    entry->signature = signature;
    entry->lowered = lowered;
    entry->negative = lowered == NULL;
    if (lowered == NULL) {
        uint8_t unsupported_layout_mask = 0;
        for (Py_ssize_t i = 0; i < nargs && i < 8; i++) {
            if (PyArray_CheckExact(args[i]) &&
                numpy_jit_region_layout_class(
                    (PyArrayObject *)args[i]) == 0) {
                unsupported_layout_mask |= (uint8_t)(1u << i);
            }
        }
        entry->unsupported_layout_mask = unsupported_layout_mask;
        numpy_jit_region_inc_u16(
            site != NULL ? &site->region_specialize_decline_count : NULL);
        numpy_jit_region_inc_u16(
            site != NULL ? &site->region_negative_count : NULL);
        return numpy_region_local_fallback(payload, args, nargs, result);
    }
    PyJitExtResult evaluated = lowered->guard_eval(
        lowered->payload, args, nargs, result);
    if (evaluated == PYJIT_EXT_HIT) {
        numpy_jit_region_inc_u16(
            site != NULL ? &site->region_pic_hit_count : NULL);
        if (site != NULL) {
            site->region_lowering_name = lowered->debug_name;
        }
        return evaluated;
    }
    if (evaluated == PYJIT_EXT_ERROR) {
        return evaluated;
    }
    return numpy_region_local_fallback(payload, args, nargs, result);
}

static PyObject *
numpy_region_apply_binary(NumpyJitOp op, int inplace,
                          PyObject *left, PyObject *right)
{
    switch (op) {
        case NUMPY_JIT_OP_ADD:
            return inplace ? PyNumber_InPlaceAdd(left, right) :
                             PyNumber_Add(left, right);
        case NUMPY_JIT_OP_SUBTRACT:
            return inplace ? PyNumber_InPlaceSubtract(left, right) :
                             PyNumber_Subtract(left, right);
        case NUMPY_JIT_OP_MULTIPLY:
            return inplace ? PyNumber_InPlaceMultiply(left, right) :
                             PyNumber_Multiply(left, right);
        case NUMPY_JIT_OP_TRUE_DIVIDE:
            return inplace ? PyNumber_InPlaceTrueDivide(left, right) :
                             PyNumber_TrueDivide(left, right);
        case NUMPY_JIT_OP_POWER:
            if (inplace) {
                return PyNumber_InPlacePower(left, right, Py_None);
            }
            return PyNumber_Power(left, right, Py_None);
        case NUMPY_JIT_OP_MATMUL:
            if (inplace) {
                return PyNumber_InPlaceMatrixMultiply(left, right);
            }
            return PyNumber_MatrixMultiply(left, right);
        default:
            PyErr_SetString(PyExc_SystemError,
                            "unsupported NumPy region fallback operation");
            return NULL;
    }
}

static PyJitExtResult
numpy_region_fallback_eval(void *payload_obj, PyObject *const *args,
                           Py_ssize_t nargs, PyObject **result)
{
    if (payload_obj == NULL || args == NULL) {
        PyErr_SetString(PyExc_SystemError, "invalid NumPy region fallback");
        return PYJIT_EXT_ERROR;
    }
    NumpyJitPlanPayload *payload = (NumpyJitPlanPayload *)payload_obj;
    /* Generic elementwise regions and heterogeneous matmul regions both
       retain the provider-neutral DAG schedule.  Replay that schedule for
       cold executions and every PIC/guard miss so matmul anchors preserve
       the exact ``(A @ B) op other`` evaluation order. */
    if (payload->region_generic || payload->region_matmul_epilogue) {
        if (nargs != payload->region_input_count) {
            PyErr_SetString(
                PyExc_SystemError, "invalid generic NumPy region fallback");
            return PYJIT_EXT_ERROR;
        }
        PyObject *slots[NUMPY_JIT_KERNEL_REGION_MAX_SLOTS] = {0};
        for (Py_ssize_t i = 0; i < nargs; i++) {
            slots[i] = args[i];
        }
        for (int i = 0; i < payload->region_node_count; i++) {
            int out_slot = payload->region_input_count + i;
            int inplace = payload->inplace &&
                          i + 1 == payload->region_node_count;
            PyJitExtPlan *atomic = payload->region_atomic_plans[i];
            if (atomic != NULL && atomic->kind == PYJIT_EXT_PLAN_UNARY) {
                slots[out_slot] = PyNumber_Negative(
                    slots[payload->region_left[i]]);
            }
            else {
                slots[out_slot] = numpy_region_apply_binary(
                    payload->region_ops[i], inplace,
                    slots[payload->region_left[i]],
                    slots[payload->region_right[i]]);
            }
            if (slots[out_slot] == NULL) {
                for (int cleanup = payload->region_input_count;
                     cleanup < out_slot; cleanup++) {
                    Py_XDECREF(slots[cleanup]);
                }
                return PYJIT_EXT_ERROR;
            }
        }
        int final_slot = payload->region_output_slot;
        *result = slots[final_slot];
        for (int cleanup = payload->region_input_count;
             cleanup < final_slot; cleanup++) {
            Py_DECREF(slots[cleanup]);
        }
        return PYJIT_EXT_HIT;
    }
    if (nargs != 3) {
        PyErr_SetString(PyExc_SystemError, "invalid NumPy region fallback");
        return PYJIT_EXT_ERROR;
    }
    PyObject *first_left;
    PyObject *first_right;
    PyObject *second_other;
    numpy_region_logical_args(payload, args, &first_left, &first_right,
                              &second_other);

    PyObject *temporary = numpy_region_apply_binary(
        payload->region_first_op, 0, first_left, first_right);
    if (temporary == NULL) {
        return PYJIT_EXT_ERROR;
    }
    PyObject *out;
    if (payload->region_second_consumes_right) {
        out = numpy_region_apply_binary(
            payload->region_second_op, payload->inplace,
            second_other, temporary);
    }
    else {
        out = numpy_region_apply_binary(
            payload->region_second_op, payload->inplace,
            temporary, second_other);
    }
    Py_DECREF(temporary);
    if (out == NULL) {
        return PYJIT_EXT_ERROR;
    }
    *result = out;
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
        NULL, a, scalar, b, 0.0, NUMPY_JIT_FUSED_MUL_SCALAR_ADD_ARRAY);
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
        NULL, a, scalar, b, 0.0, NUMPY_JIT_FUSED_ADD_ARRAY_MUL_SCALAR);
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
    return numpy_jit_fused_eval(
        NULL, a, alpha, b, beta, NUMPY_JIT_FUSED_AXPBY);
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
                    if (PyJit_RegisterBinaryProvider(
                            &PyArray_Type, &PyFloatArrType_Type, oparg,
                            &numpy_jit_provider) < 0) {
                        PyErr_Clear();
                    }
                    if (PyJit_RegisterBinaryProvider(
                            &PyArray_Type, &PyDoubleArrType_Type, oparg,
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
                    if (PyJit_RegisterBinaryProvider(
                            &PyFloatArrType_Type, &PyArray_Type, oparg,
                            &numpy_jit_provider) < 0) {
                        PyErr_Clear();
                    }
                    if (PyJit_RegisterBinaryProvider(
                            &PyDoubleArrType_Type, &PyArray_Type, oparg,
                            &numpy_jit_provider) < 0) {
                        PyErr_Clear();
                    }
                    break;
            }
        }
    }

    static const int inplace_opargs[] = {
        NB_INPLACE_ADD,
        NB_INPLACE_SUBTRACT,
        NB_INPLACE_MULTIPLY,
        NB_INPLACE_TRUE_DIVIDE,
    };
    for (Py_ssize_t i = 0; i < Py_ARRAY_LENGTH(inplace_opargs); i++) {
        int oparg = inplace_opargs[i];
        if (PyJit_RegisterBinaryProvider(
                &PyArray_Type, &PyArray_Type, oparg,
                &numpy_jit_provider) < 0) {
            PyErr_Clear();
        }
        PyTypeObject *scalar_types[] = {
            &PyFloat_Type,
            &PyLong_Type,
            &PyFloatArrType_Type,
            &PyDoubleArrType_Type,
        };
        for (Py_ssize_t j = 0; j < Py_ARRAY_LENGTH(scalar_types); j++) {
            if (PyJit_RegisterBinaryProvider(
                    &PyArray_Type, scalar_types[j], oparg,
                    &numpy_jit_provider) < 0) {
                PyErr_Clear();
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
    if (PyJit_RegisterUnaryProvider(
            &PyArray_Type, PYJIT_EXT_UNARY_NEGATIVE,
            &numpy_jit_provider) < 0) {
        PyErr_Clear();
    }
    return 0;
}
