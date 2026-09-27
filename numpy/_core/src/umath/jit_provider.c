/* NumPy's CrossJIT semantic provider.  It deliberately owns no trace policy,
 * guard placement, deoptimization, or code generation. */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include <Python.h>
#include <errno.h>
#include <limits.h>
#include <opcode.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include "numpy/ufuncobject.h"
#include "jit_kernel.h"
#include "jit_provider.h"

#ifdef PYJIT_PROVIDER_ABI_VERSION

/* Semantic operation identities recognized by the NumPy provider.  The set is
 * intentionally broader than the subset accepted by v1 elementwise lowering. */
typedef enum {
    NUMPY_CROSSJIT_ADD,
    NUMPY_CROSSJIT_SUBTRACT,
    NUMPY_CROSSJIT_MULTIPLY,
    NUMPY_CROSSJIT_TRUE_DIVIDE,
    NUMPY_CROSSJIT_POWER,
    NUMPY_CROSSJIT_NEGATIVE,
    NUMPY_CROSSJIT_MINIMUM,
    NUMPY_CROSSJIT_MAXIMUM,
    NUMPY_CROSSJIT_LESS,
    NUMPY_CROSSJIT_LESS_EQUAL,
    NUMPY_CROSSJIT_LOGICAL_AND,
    NUMPY_CROSSJIT_MATMUL,
    NUMPY_CROSSJIT_ADD_OUTER,
    NUMPY_CROSSJIT_MEAN,
    NUMPY_CROSSJIT_MAXIMUM_REDUCE,
    NUMPY_CROSSJIT_BOOLEAN_SUBSCRIPT,
} NumpyCrossJitOp;

/* The high semantic-id bit records Python's in-place spelling.  Such nodes
 * remain analyzable but are mutation barriers and are never lowered. */
#define NUMPY_CROSSJIT_INPLACE_FLAG (UINT64_C(1) << 63)

/*
 * One work unit represents one eliminated intermediate-array element.  The
 * default keeps short regions on NumPy's mature ufunc loops, while allowing
 * a larger DAG to amortize dispatch and allocation costs at a smaller array
 * size.  NPY_CROSSJIT_MIN_WORK=0 disables the profitability threshold.
 */
#define NUMPY_CROSSJIT_DEFAULT_MIN_WORK UINT64_C(16384)

typedef struct {
    /* Borrowed exact callable identities owned by the initialized umath
     * module.  They remain valid for the lifetime of the static provider. */
    PyObject *add;
    PyObject *subtract;
    PyObject *multiply;
    PyObject *true_divide;
    PyObject *power;
    PyObject *negative;
    PyObject *minimum;
    PyObject *maximum;
    PyObject *less;
    PyObject *less_equal;
    PyObject *logical_and;
    PyObject *matmul;
    PyObject *mean;
    PyObject *add_outer;
    PyObject *maximum_reduce;
} NumpyCrossJitState;

/* A canonical copy of the ndarray properties used by strict v1 lowering. */
typedef struct {
    int typenum;                  /* exact NumPy dtype number */
    uint8_t ndim;                 /* rank copied from the observed array */
    uint8_t c_contiguous;         /* layout eligibility for strict lowering */
    uint8_t native_byteorder;     /* excludes swapped element interpretation */
    uint8_t aligned;              /* permits typed loads in precompiled loops */
    npy_intp dimensions[NPY_MAXDIMS]; /* exact guarded iteration domain */
    npy_intp strides[NPY_MAXDIMS];    /* byte strides, including broadcasts */
} NumpyCrossJitArrayFact;

enum {
    NUMPY_CROSSJIT_ASSUME_ARRAY = 1,
    NUMPY_CROSSJIT_ASSUME_SCALAR = 2,
};

typedef enum {
    NUMPY_CROSSJIT_CONTIGUOUS_SAME_SHAPE = 1,
    NUMPY_CROSSJIT_CONTIGUOUS_BROADCAST = 2,
    NUMPY_CROSSJIT_ZERO_STRIDE = 3,
    NUMPY_CROSSJIT_INNER_CONTIGUOUS = 4,
    NUMPY_CROSSJIT_GENERAL_STRIDED = 5,
} NumpyCrossJitIterationClass;

typedef enum {
    NUMPY_CROSSJIT_REJECT_NONE = 0,
    NUMPY_CROSSJIT_REJECT_LIVE_INS,
    NUMPY_CROSSJIT_REJECT_INPUT_KIND,
    NUMPY_CROSSJIT_REJECT_DTYPE_LAYOUT,
    NUMPY_CROSSJIT_REJECT_DTYPE_MISMATCH,
    NUMPY_CROSSJIT_REJECT_NO_ARRAY,
    NUMPY_CROSSJIT_REJECT_BROADCAST,
    NUMPY_CROSSJIT_REJECT_SCALAR_PROMOTION,
    NUMPY_CROSSJIT_REJECT_SCALAR_RANGE,
    NUMPY_CROSSJIT_REJECT_NODE_COUNT,
    NUMPY_CROSSJIT_REJECT_UNPROFITABLE,
    NUMPY_CROSSJIT_REJECT_KERNEL,
    NUMPY_CROSSJIT_REJECT_VALUE_MAPPING,
    NUMPY_CROSSJIT_REJECT_OTHER,
    NUMPY_CROSSJIT_REJECT_COUNT,
} NumpyCrossJitRejectReason;

typedef enum {
    NUMPY_CROSSJIT_DTYPE_OTHER = 0,
    NUMPY_CROSSJIT_DTYPE_FLOAT32,
    NUMPY_CROSSJIT_DTYPE_FLOAT64,
    NUMPY_CROSSJIT_DTYPE_INT32,
    NUMPY_CROSSJIT_DTYPE_INT64,
    NUMPY_CROSSJIT_DTYPE_COUNT,
} NumpyCrossJitDtypeClass;

typedef struct {
    uint64_t element_count;
    uint64_t profitability_score;
    uint64_t profitability_threshold;
    uint8_t region_length;
    uint8_t layout_class;
    uint8_t dtype_class;
    uint8_t operand_count;
    uint8_t result_count;
    uint8_t internal_value_count;
    uint8_t eliminable_materialization_count;
    uint8_t guard_count;
    uint8_t unique_guard_count;
    uint8_t prepare_decision;
    uint8_t rejection_reason;
} NumpyCrossJitCandidateDiagnostic;

#define NUMPY_CROSSJIT_MAX_DIAGNOSTIC_CANDIDATES 1024

typedef struct {
    uint64_t nodes_recognized;
    uint64_t regions_prepare_called;
    uint64_t regions_provider_accepted;
    uint64_t regions_rejected_unsupported_node;
    uint64_t regions_rejected_unsupported_contract;
    uint64_t regions_rejected_unprofitable;
    uint64_t regions_rejected_other;
    uint64_t guard_plans_lowered;
    uint64_t execution_plans_lowered;
    uint64_t regions_executed;
    uint64_t guard_hit;
    uint64_t guard_miss;
    uint64_t guard_error;
    uint64_t exec_hit;
    uint64_t exec_miss;
    uint64_t exec_error;
    uint64_t fixed_fast_path_hit;
    uint64_t fixed_generic_path_hit;
    uint64_t output_allocations;
    uint64_t internal_allocations;
    uint64_t guard_plans_destroyed;
    uint64_t execution_plans_destroyed;
    uint64_t guard_metadata_reads;
    uint64_t guard_checks;
    uint64_t prepare_time_ns;
    uint64_t guard_lower_time_ns;
    uint64_t execution_lower_time_ns;
    uint64_t guard_time_ns;
    uint64_t region_time_ns;
    uint64_t allocation_time_ns;
    uint64_t kernel_time_ns;
    uint64_t region_length_hist[NUMPY_CROSSJIT_MAX_NODES + 1];
    uint64_t layout_hist[NUMPY_CROSSJIT_GENERAL_STRIDED + 1];
    uint64_t dtype_hist[NUMPY_CROSSJIT_DTYPE_COUNT];
    uint64_t rejection_hist[NUMPY_CROSSJIT_REJECT_COUNT];
    NumpyCrossJitCandidateDiagnostic last;
    NumpyCrossJitCandidateDiagnostic
        candidates[NUMPY_CROSSJIT_MAX_DIAGNOSTIC_CANDIDATES];
    uint64_t candidate_count;
    uint64_t candidate_records_dropped;
    int diagnostics_enabled;
    int timing_enabled;
} NumpyCrossJitStats;

static NumpyCrossJitState crossjit_state;
static PyJitProviderKey crossjit_keys[64];
static PyJitProviderSpec crossjit_provider;
static NumpyCrossJitStats crossjit_stats;

static uint8_t
numpy_crossjit_dtype_class(int typenum)
{
    if (typenum == NPY_FLOAT) {
        return NUMPY_CROSSJIT_DTYPE_FLOAT32;
    }
    if (typenum == NPY_DOUBLE) {
        return NUMPY_CROSSJIT_DTYPE_FLOAT64;
    }
    if (typenum == NPY_INT && sizeof(npy_int) == 4) {
        return NUMPY_CROSSJIT_DTYPE_INT32;
    }
    if ((typenum == NPY_LONG && sizeof(npy_long) == 8) ||
        (typenum == NPY_LONGLONG && sizeof(npy_longlong) == 8)) {
        return NUMPY_CROSSJIT_DTYPE_INT64;
    }
    return NUMPY_CROSSJIT_DTYPE_OTHER;
}

static uint64_t
numpy_crossjit_clock_ns(void)
{
    if (!crossjit_stats.timing_enabled) {
        return 0;
    }
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return 0;
    }
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) +
        (uint64_t)now.tv_nsec;
}

static void
numpy_crossjit_record_elapsed(uint64_t *total, uint64_t started)
{
    if (started == 0) {
        return;
    }
    uint64_t ended = numpy_crossjit_clock_ns();
    if (ended >= started) {
        *total += ended - started;
    }
}

static void
numpy_crossjit_record_candidate(void)
{
    if (!crossjit_stats.diagnostics_enabled) {
        return;
    }
    if (crossjit_stats.candidate_count >=
            NUMPY_CROSSJIT_MAX_DIAGNOSTIC_CANDIDATES) {
        crossjit_stats.candidate_records_dropped++;
        return;
    }
    crossjit_stats.candidates[crossjit_stats.candidate_count++] =
        crossjit_stats.last;
}

/* Reads the experimental profitability override without caching test state. */
static uint64_t
numpy_crossjit_min_work(void)
{
    const char *text = Py_GETENV("NPY_CROSSJIT_MIN_WORK");
    if (text == NULL || *text < '0' || *text > '9') {
        return NUMPY_CROSSJIT_DEFAULT_MIN_WORK;
    }
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return NUMPY_CROSSJIT_DEFAULT_MIN_WORK;
    }
    return (uint64_t)value;
}

/* Computes the guarded broadcast-domain size with saturating arithmetic. */
static uint64_t
numpy_crossjit_domain_elements(const NumpyCrossJitArrayFact *domain)
{
    uint64_t elements = 1;
    for (uint8_t axis = 0; axis < domain->ndim; axis++) {
        uint64_t dimension = (uint64_t)domain->dimensions[axis];
        if (dimension == 0) {
            return 0;
        }
        if (elements > UINT64_MAX / dimension) {
            return UINT64_MAX;
        }
        elements *= dimension;
    }
    return elements;
}

/* Runtime guards copy every property needed to validate a region live-in. */
struct PyJitGuardPlan {
    uint8_t input_count;          /* semantic live-ins in Host order */
    uint8_t iteration_class;      /* lowering-time domain specialization */
    uint16_t scalar_mask;         /* Python scalar positions */
    uint16_t int_scalar_mask;     /* scalar positions needing exact int checks */
    NumpyCrossJitArrayFact domain; /* common output dtype, shape, and layout */
    NumpyCrossJitArrayFact inputs[NUMPY_CROSSJIT_MAX_INPUTS]; /* array contracts */
};

/*
 * The execution plan is a provider-owned, allocation-free elementwise DAG.
 * node_count selects one of the precompiled two-to-six-node loop bodies.
 */
struct PyJitExecPlan {
    struct PyJitGuardPlan guards; /* private copy keeps execution self-contained */
    uint8_t node_count;           /* selects a two-to-six-node loop body */
    uint8_t output_slot;          /* final virtual slot written to the ndarray */
    NumpyCrossJitKernelNode nodes[NUMPY_CROSSJIT_MAX_NODES]; /* topological DAG */
};

/* C_NumPy is created only after HostRegionProof.  It owns every dtype,
 * shape, stride, broadcast, scalar, guard-class and kernel decision. */
typedef struct {
    struct PyJitExecPlan execution;
    PyJitAssumption assumptions[NUMPY_CROSSJIT_MAX_INPUTS];
    const PyJitValueRef *dependencies[NUMPY_CROSSJIT_MAX_INPUTS][1];
    uint8_t scalar_keys[NUMPY_CROSSJIT_MAX_INPUTS];
} NumpyCrossJitContract;

/* Captures only stable exact-ndarray layout data; subclasses stay generic. */
static int
numpy_crossjit_array_fact(PyArrayObject *array, NumpyCrossJitArrayFact *fact)
{
    if (!PyArray_CheckExact((PyObject *)array)) {
        return 0;
    }
    memset(fact, 0, sizeof(*fact));
    fact->typenum = PyArray_TYPE(array);
    fact->ndim = (uint8_t)PyArray_NDIM(array);
    fact->c_contiguous = PyArray_IS_C_CONTIGUOUS(array);
    fact->native_byteorder = PyArray_ISNOTSWAPPED(array);
    fact->aligned = PyArray_ISALIGNED(array);
    memcpy(fact->dimensions, PyArray_DIMS(array),
           (size_t)fact->ndim * sizeof(npy_intp));
    memcpy(fact->strides, PyArray_STRIDES(array),
           (size_t)fact->ndim * sizeof(npy_intp));
    return 1;
}

/* Maps a Python/NumPy operation to semantic identity without choosing fusion. */
static PyJitResult
numpy_crossjit_recognize(const PyJitRecognizeRequest *request,
                         PyJitRecognition *recognition)
{
    NumpyCrossJitOp op;
    int inplace = 0;
    if (request->kind == PYJIT_OPERATION_BINARY) {
        switch (request->operation) {
            case NB_ADD: op = NUMPY_CROSSJIT_ADD; break;
            case NB_INPLACE_ADD: op = NUMPY_CROSSJIT_ADD; inplace = 1; break;
            case NB_SUBTRACT: op = NUMPY_CROSSJIT_SUBTRACT; break;
            case NB_INPLACE_SUBTRACT: op = NUMPY_CROSSJIT_SUBTRACT; inplace = 1; break;
            case NB_MULTIPLY: op = NUMPY_CROSSJIT_MULTIPLY; break;
            case NB_INPLACE_MULTIPLY: op = NUMPY_CROSSJIT_MULTIPLY; inplace = 1; break;
            case NB_TRUE_DIVIDE: op = NUMPY_CROSSJIT_TRUE_DIVIDE; break;
            case NB_INPLACE_TRUE_DIVIDE: op = NUMPY_CROSSJIT_TRUE_DIVIDE; inplace = 1; break;
            case NB_POWER: op = NUMPY_CROSSJIT_POWER; break;
            case NB_INPLACE_POWER: op = NUMPY_CROSSJIT_POWER; inplace = 1; break;
            case NB_MATRIX_MULTIPLY: op = NUMPY_CROSSJIT_MATMUL; break;
            case NB_INPLACE_MATRIX_MULTIPLY: op = NUMPY_CROSSJIT_MATMUL; inplace = 1; break;
            default: return PYJIT_RESULT_MISS;
        }
    }
    else if (request->kind == PYJIT_OPERATION_UNARY &&
             request->operation == PYJIT_UNARY_NEGATIVE) {
        op = NUMPY_CROSSJIT_NEGATIVE;
    }
    else if (request->kind == PYJIT_OPERATION_COMPARE) {
        if (request->operation == Py_LT) {
            op = NUMPY_CROSSJIT_LESS;
        }
        else if (request->operation == Py_LE) {
            op = NUMPY_CROSSJIT_LESS_EQUAL;
        }
        else {
            return PYJIT_RESULT_MISS;
        }
    }
    else if (request->kind == PYJIT_OPERATION_CALL && request->callable != NULL) {
        PyObject *callable = request->callable;
        if (callable == crossjit_state.add) op = NUMPY_CROSSJIT_ADD;
        else if (callable == crossjit_state.subtract) op = NUMPY_CROSSJIT_SUBTRACT;
        else if (callable == crossjit_state.multiply) op = NUMPY_CROSSJIT_MULTIPLY;
        else if (callable == crossjit_state.true_divide) op = NUMPY_CROSSJIT_TRUE_DIVIDE;
        else if (callable == crossjit_state.power) op = NUMPY_CROSSJIT_POWER;
        else if (callable == crossjit_state.negative) op = NUMPY_CROSSJIT_NEGATIVE;
        else if (callable == crossjit_state.minimum) op = NUMPY_CROSSJIT_MINIMUM;
        else if (callable == crossjit_state.maximum) op = NUMPY_CROSSJIT_MAXIMUM;
        else if (callable == crossjit_state.less) op = NUMPY_CROSSJIT_LESS;
        else if (callable == crossjit_state.less_equal) op = NUMPY_CROSSJIT_LESS_EQUAL;
        else if (callable == crossjit_state.logical_and) op = NUMPY_CROSSJIT_LOGICAL_AND;
        else if (callable == crossjit_state.matmul) op = NUMPY_CROSSJIT_MATMUL;
        else if (callable == crossjit_state.add_outer) op = NUMPY_CROSSJIT_ADD_OUTER;
        else if (callable == crossjit_state.mean) op = NUMPY_CROSSJIT_MEAN;
        else if (callable == crossjit_state.maximum_reduce) op = NUMPY_CROSSJIT_MAXIMUM_REDUCE;
        else return PYJIT_RESULT_MISS;
    }
    else {
        return PYJIT_RESULT_MISS;
    }
    recognition->semantic_id = ((uint64_t)op + 1) |
        (inplace ? NUMPY_CROSSJIT_INPLACE_FLAG : 0);
    crossjit_stats.nodes_recognized++;
    return PYJIT_RESULT_HIT;
}

/* Projects NumPy domain semantics to S_H without constructing C_NumPy.
 * Shape, strides, broadcasting and kernel selection are deliberately delayed
 * until numpy_crossjit_prepare_region(). */
static PyJitResult
numpy_crossjit_project_semantics(const PyJitRecognizeRequest *request,
                                 const PyJitRecognition *recognition,
                                 PyJitSemanticSummary *summary)
{
    int inplace =
        (recognition->semantic_id & NUMPY_CROSSJIT_INPLACE_FLAG) != 0;
    NumpyCrossJitOp op = (NumpyCrossJitOp)(
        (recognition->semantic_id & ~NUMPY_CROSSJIT_INPLACE_FLAG) - 1);
    int has_array = 0;
    int may_call_host = 0;
    for (uint8_t i = 0; i < request->operand_count; i++) {
        PyObject *object = PyJit_ValueGetObservedObject(
            request->operands[i]);
        if (object != NULL && PyArray_CheckExact(object)) {
            has_array = 1;
            if (PyArray_TYPE((PyArrayObject *)object) == NPY_OBJECT) {
                may_call_host = 1;
            }
        }
    }

    int elementwise =
        op == NUMPY_CROSSJIT_ADD ||
        op == NUMPY_CROSSJIT_SUBTRACT ||
        op == NUMPY_CROSSJIT_MULTIPLY ||
        op == NUMPY_CROSSJIT_TRUE_DIVIDE ||
        op == NUMPY_CROSSJIT_NEGATIVE ||
        op == NUMPY_CROSSJIT_MINIMUM ||
        op == NUMPY_CROSSJIT_MAXIMUM;
    *summary = (PyJitSemanticSummary){
        .struct_size = sizeof(PyJitSemanticSummary),
        .mutation = inplace ? PYJIT_MUTATION_ARGUMENT :
                              PYJIT_MUTATION_NONE,
        .result_alias = inplace ? PYJIT_ALIAS_MAY_ALIAS_ARGUMENT :
                                  PYJIT_ALIAS_FRESH,
        .materialization = !inplace && elementwise && !may_call_host ?
            PYJIT_MATERIALIZATION_VIRTUALIZABLE :
            PYJIT_MATERIALIZATION_REQUIRED,
        .replayability = !inplace && !may_call_host ?
            PYJIT_REPLAYABLE : PYJIT_NOT_REPLAYABLE,
        .may_raise = 1,
        .may_call_host = (uint8_t)may_call_host,
        .exact_result_type = has_array ? &PyArray_Type : NULL,
    };
    return PYJIT_RESULT_HIT;
}

/* Returns whether a dtype is implemented by the strict precompiled loop. */
static int
numpy_crossjit_supported_dtype(int typenum)
{
    return typenum == NPY_FLOAT || typenum == NPY_DOUBLE ||
           (typenum == NPY_INT && sizeof(npy_int) == 4) ||
           (typenum == NPY_LONG && sizeof(npy_long) == 8) ||
           (typenum == NPY_LONGLONG && sizeof(npy_longlong) == 8);
}

/* Emits opt-in provider diagnostics without exposing lowering policy to Host. */
static int
numpy_crossjit_lowering_reject(NumpyCrossJitRejectReason code,
                               const char *reason)
{
    if (crossjit_stats.diagnostics_enabled) {
        crossjit_stats.last.rejection_reason = (uint8_t)code;
    }
    if (Py_GETENV("PYTHON_JIT_CROSSJIT_DEBUG") != NULL) {
        fprintf(stderr, "NumPy CrossJIT lowering rejected: %s\n", reason);
    }
    return 0;
}

/* Accepts only Python integers whose conversion to the floating domain is
 * exact, avoiding value-dependent weak-promotion or double-rounding cases. */
static int
numpy_crossjit_safe_int_scalar(PyObject *object, int typenum)
{
    if (!PyLong_CheckExact(object)) {
        return 0;
    }
    long long value = PyLong_AsLongLong(object);
    if (value == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }
    long long limit = typenum == NPY_FLOAT ? INT64_C(16777216) :
        INT64_C(9007199254740992);
    return (typenum == NPY_FLOAT || typenum == NPY_DOUBLE) &&
           value >= -limit && value <= limit;
}

/* Maps a semantic operation to the provider's precompiled kernel opcode. */
static int
numpy_crossjit_kernel_op(uint64_t semantic_id, int typenum, uint8_t *kernel_op)
{
    if (semantic_id & NUMPY_CROSSJIT_INPLACE_FLAG) {
        return 0;
    }
    NumpyCrossJitOp op = (NumpyCrossJitOp)(semantic_id - 1);
    switch (op) {
        case NUMPY_CROSSJIT_ADD:
            *kernel_op = NUMPY_CROSSJIT_KERNEL_ADD;
            return 1;
        case NUMPY_CROSSJIT_SUBTRACT:
            *kernel_op = NUMPY_CROSSJIT_KERNEL_SUBTRACT;
            return 1;
        case NUMPY_CROSSJIT_MULTIPLY:
            *kernel_op = NUMPY_CROSSJIT_KERNEL_MULTIPLY;
            return 1;
        case NUMPY_CROSSJIT_TRUE_DIVIDE:
            if (typenum == NPY_FLOAT || typenum == NPY_DOUBLE) {
                *kernel_op = NUMPY_CROSSJIT_KERNEL_TRUE_DIVIDE;
                return 1;
            }
            return 0;
        case NUMPY_CROSSJIT_NEGATIVE:
            *kernel_op = NUMPY_CROSSJIT_KERNEL_NEGATIVE;
            return 1;
        case NUMPY_CROSSJIT_MINIMUM:
            if (typenum == NPY_FLOAT || typenum == NPY_DOUBLE) {
                *kernel_op = NUMPY_CROSSJIT_KERNEL_MINIMUM;
                return 1;
            }
            return 0;
        case NUMPY_CROSSJIT_MAXIMUM:
            if (typenum == NPY_FLOAT || typenum == NPY_DOUBLE) {
                *kernel_op = NUMPY_CROSSJIT_KERNEL_MAXIMUM;
                return 1;
            }
            return 0;
        default:
            return 0;
    }
}

/* Copies and cross-checks the exact array contract for every region input. */
static int
numpy_crossjit_build_guards(const PyJitRegionView *region,
                            struct PyJitGuardPlan *guards)
{
    memset(guards, 0, sizeof(*guards));
    guards->input_count = PyJit_RegionGetInputCount(region);
    if (guards->input_count == 0 ||
        guards->input_count > NUMPY_CROSSJIT_MAX_INPUTS) {
        return numpy_crossjit_lowering_reject(
            NUMPY_CROSSJIT_REJECT_LIVE_INS, "invalid live-in count");
    }
    int have_domain = 0;
    uint8_t domain_ndim = 0;
    for (uint8_t i = 0; i < guards->input_count; i++) {
        const PyJitRegionValue *input = PyJit_RegionGetInput(region, i);
        PyObject *object = input == NULL ? NULL :
            PyJit_ValueGetObservedObject(input->value);
        if (object != NULL && PyFloat_CheckExact(object)) {
            guards->scalar_mask |= (uint16_t)(1u << i);
            continue;
        }
        if (object != NULL && PyLong_CheckExact(object)) {
            guards->scalar_mask |= (uint16_t)(1u << i);
            guards->int_scalar_mask |= (uint16_t)(1u << i);
            continue;
        }
        if (object == NULL || !PyArray_CheckExact(object) ||
            !numpy_crossjit_array_fact(
                (PyArrayObject *)object, &guards->inputs[i])) {
            return numpy_crossjit_lowering_reject(
                NUMPY_CROSSJIT_REJECT_INPUT_KIND,
                "live-in is neither an exact ndarray nor an exact float");
        }
        NumpyCrossJitArrayFact *fact = &guards->inputs[i];
        if (!fact->native_byteorder || !fact->aligned ||
            !numpy_crossjit_supported_dtype(fact->typenum)) {
            return numpy_crossjit_lowering_reject(
                NUMPY_CROSSJIT_REJECT_DTYPE_LAYOUT,
                "array live-in has unsupported layout or dtype");
        }
        if (!have_domain) {
            guards->domain = *fact;
            have_domain = 1;
        }
        else if (fact->typenum != guards->domain.typenum) {
            return numpy_crossjit_lowering_reject(
                NUMPY_CROSSJIT_REJECT_DTYPE_MISMATCH,
                "array live-ins have different dtypes");
        }
        if (fact->ndim > domain_ndim) {
            domain_ndim = fact->ndim;
        }
    }
    /* Python floats participate in NumPy's weak scalar promotion for the
     * supported floating domains; integer-array promotion is left generic. */
    if (!have_domain) {
        return numpy_crossjit_lowering_reject(
            NUMPY_CROSSJIT_REJECT_NO_ARRAY, "region has no array domain");
    }
    /* Build NumPy's trailing-axis broadcast domain.  Original input facts
     * remain untouched so the guard still checks exact runtime layouts. */
    guards->domain.ndim = domain_ndim;
    guards->domain.c_contiguous = 1;
    memset(guards->domain.dimensions, 0,
           sizeof(guards->domain.dimensions));
    memset(guards->domain.strides, 0, sizeof(guards->domain.strides));
    for (uint8_t axis = 0; axis < domain_ndim; axis++) {
        npy_intp dimension = 1;
        for (uint8_t i = 0; i < guards->input_count; i++) {
            if (guards->scalar_mask & (1u << i)) {
                continue;
            }
            NumpyCrossJitArrayFact *input = &guards->inputs[i];
            int input_axis = (int)axis - ((int)domain_ndim - input->ndim);
            npy_intp candidate = input_axis < 0 ? 1 :
                input->dimensions[input_axis];
            if (candidate != 1 && dimension != 1 && candidate != dimension) {
                return numpy_crossjit_lowering_reject(
                    NUMPY_CROSSJIT_REJECT_BROADCAST,
                    "array live-ins cannot be broadcast together");
            }
            if (candidate != 1) {
                dimension = candidate;
            }
        }
        guards->domain.dimensions[axis] = dimension;
    }
    if (guards->scalar_mask != 0 && guards->domain.typenum != NPY_FLOAT &&
        guards->domain.typenum != NPY_DOUBLE) {
        return numpy_crossjit_lowering_reject(
            NUMPY_CROSSJIT_REJECT_SCALAR_PROMOTION,
            "Python float would promote the integer array domain");
    }
    for (uint8_t i = 0; i < guards->input_count; i++) {
        if (guards->int_scalar_mask & (1u << i)) {
            const PyJitRegionValue *input = PyJit_RegionGetInput(region, i);
            PyObject *object = input == NULL ? NULL :
                PyJit_ValueGetObservedObject(input->value);
            if (object == NULL || !numpy_crossjit_safe_int_scalar(
                    object, guards->domain.typenum)) {
                return numpy_crossjit_lowering_reject(
                    NUMPY_CROSSJIT_REJECT_SCALAR_RANGE,
                    "Python int is not exact in the floating domain");
            }
        }
    }
    int all_contiguous = 1;
    int any_broadcast = 0;
    int inner_contiguous = domain_ndim != 0;
    for (uint8_t i = 0; i < guards->input_count; i++) {
        if (guards->scalar_mask & (1u << i)) {
            continue;
        }
        NumpyCrossJitArrayFact *input = &guards->inputs[i];
        const PyJitRegionValue *region_input =
            PyJit_RegionGetInput(region, i);
        PyArrayObject *array = (PyArrayObject *)
            PyJit_ValueGetObservedObject(region_input->value);
        all_contiguous &= input->c_contiguous;
        int leading = (int)domain_ndim - input->ndim;
        for (uint8_t axis = 0; axis < domain_ndim; axis++) {
            int input_axis = (int)axis - leading;
            if (input_axis < 0 ||
                (input->dimensions[input_axis] == 1 &&
                 guards->domain.dimensions[axis] != 1)) {
                any_broadcast = 1;
            }
        }
        if (input->ndim != 0 &&
            input->dimensions[input->ndim - 1] > 1 &&
            input->strides[input->ndim - 1] !=
                PyArray_ITEMSIZE(array)) {
            inner_contiguous = 0;
        }
    }
    if (!any_broadcast && all_contiguous) {
        guards->iteration_class =
            NUMPY_CROSSJIT_CONTIGUOUS_SAME_SHAPE;
    }
    else if (any_broadcast && all_contiguous) {
        guards->iteration_class = NUMPY_CROSSJIT_CONTIGUOUS_BROADCAST;
    }
    else if (any_broadcast) {
        guards->iteration_class = NUMPY_CROSSJIT_ZERO_STRIDE;
    }
    else if (inner_contiguous) {
        guards->iteration_class = NUMPY_CROSSJIT_INNER_CONTIGUOUS;
    }
    else {
        guards->iteration_class = NUMPY_CROSSJIT_GENERAL_STRIDED;
    }
    return 1;
}

/* Maps a region value to a live-in or an earlier node's virtual output. */
static int
numpy_crossjit_value_slot(const PyJitRegionView *region, uint8_t node_index,
                          const PyJitValueRef *value)
{
    uint8_t input_count = PyJit_RegionGetInputCount(region);
    for (uint8_t i = 0; i < input_count; i++) {
        const PyJitRegionValue *input = PyJit_RegionGetInput(region, i);
        if (input != NULL && input->value == value) {
            return i;
        }
    }
    for (uint8_t i = 0; i < node_index; i++) {
        const PyJitNodeRef *producer = PyJit_RegionGetNode(region, i);
        if (PyJit_NodeGetResult(producer, 0) == value) {
            return input_count + i;
        }
    }
    return -1;
}

/* Builds the immutable provider program after Host validation succeeds. */
static PyJitPrepareDecision
numpy_crossjit_build_exec(const PyJitRegionView *region,
                          struct PyJitExecPlan *plan)
{
    memset(plan, 0, sizeof(*plan));
    if (crossjit_stats.diagnostics_enabled) {
        memset(&crossjit_stats.last, 0, sizeof(crossjit_stats.last));
    }
    if (!numpy_crossjit_build_guards(region, &plan->guards)) {
        return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
    }
    plan->node_count = PyJit_RegionGetNodeCount(region);
    if (crossjit_stats.diagnostics_enabled) {
        crossjit_stats.last.region_length = plan->node_count;
        crossjit_stats.last.layout_class = plan->guards.iteration_class;
        crossjit_stats.last.dtype_class =
            numpy_crossjit_dtype_class(plan->guards.domain.typenum);
        crossjit_stats.last.operand_count = plan->guards.input_count;
        crossjit_stats.last.result_count = PyJit_RegionGetResultCount(region);
        crossjit_stats.last.guard_count = plan->guards.input_count;
        if (plan->node_count <= NUMPY_CROSSJIT_MAX_NODES) {
            crossjit_stats.region_length_hist[plan->node_count]++;
        }
        if (plan->guards.iteration_class <=
                NUMPY_CROSSJIT_GENERAL_STRIDED) {
            crossjit_stats.layout_hist[plan->guards.iteration_class]++;
        }
        if (crossjit_stats.last.dtype_class < NUMPY_CROSSJIT_DTYPE_COUNT) {
            crossjit_stats.dtype_hist[crossjit_stats.last.dtype_class]++;
        }
    }
    if (plan->node_count < 2 || plan->node_count > 6) {
        numpy_crossjit_lowering_reject(
            NUMPY_CROSSJIT_REJECT_NODE_COUNT, "unsupported node count");
        return PYJIT_PREPARE_UNSUPPORTED_NODE;
    }
    uint64_t elements = numpy_crossjit_domain_elements(&plan->guards.domain);
    uint64_t eliminated = (uint64_t)(plan->node_count - 1);
    uint64_t work = elements > UINT64_MAX / eliminated ? UINT64_MAX :
        elements * eliminated;
    uint64_t min_work = numpy_crossjit_min_work();
    if (crossjit_stats.diagnostics_enabled) {
        crossjit_stats.last.element_count = elements;
        crossjit_stats.last.internal_value_count = (uint8_t)eliminated;
        crossjit_stats.last.eliminable_materialization_count =
            (uint8_t)eliminated;
        crossjit_stats.last.profitability_score = work;
        crossjit_stats.last.profitability_threshold = min_work;
    }
    if (work < min_work) {
        if (Py_GETENV("PYTHON_JIT_CROSSJIT_DEBUG") != NULL) {
            fprintf(stderr,
                    "NumPy CrossJIT rejected short region: nodes=%u, "
                    "elements=%llu, work=%llu, minimum=%llu\n",
                    plan->node_count, (unsigned long long)elements,
                    (unsigned long long)work, (unsigned long long)min_work);
        }
        numpy_crossjit_lowering_reject(
            NUMPY_CROSSJIT_REJECT_UNPROFITABLE,
            "iteration domain is below the provider profitability threshold");
        return PYJIT_PREPARE_UNPROFITABLE;
    }
    int typenum = plan->guards.domain.typenum;
    for (uint8_t i = 0; i < plan->node_count; i++) {
        const PyJitNodeRef *node = PyJit_RegionGetNode(region, i);
        NumpyCrossJitKernelNode *lowered = &plan->nodes[i];
        if (node == NULL ||
            !numpy_crossjit_kernel_op(PyJit_NodeGetSemanticId(node), typenum,
                                      &lowered->op)) {
            numpy_crossjit_lowering_reject(
                NUMPY_CROSSJIT_REJECT_KERNEL,
                "semantic operation has no strict kernel");
            return PYJIT_PREPARE_UNSUPPORTED_NODE;
        }
        int left = numpy_crossjit_value_slot(
            region, i, PyJit_NodeGetOperand(node, 0));
        const PyJitValueRef *right_value = PyJit_NodeGetOperand(node, 1);
        int right = right_value == NULL ? left :
            numpy_crossjit_value_slot(region, i, right_value);
        if (left < 0 || right < 0 ||
            left >= NUMPY_CROSSJIT_MAX_SLOTS ||
            right >= NUMPY_CROSSJIT_MAX_SLOTS) {
            numpy_crossjit_lowering_reject(
                NUMPY_CROSSJIT_REJECT_VALUE_MAPPING,
                "node operand is not a live-in or earlier virtual value");
            return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
        }
        lowered->left = (uint8_t)left;
        lowered->right = (uint8_t)right;
        lowered->output = plan->guards.input_count + i;
    }
    plan->output_slot = plan->guards.input_count + plan->node_count - 1;
    if (Py_GETENV("PYTHON_JIT_CROSSJIT_DEBUG") != NULL) {
        fprintf(stderr, "NumPy CrossJIT accepted %u-node kernel:",
                plan->node_count);
        for (uint8_t i = 0; i < plan->node_count; i++) {
            fprintf(stderr, " %u", plan->nodes[i].op);
        }
        fputc('\n', stderr);
    }
    return PYJIT_PREPARE_ACCEPT;
}

/* Builds C_NumPy only after the Host has supplied a proven region. */
static PyJitResult
numpy_crossjit_prepare_region(const PyJitRegionView *region,
                              PyJitPrepareResult *result)
{
    uint64_t started = numpy_crossjit_clock_ns();
    crossjit_stats.regions_prepare_called++;
    if (result == NULL || result->struct_size != sizeof(*result)) {
        crossjit_stats.regions_rejected_other++;
        PyErr_SetString(PyExc_ValueError,
                        "invalid NumPy CrossJIT prepare result");
        numpy_crossjit_record_elapsed(&crossjit_stats.prepare_time_ns,
                                      started);
        return PYJIT_RESULT_ERROR;
    }
    NumpyCrossJitContract *contract =
        PyMem_Calloc(1, sizeof(*contract));
    if (contract == NULL) {
        crossjit_stats.regions_rejected_other++;
        PyErr_NoMemory();
        numpy_crossjit_record_elapsed(&crossjit_stats.prepare_time_ns,
                                      started);
        return PYJIT_RESULT_ERROR;
    }
    PyJitPrepareDecision decision = numpy_crossjit_build_exec(
        region, &contract->execution);
    result->decision = decision;
    if (crossjit_stats.diagnostics_enabled) {
        crossjit_stats.last.prepare_decision = (uint8_t)decision;
    }
    if (decision != PYJIT_PREPARE_ACCEPT) {
        if (crossjit_stats.diagnostics_enabled &&
                crossjit_stats.last.rejection_reason <
                NUMPY_CROSSJIT_REJECT_COUNT) {
            crossjit_stats.rejection_hist[
                crossjit_stats.last.rejection_reason]++;
        }
        if (decision == PYJIT_PREPARE_UNPROFITABLE) {
            crossjit_stats.regions_rejected_unprofitable++;
        }
        else if (decision == PYJIT_PREPARE_UNSUPPORTED_NODE) {
            crossjit_stats.regions_rejected_unsupported_node++;
        }
        else if (decision == PYJIT_PREPARE_UNSUPPORTED_CONTRACT) {
            crossjit_stats.regions_rejected_unsupported_contract++;
        }
        else {
            crossjit_stats.regions_rejected_other++;
        }
        numpy_crossjit_record_candidate();
        PyMem_Free(contract);
        numpy_crossjit_record_elapsed(&crossjit_stats.prepare_time_ns,
                                      started);
        return PYJIT_RESULT_HIT;
    }
    crossjit_stats.regions_provider_accepted++;

    uint8_t input_count = PyJit_RegionGetInputCount(region);
    for (uint8_t i = 0; i < input_count; i++) {
        const PyJitRegionValue *input = PyJit_RegionGetInput(region, i);
        contract->dependencies[i][0] = input->value;
        contract->scalar_keys[i] =
            (contract->execution.guards.scalar_mask & (1u << i)) ?
            ((contract->execution.guards.int_scalar_mask & (1u << i)) ?
                2 : 1) : 0;
        if (contract->scalar_keys[i] != 0) {
            contract->assumptions[i] = (PyJitAssumption){
                .kind = NUMPY_CROSSJIT_ASSUME_SCALAR,
                .dependencies = contract->dependencies[i],
                .dependency_count = 1,
                .canonical_key = &contract->scalar_keys[i],
                .canonical_key_size = sizeof(contract->scalar_keys[i]),
            };
        }
        else {
            contract->assumptions[i] = (PyJitAssumption){
                .kind = NUMPY_CROSSJIT_ASSUME_ARRAY,
                .dependencies = contract->dependencies[i],
                .dependency_count = 1,
                /* Object dtype can change Host callback, replayability, and
                 * virtual-materialization facts projected into S_H.  The
                 * exact ndarray/dtype guard justifies those facts at the
                 * Host-selected region entry as well as specializing C_P. */
                .semantic_property_mask =
                    PYJIT_SEMANTIC_PROPERTY_MATERIALIZATION |
                    PYJIT_SEMANTIC_PROPERTY_REPLAYABILITY |
                    PYJIT_SEMANTIC_PROPERTY_MAY_CALL_HOST |
                    PYJIT_SEMANTIC_PROPERTY_EXACT_RESULT_TYPE,
                .canonical_key = &contract->execution.guards.inputs[i],
                .canonical_key_size =
                    sizeof(contract->execution.guards.inputs[i]),
            };
        }
    }
    result->contract = contract;
    result->contract_size = sizeof(*contract);
    result->assumptions = contract->assumptions;
    result->assumption_count = input_count;
    if (crossjit_stats.diagnostics_enabled) {
        crossjit_stats.last.unique_guard_count = input_count;
    }
    numpy_crossjit_record_candidate();
    numpy_crossjit_record_elapsed(&crossjit_stats.prepare_time_ns, started);
    return PYJIT_RESULT_HIT;
}

/* Lowers the Host-scheduled guard batch by copying the relevant C_NumPy
 * portion; it cannot choose guard position or side-exit state. */
static PyJitResult
numpy_crossjit_lower_guards(
    const PyJitSpecializationView *specialization,
    PyJitGuardPlan **result)
{
    uint64_t started = numpy_crossjit_clock_ns();
    *result = NULL;
    if (specialization == NULL ||
        specialization->contract_size != sizeof(NumpyCrossJitContract) ||
        specialization->assumption_count == 0) {
        numpy_crossjit_record_elapsed(&crossjit_stats.guard_lower_time_ns,
                                      started);
        return PYJIT_RESULT_MISS;
    }
    const NumpyCrossJitContract *contract =
        (const NumpyCrossJitContract *)specialization->contract;
    struct PyJitGuardPlan *plan = PyMem_Malloc(sizeof(*plan));
    if (plan == NULL) {
        PyErr_NoMemory();
        numpy_crossjit_record_elapsed(&crossjit_stats.guard_lower_time_ns,
                                      started);
        return PYJIT_RESULT_ERROR;
    }
    *plan = contract->execution.guards;
    crossjit_stats.guard_plans_lowered++;
    if (crossjit_stats.diagnostics_enabled) {
        crossjit_stats.last.unique_guard_count =
            (uint8_t)specialization->assumption_count;
    }
    *result = plan;
    numpy_crossjit_record_elapsed(&crossjit_stats.guard_lower_time_ns,
                                  started);
    return PYJIT_RESULT_HIT;
}

/* Pi_P is an immutable execution copy derived from the opaque C_NumPy. */
static PyJitResult
numpy_crossjit_lower_region(
    const PyJitSpecializationView *specialization,
    PyJitExecPlan **result)
{
    uint64_t started = numpy_crossjit_clock_ns();
    *result = NULL;
    if (specialization == NULL ||
        specialization->contract_size != sizeof(NumpyCrossJitContract)) {
        numpy_crossjit_record_elapsed(
            &crossjit_stats.execution_lower_time_ns, started);
        return PYJIT_RESULT_MISS;
    }
    const NumpyCrossJitContract *contract =
        (const NumpyCrossJitContract *)specialization->contract;
    struct PyJitExecPlan *plan = PyMem_Malloc(sizeof(*plan));
    if (plan == NULL) {
        PyErr_NoMemory();
        numpy_crossjit_record_elapsed(
            &crossjit_stats.execution_lower_time_ns, started);
        return PYJIT_RESULT_ERROR;
    }
    *plan = contract->execution;
    crossjit_stats.execution_plans_lowered++;
    *result = plan;
    numpy_crossjit_record_elapsed(&crossjit_stats.execution_lower_time_ns,
                                  started);
    return PYJIT_RESULT_HIT;
}

static void
numpy_crossjit_destroy_contract(void *contract)
{
    PyMem_Free(contract);
}

static PyJitResult
numpy_crossjit_record_guard(PyJitResult status, uint64_t started)
{
    if (status == PYJIT_RESULT_HIT) {
        crossjit_stats.guard_hit++;
    }
    else if (status == PYJIT_RESULT_MISS) {
        crossjit_stats.guard_miss++;
    }
    else {
        crossjit_stats.guard_error++;
    }
    if (started != 0) {
        uint64_t ended = numpy_crossjit_clock_ns();
        if (ended >= started) {
            crossjit_stats.guard_time_ns += ended - started;
        }
    }
    return status;
}

/* Checks current runtime arrays against an immutable lowering-time contract. */
static PyJitResult
numpy_crossjit_run_guards(PyJitGuardPlan *opaque, PyObject *const *inputs,
                          Py_ssize_t input_count)
{
    uint64_t started = numpy_crossjit_clock_ns();
    struct PyJitGuardPlan *plan = (struct PyJitGuardPlan *)opaque;
    if (plan == NULL || input_count != plan->input_count) {
        return numpy_crossjit_record_guard(PYJIT_RESULT_MISS, started);
    }
    for (uint8_t i = 0; i < plan->input_count; i++) {
        if (plan->scalar_mask & (1u << i)) {
            if (crossjit_stats.diagnostics_enabled) {
                crossjit_stats.guard_checks++;
            }
            if (plan->int_scalar_mask & (1u << i)) {
                if (!numpy_crossjit_safe_int_scalar(
                        inputs[i], plan->domain.typenum)) {
                    return numpy_crossjit_record_guard(
                        PYJIT_RESULT_MISS, started);
                }
            }
            else if (!PyFloat_CheckExact(inputs[i])) {
                return numpy_crossjit_record_guard(
                    PYJIT_RESULT_MISS, started);
            }
            continue;
        }
        if (crossjit_stats.diagnostics_enabled) {
            crossjit_stats.guard_checks++;
        }
        if (!PyArray_CheckExact(inputs[i])) {
            return numpy_crossjit_record_guard(
                PYJIT_RESULT_MISS, started);
        }
        NumpyCrossJitArrayFact current;
        if (!numpy_crossjit_array_fact((PyArrayObject *)inputs[i], &current)) {
            return numpy_crossjit_record_guard(
                PYJIT_RESULT_MISS, started);
        }
        if (crossjit_stats.diagnostics_enabled) {
            crossjit_stats.guard_metadata_reads +=
                5 + (uint64_t)current.ndim * 2;
        }
        const NumpyCrossJitArrayFact *expected = &plan->inputs[i];
        if (crossjit_stats.diagnostics_enabled) {
            crossjit_stats.guard_checks += 5 + current.ndim;
        }
        if (current.typenum != expected->typenum ||
            current.ndim != expected->ndim ||
            current.c_contiguous != expected->c_contiguous ||
            current.native_byteorder != expected->native_byteorder ||
            current.aligned != expected->aligned ||
            memcmp(current.dimensions, expected->dimensions,
                   (size_t)current.ndim * sizeof(npy_intp)) != 0) {
            return numpy_crossjit_record_guard(
                PYJIT_RESULT_MISS, started);
        }
        /* Contiguous strides are derived from dtype and dimensions.  The
         * remaining classes retain exact stride guards for their kernels. */
        if (plan->iteration_class !=
                NUMPY_CROSSJIT_CONTIGUOUS_SAME_SHAPE &&
            plan->iteration_class != NUMPY_CROSSJIT_CONTIGUOUS_BROADCAST &&
            memcmp(current.strides, expected->strides,
                   (size_t)current.ndim * sizeof(npy_intp)) != 0) {
            if (crossjit_stats.diagnostics_enabled) {
                crossjit_stats.guard_checks += current.ndim;
            }
            return numpy_crossjit_record_guard(
                PYJIT_RESULT_MISS, started);
        }
    }
    return numpy_crossjit_record_guard(PYJIT_RESULT_HIT, started);
}

static PyJitResult
numpy_crossjit_record_exec(PyJitResult status, uint64_t started,
                           uint64_t allocation_ns, uint64_t kernel_ns)
{
    if (status == PYJIT_RESULT_HIT) {
        crossjit_stats.exec_hit++;
    }
    else if (status == PYJIT_RESULT_MISS) {
        crossjit_stats.exec_miss++;
    }
    else {
        crossjit_stats.exec_error++;
    }
    crossjit_stats.allocation_time_ns += allocation_ns;
    crossjit_stats.kernel_time_ns += kernel_ns;
    if (started != 0) {
        uint64_t ended = numpy_crossjit_clock_ns();
        if (ended >= started) {
            crossjit_stats.region_time_ns += ended - started;
        }
    }
    return status;
}

/* Executes without publishing output until allocation, kernel, and FPE checks pass. */
static PyJitResult
numpy_crossjit_run_region(PyJitExecPlan *opaque, PyObject *const *inputs,
                          Py_ssize_t input_count, PyObject **result)
{
    uint64_t started = numpy_crossjit_clock_ns();
    uint64_t allocation_ns = 0;
    uint64_t kernel_ns = 0;
    crossjit_stats.regions_executed++;
    struct PyJitExecPlan *plan = (struct PyJitExecPlan *)opaque;
    *result = NULL;
    const char *test_error = Py_GETENV("NPY_CROSSJIT_TEST_EXEC_ERROR");
    if (test_error != NULL && strcmp(test_error, "1") == 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "injected NumPy CrossJIT execution error");
        return numpy_crossjit_record_exec(
            PYJIT_RESULT_ERROR, started, allocation_ns, kernel_ns);
    }
    if (plan == NULL ||
        numpy_crossjit_run_guards(&plan->guards, inputs, input_count) !=
        PYJIT_RESULT_HIT) {
        return numpy_crossjit_record_exec(
            PYJIT_RESULT_MISS, started, allocation_ns, kernel_ns);
    }
    const NumpyCrossJitArrayFact *fact = &plan->guards.domain;
    uint64_t allocation_started = numpy_crossjit_clock_ns();
    PyArrayObject *output = (PyArrayObject *)PyArray_SimpleNew(
        fact->ndim, fact->dimensions, fact->typenum);
    if (allocation_started != 0) {
        uint64_t allocation_ended = numpy_crossjit_clock_ns();
        if (allocation_ended >= allocation_started) {
            allocation_ns = allocation_ended - allocation_started;
        }
    }
    if (output == NULL) {
        PyErr_Clear();
        return numpy_crossjit_record_exec(
            PYJIT_RESULT_MISS, started, allocation_ns, kernel_ns);
    }
    crossjit_stats.output_allocations++;
    const void *data[NUMPY_CROSSJIT_MAX_INPUTS];
    npy_intp input_strides[NUMPY_CROSSJIT_MAX_INPUTS][NPY_MAXDIMS] = {{0}};
    double scalar_double[NUMPY_CROSSJIT_MAX_INPUTS];
    float scalar_float[NUMPY_CROSSJIT_MAX_INPUTS];
    int linear_domain = 1;
    for (uint8_t i = 0; i < plan->guards.input_count; i++) {
        if (plan->guards.scalar_mask & (1u << i)) {
            double value = plan->guards.int_scalar_mask & (1u << i) ?
                PyLong_AsDouble(inputs[i]) : PyFloat_AS_DOUBLE(inputs[i]);
            if (value == -1.0 && PyErr_Occurred()) {
                PyErr_Clear();
                Py_DECREF(output);
                return numpy_crossjit_record_exec(
                    PYJIT_RESULT_MISS, started, allocation_ns, kernel_ns);
            }
            if (fact->typenum == NPY_FLOAT) {
                scalar_float[i] = (float)value;
                data[i] = &scalar_float[i];
            }
            else {
                scalar_double[i] = value;
                data[i] = &scalar_double[i];
            }
        }
        else {
            PyArrayObject *array = (PyArrayObject *)inputs[i];
            data[i] = PyArray_DATA(array);
            int same_domain = PyArray_NDIM(array) == fact->ndim &&
                memcmp(PyArray_DIMS(array), fact->dimensions,
                       (size_t)fact->ndim * sizeof(npy_intp)) == 0;
            if (!same_domain ||
                (!PyArray_IS_C_CONTIGUOUS(array) && fact->ndim != 1)) {
                linear_domain = 0;
            }
            int leading = (int)fact->ndim - PyArray_NDIM(array);
            for (uint8_t axis = 0; axis < fact->ndim; axis++) {
                int input_axis = (int)axis - leading;
                if (input_axis < 0 || PyArray_DIM(array, input_axis) == 1) {
                    input_strides[i][axis] = 0;
                }
                else {
                    input_strides[i][axis] = PyArray_STRIDE(array, input_axis);
                }
            }
        }
    }
    npy_intp linear_dimension = PyArray_SIZE(output);
    if (linear_domain) {
        /* Collapse C-contiguous arrays and one-dimensional views to the
         * low-overhead pointer-increment loop used by the common path. */
        for (uint8_t i = 0; i < plan->guards.input_count; i++) {
            if (plan->guards.scalar_mask & (1u << i)) {
                input_strides[i][0] = 0;
            }
            else {
                PyArrayObject *array = (PyArrayObject *)inputs[i];
                input_strides[i][0] = PyArray_IS_C_CONTIGUOUS(array) ?
                    PyArray_ITEMSIZE(array) : PyArray_STRIDE(array, 0);
            }
        }
    }
    char fp_barrier;
    int floating = fact->typenum == NPY_FLOAT || fact->typenum == NPY_DOUBLE;
    if (floating) {
        npy_clear_floatstatus_barrier(&fp_barrier);
    }
    uint64_t kernel_started = numpy_crossjit_clock_ns();
    NumpyCrossJitFixedPath executed_path = NUMPY_CROSSJIT_FIXED_GENERIC_PATH;
    int ok = numpy_crossjit_kernel_strided(
        plan->nodes, plan->node_count, plan->guards.input_count,
        plan->guards.scalar_mask, plan->output_slot, fact->typenum,
        data, input_strides, linear_domain ? 1 : fact->ndim,
        linear_domain ? &linear_dimension : fact->dimensions,
        PyArray_DATA(output),
        PyArray_SIZE(output),
        plan->guards.iteration_class ==
            NUMPY_CROSSJIT_CONTIGUOUS_SAME_SHAPE,
        &executed_path);
    if (ok && executed_path == NUMPY_CROSSJIT_FIXED_FAST_PATH) {
        crossjit_stats.fixed_fast_path_hit++;
    }
    else if (ok) {
        crossjit_stats.fixed_generic_path_hit++;
    }
    if (kernel_started != 0) {
        uint64_t kernel_ended = numpy_crossjit_clock_ns();
        if (kernel_ended >= kernel_started) {
            kernel_ns = kernel_ended - kernel_started;
        }
    }
    int fpe = floating ? npy_get_floatstatus_barrier(&fp_barrier) : 0;
    if (!ok || fpe != 0) {
        Py_DECREF(output);
        return numpy_crossjit_record_exec(
            PYJIT_RESULT_MISS, started, allocation_ns, kernel_ns);
    }
    *result = (PyObject *)output;
    return numpy_crossjit_record_exec(
        PYJIT_RESULT_HIT, started, allocation_ns, kernel_ns);
}

/* Releases a guard plan exactly once when the owning executor is destroyed. */
static void
numpy_crossjit_destroy_guard(PyJitGuardPlan *plan)
{
    crossjit_stats.guard_plans_destroyed++;
    PyMem_Free(plan);
}

/* Releases an execution plan exactly once when the owning executor is destroyed. */
static void
numpy_crossjit_destroy_exec(PyJitExecPlan *plan)
{
    crossjit_stats.execution_plans_destroyed++;
    PyMem_Free(plan);
}

static int
numpy_crossjit_dict_u64(PyObject *dict, const char *name, uint64_t value)
{
    PyObject *number = PyLong_FromUnsignedLongLong(value);
    if (number == NULL) {
        return -1;
    }
    int status = PyDict_SetItemString(dict, name, number);
    Py_DECREF(number);
    return status;
}

static PyObject *
numpy_crossjit_u64_tuple(const uint64_t *values, Py_ssize_t count)
{
    PyObject *tuple = PyTuple_New(count);
    if (tuple == NULL) {
        return NULL;
    }
    for (Py_ssize_t i = 0; i < count; i++) {
        PyObject *number = PyLong_FromUnsignedLongLong(values[i]);
        if (number == NULL) {
            Py_DECREF(tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(tuple, i, number);
    }
    return tuple;
}

static PyObject *
numpy_crossjit_candidate_dict(
    const NumpyCrossJitCandidateDiagnostic *candidate)
{
    PyObject *result = PyDict_New();
    if (result == NULL) {
        return NULL;
    }
#define ADD_CANDIDATE(name) do {                                          \
        if (numpy_crossjit_dict_u64(result, #name,                         \
                                    candidate->name) < 0) {                \
            Py_DECREF(result);                                             \
            return NULL;                                                   \
        }                                                                  \
    } while (0)
    ADD_CANDIDATE(element_count);
    ADD_CANDIDATE(profitability_score);
    ADD_CANDIDATE(profitability_threshold);
    ADD_CANDIDATE(region_length);
    ADD_CANDIDATE(layout_class);
    ADD_CANDIDATE(dtype_class);
    ADD_CANDIDATE(operand_count);
    ADD_CANDIDATE(result_count);
    ADD_CANDIDATE(internal_value_count);
    ADD_CANDIDATE(eliminable_materialization_count);
    ADD_CANDIDATE(guard_count);
    ADD_CANDIDATE(unique_guard_count);
    ADD_CANDIDATE(prepare_decision);
    ADD_CANDIDATE(rejection_reason);
#undef ADD_CANDIDATE
    return result;
}

static PyObject *
numpy_crossjit_get_stats(PyObject *NPY_UNUSED(module),
                         PyObject *NPY_UNUSED(ignored))
{
    PyObject *result = PyDict_New();
    PyObject *last = NULL;
    PyObject *candidates = NULL;
    PyObject *region_lengths = NULL;
    PyObject *layouts = NULL;
    PyObject *dtypes = NULL;
    PyObject *rejections = NULL;
    if (result == NULL) {
        return NULL;
    }
#define ADD_COUNTER(name) do {                                             \
        if (numpy_crossjit_dict_u64(result, #name,                         \
                                    crossjit_stats.name) < 0) {            \
            goto error;                                                    \
        }                                                                  \
    } while (0)
    ADD_COUNTER(nodes_recognized);
    ADD_COUNTER(regions_prepare_called);
    ADD_COUNTER(regions_provider_accepted);
    ADD_COUNTER(regions_rejected_unsupported_node);
    ADD_COUNTER(regions_rejected_unsupported_contract);
    ADD_COUNTER(regions_rejected_unprofitable);
    ADD_COUNTER(regions_rejected_other);
    ADD_COUNTER(guard_plans_lowered);
    ADD_COUNTER(execution_plans_lowered);
    ADD_COUNTER(regions_executed);
    ADD_COUNTER(guard_hit);
    ADD_COUNTER(guard_miss);
    ADD_COUNTER(guard_error);
    ADD_COUNTER(exec_hit);
    ADD_COUNTER(exec_miss);
    ADD_COUNTER(exec_error);
    ADD_COUNTER(fixed_fast_path_hit);
    ADD_COUNTER(fixed_generic_path_hit);
    ADD_COUNTER(output_allocations);
    ADD_COUNTER(internal_allocations);
    ADD_COUNTER(guard_plans_destroyed);
    ADD_COUNTER(execution_plans_destroyed);
    ADD_COUNTER(guard_metadata_reads);
    ADD_COUNTER(guard_checks);
    ADD_COUNTER(prepare_time_ns);
    ADD_COUNTER(guard_lower_time_ns);
    ADD_COUNTER(execution_lower_time_ns);
    ADD_COUNTER(guard_time_ns);
    ADD_COUNTER(region_time_ns);
    ADD_COUNTER(allocation_time_ns);
    ADD_COUNTER(kernel_time_ns);
    ADD_COUNTER(candidate_records_dropped);
#undef ADD_COUNTER

    last = crossjit_stats.diagnostics_enabled ?
        numpy_crossjit_candidate_dict(&crossjit_stats.last) :
        Py_NewRef(Py_None);
    candidates = PyTuple_New((Py_ssize_t)crossjit_stats.candidate_count);
    if (last == NULL || candidates == NULL) {
        goto error;
    }
    for (Py_ssize_t i = 0;
         i < (Py_ssize_t)crossjit_stats.candidate_count; i++) {
        PyObject *candidate = numpy_crossjit_candidate_dict(
            &crossjit_stats.candidates[i]);
        if (candidate == NULL) {
            goto error;
        }
        PyTuple_SET_ITEM(candidates, i, candidate);
    }
    if (PyDict_SetItemString(result, "last_candidate", last) < 0 ||
        PyDict_SetItemString(result, "candidates", candidates) < 0) {
        goto error;
    }
    Py_CLEAR(last);
    Py_CLEAR(candidates);

    region_lengths = numpy_crossjit_u64_tuple(
        crossjit_stats.region_length_hist,
        Py_ARRAY_LENGTH(crossjit_stats.region_length_hist));
    layouts = numpy_crossjit_u64_tuple(
        crossjit_stats.layout_hist,
        Py_ARRAY_LENGTH(crossjit_stats.layout_hist));
    dtypes = numpy_crossjit_u64_tuple(
        crossjit_stats.dtype_hist,
        Py_ARRAY_LENGTH(crossjit_stats.dtype_hist));
    rejections = numpy_crossjit_u64_tuple(
        crossjit_stats.rejection_hist,
        Py_ARRAY_LENGTH(crossjit_stats.rejection_hist));
    if (region_lengths == NULL || layouts == NULL || dtypes == NULL ||
        rejections == NULL ||
        PyDict_SetItemString(result, "region_length_histogram",
                             region_lengths) < 0 ||
        PyDict_SetItemString(result, "layout_histogram", layouts) < 0 ||
        PyDict_SetItemString(result, "dtype_histogram", dtypes) < 0 ||
        PyDict_SetItemString(result, "rejection_histogram", rejections) < 0) {
        goto error;
    }
    Py_CLEAR(region_lengths);
    Py_CLEAR(layouts);
    Py_CLEAR(dtypes);
    Py_CLEAR(rejections);
    if (PyDict_SetItemString(
            result, "diagnostics_enabled",
            crossjit_stats.diagnostics_enabled ? Py_True : Py_False) < 0 ||
        PyDict_SetItemString(
            result, "timing_enabled",
            crossjit_stats.timing_enabled ? Py_True : Py_False) < 0) {
        goto error;
    }
    return result;

error:
    Py_XDECREF(last);
    Py_XDECREF(candidates);
    Py_XDECREF(region_lengths);
    Py_XDECREF(layouts);
    Py_XDECREF(dtypes);
    Py_XDECREF(rejections);
    Py_DECREF(result);
    return NULL;
}

static void
numpy_crossjit_reset_stats_data(void)
{
    memset(&crossjit_stats, 0, sizeof(crossjit_stats));
    const char *diagnostics = Py_GETENV("NPY_CROSSJIT_DIAGNOSTICS");
    const char *timing = Py_GETENV("NPY_CROSSJIT_TELEMETRY_TIMING");
    crossjit_stats.diagnostics_enabled =
        diagnostics != NULL && strcmp(diagnostics, "1") == 0;
    crossjit_stats.timing_enabled =
        crossjit_stats.diagnostics_enabled &&
        timing != NULL && strcmp(timing, "1") == 0;
}

static PyObject *
numpy_crossjit_reset_stats(PyObject *NPY_UNUSED(module),
                           PyObject *NPY_UNUSED(ignored))
{
    numpy_crossjit_reset_stats_data();
    Py_RETURN_NONE;
}

static PyMethodDef numpy_crossjit_methods[] = {
    {"_crossjit_stats", numpy_crossjit_get_stats, METH_NOARGS, NULL},
    {"_crossjit_reset_stats", numpy_crossjit_reset_stats, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL},
};

/* Adds an exact ufunc callable to the static provider registration table. */
static int
add_ufunc_key(PyObject *module, const char *name, PyObject **slot, size_t *count)
{
    PyObject *ufunc = PyObject_GetAttrString(module, name);
    if (ufunc == NULL) {
        return -1;
    }
    *slot = ufunc;  /* NumPy has no multi-interpreter support in this build. */
    crossjit_keys[(*count)++] = (PyJitProviderKey){
        PYJIT_KEY_CALLABLE, 0, NULL, NULL, ufunc};
    return 0;
}

/* Registers NumPy's static provider after all referenced ufuncs exist. */
int
numpy_crossjit_init(PyObject *module)
{
#ifdef Py_GIL_DISABLED
    /* This provider's facts use the standard GIL build's NumPy globals. */
    return 0;
#else
    numpy_crossjit_reset_stats_data();
    if (PyModule_AddFunctions(module, numpy_crossjit_methods) < 0) {
        return -1;
    }
    size_t count = 0;
    const int binary_ops[] = {
        NB_ADD, NB_INPLACE_ADD, NB_SUBTRACT, NB_INPLACE_SUBTRACT,
        NB_MULTIPLY, NB_INPLACE_MULTIPLY, NB_TRUE_DIVIDE,
        NB_INPLACE_TRUE_DIVIDE, NB_POWER, NB_INPLACE_POWER,
        NB_MATRIX_MULTIPLY, NB_INPLACE_MATRIX_MULTIPLY,
    };
    for (size_t i = 0; i < sizeof(binary_ops) / sizeof(binary_ops[0]); i++) {
        crossjit_keys[count++] = (PyJitProviderKey){
            PYJIT_KEY_BINARY, binary_ops[i], &PyArray_Type, &PyArray_Type, NULL};
    }
    const int scalar_ops[] = {
        NB_ADD, NB_SUBTRACT, NB_MULTIPLY, NB_TRUE_DIVIDE, NB_POWER,
    };
    for (size_t i = 0; i < sizeof(scalar_ops) / sizeof(scalar_ops[0]); i++) {
        crossjit_keys[count++] = (PyJitProviderKey){
            PYJIT_KEY_BINARY, scalar_ops[i], &PyArray_Type, &PyFloat_Type, NULL};
        crossjit_keys[count++] = (PyJitProviderKey){
            PYJIT_KEY_BINARY, scalar_ops[i], &PyFloat_Type, &PyArray_Type, NULL};
        crossjit_keys[count++] = (PyJitProviderKey){
            PYJIT_KEY_BINARY, scalar_ops[i], &PyArray_Type, &PyLong_Type, NULL};
        crossjit_keys[count++] = (PyJitProviderKey){
            PYJIT_KEY_BINARY, scalar_ops[i], &PyLong_Type, &PyArray_Type, NULL};
    }
    crossjit_keys[count++] = (PyJitProviderKey){
        PYJIT_KEY_UNARY, PYJIT_UNARY_NEGATIVE, &PyArray_Type, NULL, NULL};
    /* Operator spelling and ufunc spelling share the same semantic ids. */
    crossjit_keys[count++] = (PyJitProviderKey){
        PYJIT_KEY_COMPARE, Py_LT, &PyArray_Type, &PyArray_Type, NULL};
    crossjit_keys[count++] = (PyJitProviderKey){
        PYJIT_KEY_COMPARE, Py_LE, &PyArray_Type, &PyArray_Type, NULL};
    if (add_ufunc_key(module, "add", &crossjit_state.add, &count) < 0 ||
        add_ufunc_key(module, "subtract", &crossjit_state.subtract, &count) < 0 ||
        add_ufunc_key(module, "multiply", &crossjit_state.multiply, &count) < 0 ||
        add_ufunc_key(module, "true_divide", &crossjit_state.true_divide, &count) < 0 ||
        add_ufunc_key(module, "power", &crossjit_state.power, &count) < 0 ||
        add_ufunc_key(module, "negative", &crossjit_state.negative, &count) < 0 ||
        add_ufunc_key(module, "minimum", &crossjit_state.minimum, &count) < 0 ||
        add_ufunc_key(module, "maximum", &crossjit_state.maximum, &count) < 0 ||
        add_ufunc_key(module, "less", &crossjit_state.less, &count) < 0 ||
        add_ufunc_key(module, "less_equal", &crossjit_state.less_equal, &count) < 0 ||
        add_ufunc_key(module, "logical_and", &crossjit_state.logical_and, &count) < 0 ||
        add_ufunc_key(module, "matmul", &crossjit_state.matmul, &count) < 0) {
        return -1;
    }
    /* mean is a Python-level convenience in some NumPy builds; semantic
     * support remains available whenever this exact callable is exported. */
    crossjit_state.mean = PyObject_GetAttrString(module, "mean");
    if (crossjit_state.mean != NULL) {
        crossjit_keys[count++] = (PyJitProviderKey){PYJIT_KEY_CALLABLE, 0, NULL, NULL, crossjit_state.mean};
    }
    else {
        PyErr_Clear();
    }
    crossjit_state.add_outer = PyObject_GetAttrString(crossjit_state.add, "outer");
    crossjit_state.maximum_reduce = PyObject_GetAttrString(crossjit_state.maximum, "reduce");
    if (crossjit_state.add_outer == NULL || crossjit_state.maximum_reduce == NULL) {
        return -1;
    }
    crossjit_keys[count++] = (PyJitProviderKey){PYJIT_KEY_CALLABLE, 0, NULL, NULL, crossjit_state.add_outer};
    crossjit_keys[count++] = (PyJitProviderKey){PYJIT_KEY_CALLABLE, 0, NULL, NULL, crossjit_state.maximum_reduce};
    crossjit_provider = (PyJitProviderSpec){
        .abi_version = PYJIT_PROVIDER_ABI_VERSION,
        .struct_size = sizeof(PyJitProviderSpec),
        .name = "numpy-semantic",
        .keys = crossjit_keys,
        .key_count = count,
        .recognize = numpy_crossjit_recognize,
        .project_semantics = numpy_crossjit_project_semantics,
        .prepare_region = numpy_crossjit_prepare_region,
        .lower_guards = numpy_crossjit_lower_guards,
        .lower_region = numpy_crossjit_lower_region,
        .run_guards = numpy_crossjit_run_guards,
        .run_region = numpy_crossjit_run_region,
        .destroy_contract = numpy_crossjit_destroy_contract,
        .destroy_guard_plan = numpy_crossjit_destroy_guard,
        .destroy_exec_plan = numpy_crossjit_destroy_exec,
    };
    return PyJit_RegisterProvider(&crossjit_provider);
#endif
}

#else

/* Keeps non-CrossJIT CPython builds source-compatible without registering. */
int
numpy_crossjit_init(PyObject *NPY_UNUSED(module))
{
    return 0;
}

#endif  /* PYJIT_PROVIDER_ABI_VERSION */
