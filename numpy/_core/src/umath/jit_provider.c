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

static NumpyCrossJitState crossjit_state;
static PyJitProviderKey crossjit_keys[64];
static PyJitProviderSpec crossjit_provider;

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
numpy_crossjit_lowering_reject(const char *reason)
{
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
        return numpy_crossjit_lowering_reject("invalid live-in count");
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
                "live-in is neither an exact ndarray nor an exact float");
        }
        NumpyCrossJitArrayFact *fact = &guards->inputs[i];
        if (!fact->native_byteorder || !fact->aligned ||
            !numpy_crossjit_supported_dtype(fact->typenum)) {
            return numpy_crossjit_lowering_reject(
                "array live-in has unsupported layout or dtype");
        }
        if (!have_domain) {
            guards->domain = *fact;
            have_domain = 1;
        }
        else if (fact->typenum != guards->domain.typenum) {
            return numpy_crossjit_lowering_reject(
                "array live-ins have different dtypes");
        }
        if (fact->ndim > domain_ndim) {
            domain_ndim = fact->ndim;
        }
    }
    /* Python floats participate in NumPy's weak scalar promotion for the
     * supported floating domains; integer-array promotion is left generic. */
    if (!have_domain) {
        return numpy_crossjit_lowering_reject("region has no array domain");
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
    if (!numpy_crossjit_build_guards(region, &plan->guards)) {
        return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
    }
    plan->node_count = PyJit_RegionGetNodeCount(region);
    if (plan->node_count < 2 || plan->node_count > 6) {
        numpy_crossjit_lowering_reject("unsupported node count");
        return PYJIT_PREPARE_UNSUPPORTED_NODE;
    }
    uint64_t elements = numpy_crossjit_domain_elements(&plan->guards.domain);
    uint64_t eliminated = (uint64_t)(plan->node_count - 1);
    uint64_t work = elements > UINT64_MAX / eliminated ? UINT64_MAX :
        elements * eliminated;
    uint64_t min_work = numpy_crossjit_min_work();
    if (work < min_work) {
        if (Py_GETENV("PYTHON_JIT_CROSSJIT_DEBUG") != NULL) {
            fprintf(stderr,
                    "NumPy CrossJIT rejected short region: nodes=%u, "
                    "elements=%llu, work=%llu, minimum=%llu\n",
                    plan->node_count, (unsigned long long)elements,
                    (unsigned long long)work, (unsigned long long)min_work);
        }
        numpy_crossjit_lowering_reject(
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
    if (result == NULL || result->struct_size != sizeof(*result)) {
        PyErr_SetString(PyExc_ValueError,
                        "invalid NumPy CrossJIT prepare result");
        return PYJIT_RESULT_ERROR;
    }
    NumpyCrossJitContract *contract =
        PyMem_Calloc(1, sizeof(*contract));
    if (contract == NULL) {
        PyErr_NoMemory();
        return PYJIT_RESULT_ERROR;
    }
    PyJitPrepareDecision decision = numpy_crossjit_build_exec(
        region, &contract->execution);
    result->decision = decision;
    if (decision != PYJIT_PREPARE_ACCEPT) {
        PyMem_Free(contract);
        return PYJIT_RESULT_HIT;
    }

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
    return PYJIT_RESULT_HIT;
}

/* Lowers the Host-scheduled guard batch by copying the relevant C_NumPy
 * portion; it cannot choose guard position or side-exit state. */
static PyJitResult
numpy_crossjit_lower_guards(
    const PyJitSpecializationView *specialization,
    PyJitGuardPlan **result)
{
    *result = NULL;
    if (specialization == NULL ||
        specialization->contract_size != sizeof(NumpyCrossJitContract) ||
        specialization->assumption_count == 0) {
        return PYJIT_RESULT_MISS;
    }
    const NumpyCrossJitContract *contract =
        (const NumpyCrossJitContract *)specialization->contract;
    struct PyJitGuardPlan *plan = PyMem_Malloc(sizeof(*plan));
    if (plan == NULL) {
        PyErr_NoMemory();
        return PYJIT_RESULT_ERROR;
    }
    *plan = contract->execution.guards;
    *result = plan;
    return PYJIT_RESULT_HIT;
}

/* Pi_P is an immutable execution copy derived from the opaque C_NumPy. */
static PyJitResult
numpy_crossjit_lower_region(
    const PyJitSpecializationView *specialization,
    PyJitExecPlan **result)
{
    *result = NULL;
    if (specialization == NULL ||
        specialization->contract_size != sizeof(NumpyCrossJitContract)) {
        return PYJIT_RESULT_MISS;
    }
    const NumpyCrossJitContract *contract =
        (const NumpyCrossJitContract *)specialization->contract;
    struct PyJitExecPlan *plan = PyMem_Malloc(sizeof(*plan));
    if (plan == NULL) {
        PyErr_NoMemory();
        return PYJIT_RESULT_ERROR;
    }
    *plan = contract->execution;
    *result = plan;
    return PYJIT_RESULT_HIT;
}

static void
numpy_crossjit_destroy_contract(void *contract)
{
    PyMem_Free(contract);
}

/* Checks current runtime arrays against an immutable lowering-time contract. */
static PyJitResult
numpy_crossjit_run_guards(PyJitGuardPlan *opaque, PyObject *const *inputs,
                          Py_ssize_t input_count)
{
    struct PyJitGuardPlan *plan = (struct PyJitGuardPlan *)opaque;
    if (plan == NULL || input_count != plan->input_count) {
        return PYJIT_RESULT_MISS;
    }
    for (uint8_t i = 0; i < plan->input_count; i++) {
        if (plan->scalar_mask & (1u << i)) {
            if (plan->int_scalar_mask & (1u << i)) {
                if (!numpy_crossjit_safe_int_scalar(
                        inputs[i], plan->domain.typenum)) {
                    return PYJIT_RESULT_MISS;
                }
            }
            else if (!PyFloat_CheckExact(inputs[i])) {
                return PYJIT_RESULT_MISS;
            }
            continue;
        }
        if (!PyArray_CheckExact(inputs[i])) {
            return PYJIT_RESULT_MISS;
        }
        NumpyCrossJitArrayFact current;
        if (!numpy_crossjit_array_fact((PyArrayObject *)inputs[i], &current)) {
            return PYJIT_RESULT_MISS;
        }
        const NumpyCrossJitArrayFact *expected = &plan->inputs[i];
        if (current.typenum != expected->typenum ||
            current.ndim != expected->ndim ||
            current.c_contiguous != expected->c_contiguous ||
            current.native_byteorder != expected->native_byteorder ||
            current.aligned != expected->aligned ||
            memcmp(current.dimensions, expected->dimensions,
                   (size_t)current.ndim * sizeof(npy_intp)) != 0) {
            return PYJIT_RESULT_MISS;
        }
        /* Contiguous strides are derived from dtype and dimensions.  The
         * remaining classes retain exact stride guards for their kernels. */
        if (plan->iteration_class !=
                NUMPY_CROSSJIT_CONTIGUOUS_SAME_SHAPE &&
            plan->iteration_class != NUMPY_CROSSJIT_CONTIGUOUS_BROADCAST &&
            memcmp(current.strides, expected->strides,
                   (size_t)current.ndim * sizeof(npy_intp)) != 0) {
            return PYJIT_RESULT_MISS;
        }
    }
    return PYJIT_RESULT_HIT;
}

/* Executes without publishing output until allocation, kernel, and FPE checks pass. */
static PyJitResult
numpy_crossjit_run_region(PyJitExecPlan *opaque, PyObject *const *inputs,
                          Py_ssize_t input_count, PyObject **result)
{
    struct PyJitExecPlan *plan = (struct PyJitExecPlan *)opaque;
    *result = NULL;
    if (plan == NULL ||
        numpy_crossjit_run_guards(&plan->guards, inputs, input_count) !=
        PYJIT_RESULT_HIT) {
        return PYJIT_RESULT_MISS;
    }
    const NumpyCrossJitArrayFact *fact = &plan->guards.domain;
    PyArrayObject *output = (PyArrayObject *)PyArray_SimpleNew(
        fact->ndim, fact->dimensions, fact->typenum);
    if (output == NULL) {
        PyErr_Clear();
        return PYJIT_RESULT_MISS;
    }
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
                return PYJIT_RESULT_MISS;
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
    int ok = numpy_crossjit_kernel_strided(
        plan->nodes, plan->node_count, plan->guards.input_count,
        plan->guards.scalar_mask, plan->output_slot, fact->typenum,
        data, input_strides, linear_domain ? 1 : fact->ndim,
        linear_domain ? &linear_dimension : fact->dimensions,
        PyArray_DATA(output),
        PyArray_SIZE(output));
    int fpe = floating ? npy_get_floatstatus_barrier(&fp_barrier) : 0;
    if (!ok || fpe != 0) {
        Py_DECREF(output);
        return PYJIT_RESULT_MISS;
    }
    *result = (PyObject *)output;
    return PYJIT_RESULT_HIT;
}

/* Releases a guard plan exactly once when the owning executor is destroyed. */
static void
numpy_crossjit_destroy_guard(PyJitGuardPlan *plan)
{
    PyMem_Free(plan);
}

/* Releases an execution plan exactly once when the owning executor is destroyed. */
static void
numpy_crossjit_destroy_exec(PyJitExecPlan *plan)
{
    PyMem_Free(plan);
}

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
