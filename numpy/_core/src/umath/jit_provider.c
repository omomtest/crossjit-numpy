/* NumPy's CrossJIT semantic provider.  It deliberately owns no trace policy,
 * guard placement, deoptimization, or code generation. */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include <Python.h>
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <opcode.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"
#include "numpy/npy_math.h"
#include "numpy/ufuncobject.h"
#include "jit_kernel.h"
#include "generated_crossjit_backend.h"
#include "jit_provider.h"
#include "generated_crossjit_native_policy.h"
#include "matmul.h"
#include "ufunc_object.h"
#include "npy_static_data.h"
#include "arrayfunction_override.h"

#ifdef PYJIT_PROVIDER_ABI_VERSION

/* Stable semantic operation identities. Recognition is broader than either
 * generated elementwise lowering or the measured native execution subset. */
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
    NUMPY_CROSSJIT_BOOLEAN_SUBSCRIPT, /* reserved; do not renumber later IDs */
    NUMPY_CROSSJIT_ADD_REDUCE,
    NUMPY_CROSSJIT_SQRT,
    NUMPY_CROSSJIT_EXP,
    NUMPY_CROSSJIT_TANH,
    NUMPY_CROSSJIT_DOT,
    NUMPY_CROSSJIT_SUM,
    NUMPY_CROSSJIT_MAX,
    NUMPY_CROSSJIT_DOT_PUBLIC,
    NUMPY_CROSSJIT_AMAX,
} NumpyCrossJitOp;

/* The high semantic-id bit records Python's in-place spelling.  Such nodes
 * remain analyzable but are mutation barriers and are never lowered. */
#define NUMPY_CROSSJIT_INPLACE_FLAG (UINT64_C(1) << 63)

/*
 * One work unit represents one eliminated intermediate-array element.  The
 * default keeps short regions on NumPy's mature ufunc loops, while allowing
 * a larger DAG to amortize dispatch and allocation costs at a smaller array
 * size.  The historical global threshold is no longer production policy;
 * NPY_CROSSJIT_MIN_WORK remains a diagnostic/test override only.
 */
#define NUMPY_CROSSJIT_DEFAULT_MIN_WORK UINT64_C(0)

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
    PyObject *add_reduce;
    PyObject *sqrt;
    PyObject *exp;
    PyObject *tanh;
    PyObject *dot;
} NumpyCrossJitState;

/* Canonical ndarray facts shared by generated and native plan validation. */
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

/* Diagnostic/profile taxonomy.  These classes are Provider-owned and are
 * deliberately more precise than the historical execution-class enum. */
typedef enum {
    NUMPY_CROSSJIT_PROFILE_FULL_CONTIGUOUS = 1,
    NUMPY_CROSSJIT_PROFILE_INNER_CONTIGUOUS = 2,
    NUMPY_CROSSJIT_PROFILE_ZERO_STRIDE_BROADCAST = 3,
    NUMPY_CROSSJIT_PROFILE_SCALAR_OPERAND = 4,
    NUMPY_CROSSJIT_PROFILE_FIXED_AFFINE_STRIDE = 5,
    NUMPY_CROSSJIT_PROFILE_TRULY_GENERAL_STRIDED = 6,
} NumpyCrossJitProfileLayoutClass;

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
#define NUMPY_CROSSJIT_MAX_OBSERVED_SIGNATURES 128
#define NUMPY_CROSSJIT_MAX_UNSUPPORTED_NODES 64
#define NUMPY_CROSSJIT_ELEMENT_BUCKETS 14

typedef enum {
    NUMPY_CROSSJIT_OBSERVED_PREPARE = 0,
    NUMPY_CROSSJIT_OBSERVED_ACCEPTED = 1,
    NUMPY_CROSSJIT_OBSERVED_EXECUTION = 2,
    NUMPY_CROSSJIT_OBSERVED_GENERATED = 3,
    NUMPY_CROSSJIT_OBSERVED_GENERIC = 4,
} NumpyCrossJitObservedEvent;

typedef struct {
    uint64_t prepare_count;
    uint64_t accepted_count;
    uint64_t execution_count;
    uint64_t generated_hit_count;
    uint64_t generic_fallback_count;
    uint64_t min_elements;
    uint64_t max_elements;
    uint64_t prepare_element_buckets[NUMPY_CROSSJIT_ELEMENT_BUCKETS];
    uint64_t element_buckets[NUMPY_CROSSJIT_ELEMENT_BUCKETS];
    uint64_t outer_count;
    uint64_t inner_extent;
    uint8_t input_count;
    uint8_t node_count;
    uint8_t dtype_class;
    uint8_t layout_class;
    uint8_t refined_layout_class;
    uint8_t iteration_rank_class;
    uint8_t access_kinds[NUMPY_CROSSJIT_MAX_INPUTS];
    /* Diagnostic exemplars, never signature or selection keys.  Records can
     * aggregate several shapes/strides; min/max elements retain that envelope. */
    npy_intp sample_inner_strides[NUMPY_CROSSJIT_MAX_INPUTS];
    npy_intp sample_outer_strides[NUMPY_CROSSJIT_MAX_INPUTS];
    uint16_t sample_numpy_float64_scalar_mask;
    uint64_t sample_value_class_mask;
    uint8_t iteration_descriptor_version;
    uint8_t outer_shape_class;
    uint16_t scalar_mask;
    NumpyCrossJitKernelNode nodes[NUMPY_CROSSJIT_MAX_NODES];
} NumpyCrossJitObservedSignature;

typedef struct {
    uint64_t semantic_id;
    uint64_t occurrence_count;
    uint64_t min_elements;
    uint64_t max_elements;
    uint8_t node_index;
    uint8_t region_length;
    uint8_t dtype_class;
    uint8_t refined_layout_class;
    uint8_t inplace;
    uint16_t scalar_mask;
} NumpyCrossJitUnsupportedNode;

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
    uint64_t fixed_fast_plans_built;
    uint64_t fixed_generic_plans_built;
    uint64_t generated_registry_hit;
    uint64_t generated_registry_miss;
    uint64_t generated_exec_hit;
    uint64_t generic_fallback_hit;
    uint64_t native_exec_hit;
    uint64_t mixed_exec_hit;
    uint64_t native_plans_built;
    uint64_t output_allocations;
    uint64_t internal_allocations;
    uint64_t internal_buffer_bytes;
    uint64_t internal_object_allocations;
    uint64_t packing_allocations;
    uint64_t packing_bytes;
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
    NumpyCrossJitObservedSignature
        observed_signatures[NUMPY_CROSSJIT_MAX_OBSERVED_SIGNATURES];
    uint64_t observed_signature_count;
    uint64_t observed_signature_records_dropped;
    NumpyCrossJitUnsupportedNode
        unsupported_nodes[NUMPY_CROSSJIT_MAX_UNSUPPORTED_NODES];
    uint64_t unsupported_node_count;
    uint64_t unsupported_node_records_dropped;
    int diagnostics_enabled;
    int timing_enabled;
} NumpyCrossJitStats;

static NumpyCrossJitState crossjit_state;
static PyJitProviderKey crossjit_keys[192];
static PyObject *native_observations;
static PyJitProviderSpec crossjit_provider;
static NumpyCrossJitStats crossjit_stats;
static _Thread_local int native_packing_active;

void
numpy_crossjit_native_record_packing(size_t bytes)
{
    if (native_packing_active) {
        crossjit_stats.packing_allocations++;
        crossjit_stats.packing_bytes+=bytes;
    }
}


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
    uint8_t refined_layout_class; /* diagnostic/profile structural class */
    uint8_t iteration_rank_class; /* 1-D versus multi-dimensional domain */
    uint8_t access_kinds[NUMPY_CROSSJIT_MAX_INPUTS];
    uint16_t scalar_mask;         /* validated scalar positions */
    uint16_t int_scalar_mask;     /* scalar positions needing exact int checks */
    uint16_t double_scalar_mask;  /* exact NumPy float64, strong promotion */
    uint16_t numpy_int_scalar_mask; /* exact native signed 64-bit scalar */
    uint16_t float_scalar_mask;   /* exact NumPy float32 */
    uint64_t scalar_class_mask;   /* Provider performance class, four bits/input */
    uint8_t guard_scalar_classes; /* class-sensitive measured policy */
    uint8_t staged;              /* independent per-node iteration domains */
    uint16_t constant_mask;      /* immutable call metadata, separately guarded */
    uint8_t constant_kinds[NUMPY_CROSSJIT_MAX_INPUTS];
    uint8_t alias_mask;
    uint16_t builtin_ufunc_mask;
    uint64_t outer_count;
    uint64_t inner_extent;
    NumpyCrossJitIterationDescriptor iteration;
    NumpyCrossJitArrayFact domain; /* common output dtype, shape, and layout */
    NumpyCrossJitArrayFact inputs[NUMPY_CROSSJIT_MAX_INPUTS]; /* array contracts */
};

/*
 * The execution plan owns domain contracts and backend bindings. Generator
 * artifacts supply the elementwise drivers and mixed-stage schedule.
 */
typedef struct {
    PyUFuncObject *ufunc; /* borrowed builtin; initialized table retained by module */
    PyUFuncGenericFunction entry;
    int index;
} NumpyCrossJitNativeLoop;

typedef struct {
    uint8_t semantic_op;
    uint8_t left, right, output;
    uint8_t scalar_result;
    uint8_t axis_count, keepdims;
    int axes[NPY_MAXDIMS];
    NumpyCrossJitArrayFact domain;
    NumpyCrossJitFixedPlan fixed;
    NumpyCrossJitNativeLoop loop;
    PyUFuncGenericFunction unary_entry;
    int unary_index;
} NumpyCrossJitStage;

struct PyJitExecPlan {
    struct PyJitGuardPlan guards; /* private copy keeps execution self-contained */
    uint8_t node_count;           /* bounded primitive program */
    uint8_t output_slot;          /* final virtual slot written to the ndarray */
    NumpyCrossJitKernelNode nodes[NUMPY_CROSSJIT_MAX_NODES]; /* topological DAG */
    /* Non-NULL entry means prepare_region() completely resolved the float64
     * contiguous signature, DAG shape, scalar roles, and operand slot map. */
    NumpyCrossJitFixedPlan fixed;
    uint8_t native_backend; /* measured precompiled execution, separate from replay */
    NumpyCrossJitNativeLoop loop;
    char native_key[16384]; /* canonical, collision-checked complete-region key */
    uint8_t stage_count;
    uint8_t reuse_final_loop; /* unpublished private ndarray, last-use only */
    CjExecutionPlan execution; /* Generator schedule/lifetimes; no owned objects */
    NumpyCrossJitStage stages[NUMPY_CROSSJIT_MAX_NODES];
};

static int
numpy_crossjit_observed_signature_equal(
    const NumpyCrossJitObservedSignature *observed,
    const struct PyJitExecPlan *plan)
{
    return observed->input_count == plan->guards.input_count &&
        observed->node_count == plan->node_count &&
        observed->dtype_class ==
            numpy_crossjit_dtype_class(plan->guards.domain.typenum) &&
        observed->layout_class == plan->guards.iteration_class &&
        observed->refined_layout_class ==
            plan->guards.refined_layout_class &&
        observed->iteration_rank_class ==
            plan->guards.iteration_rank_class &&
        observed->iteration_descriptor_version ==
            plan->guards.iteration.version &&
        observed->outer_shape_class ==
            plan->guards.iteration.outer_shape_class &&
        observed->scalar_mask == plan->guards.scalar_mask &&
        memcmp(observed->access_kinds, plan->guards.access_kinds,
               (size_t)plan->guards.input_count) == 0 &&
        memcmp(observed->nodes, plan->nodes,
               (size_t)plan->node_count * sizeof(plan->nodes[0])) == 0;
}

static void
numpy_crossjit_record_observed_signature(
    const struct PyJitExecPlan *plan, NumpyCrossJitObservedEvent event)
{
    if (!crossjit_stats.diagnostics_enabled || plan == NULL) {
        return;
    }
    uint64_t elements = numpy_crossjit_domain_elements(&plan->guards.domain);
    NumpyCrossJitObservedSignature *observed = NULL;
    for (uint64_t i = 0; i < crossjit_stats.observed_signature_count; i++) {
        if (numpy_crossjit_observed_signature_equal(
                &crossjit_stats.observed_signatures[i], plan)) {
            observed = &crossjit_stats.observed_signatures[i];
            break;
        }
    }
    if (observed == NULL) {
        if (crossjit_stats.observed_signature_count >=
                NUMPY_CROSSJIT_MAX_OBSERVED_SIGNATURES) {
            crossjit_stats.observed_signature_records_dropped++;
            return;
        }
        observed = &crossjit_stats.observed_signatures[
            crossjit_stats.observed_signature_count++];
        memset(observed, 0, sizeof(*observed));
        observed->input_count = plan->guards.input_count;
        observed->node_count = plan->node_count;
        observed->dtype_class =
            numpy_crossjit_dtype_class(plan->guards.domain.typenum);
        observed->layout_class = plan->guards.iteration_class;
        observed->refined_layout_class =
            plan->guards.refined_layout_class;
        observed->iteration_rank_class =
            plan->guards.iteration_rank_class;
        observed->outer_count = plan->guards.outer_count;
        observed->inner_extent = plan->guards.inner_extent;
        observed->iteration_descriptor_version =
            plan->guards.iteration.version;
        observed->outer_shape_class =
            plan->guards.iteration.outer_shape_class;
        memcpy(observed->access_kinds, plan->guards.access_kinds,
               (size_t)plan->guards.input_count);
        for (uint8_t i = 0; i < plan->guards.input_count; i++) {
            observed->sample_inner_strides[i] =
                plan->guards.iteration.operands[i].inner_stride;
            observed->sample_outer_strides[i] =
                plan->guards.iteration.operands[i].outer_stride;
        }
        observed->sample_numpy_float64_scalar_mask =
            plan->guards.double_scalar_mask;
        observed->sample_value_class_mask = plan->guards.scalar_class_mask;
        observed->scalar_mask = plan->guards.scalar_mask;
        observed->min_elements = elements;
        observed->max_elements = elements;
        memcpy(observed->nodes, plan->nodes,
               (size_t)plan->node_count * sizeof(plan->nodes[0]));
    }
    if (event == NUMPY_CROSSJIT_OBSERVED_PREPARE ||
            event == NUMPY_CROSSJIT_OBSERVED_EXECUTION) {
        if (elements < observed->min_elements) {
            observed->min_elements = elements;
        }
        if (elements > observed->max_elements) {
            observed->max_elements = elements;
        }
    }
    switch (event) {
        case NUMPY_CROSSJIT_OBSERVED_PREPARE:
            observed->prepare_count++;
            {
                static const uint64_t upper_bounds[
                    NUMPY_CROSSJIT_ELEMENT_BUCKETS] = {
                    16, 32, 64, 128, 256, 512, 1024, 2048,
                    4096, 8192, 16384, 32768, 65536, UINT64_MAX,
                };
                uint8_t bucket = 0;
                while (elements > upper_bounds[bucket]) {
                    bucket++;
                }
                observed->prepare_element_buckets[bucket]++;
            }
            break;
        case NUMPY_CROSSJIT_OBSERVED_ACCEPTED:
            observed->accepted_count++;
            break;
        case NUMPY_CROSSJIT_OBSERVED_EXECUTION: {
            static const uint64_t upper_bounds[NUMPY_CROSSJIT_ELEMENT_BUCKETS] = {
                16, 32, 64, 128, 256, 512, 1024, 2048,
                4096, 8192, 16384, 32768, 65536, UINT64_MAX,
            };
            uint8_t bucket = 0;
            while (elements > upper_bounds[bucket]) {
                bucket++;
            }
            observed->execution_count++;
            observed->element_buckets[bucket]++;
            break;
        }
        case NUMPY_CROSSJIT_OBSERVED_GENERATED:
            observed->generated_hit_count++;
            break;
        case NUMPY_CROSSJIT_OBSERVED_GENERIC:
            observed->generic_fallback_count++;
            break;
    }
}

static void
numpy_crossjit_record_unsupported_node(const struct PyJitExecPlan *plan,
                                       uint64_t semantic_id,
                                       uint8_t node_index)
{
    if (!crossjit_stats.diagnostics_enabled || plan == NULL) {
        return;
    }
    uint8_t inplace = (semantic_id & NUMPY_CROSSJIT_INPLACE_FLAG) != 0;
    semantic_id &= ~NUMPY_CROSSJIT_INPLACE_FLAG;
    uint64_t elements = numpy_crossjit_domain_elements(&plan->guards.domain);
    NumpyCrossJitUnsupportedNode *record = NULL;
    for (uint64_t i = 0; i < crossjit_stats.unsupported_node_count; i++) {
        NumpyCrossJitUnsupportedNode *candidate =
            &crossjit_stats.unsupported_nodes[i];
        if (candidate->semantic_id == semantic_id &&
                candidate->node_index == node_index &&
                candidate->region_length == plan->node_count &&
                candidate->dtype_class == numpy_crossjit_dtype_class(
                    plan->guards.domain.typenum) &&
                candidate->refined_layout_class ==
                    plan->guards.refined_layout_class &&
                candidate->inplace == inplace &&
                candidate->scalar_mask == plan->guards.scalar_mask) {
            record = candidate;
            break;
        }
    }
    if (record == NULL) {
        if (crossjit_stats.unsupported_node_count >=
                NUMPY_CROSSJIT_MAX_UNSUPPORTED_NODES) {
            crossjit_stats.unsupported_node_records_dropped++;
            return;
        }
        record = &crossjit_stats.unsupported_nodes[
            crossjit_stats.unsupported_node_count++];
        memset(record, 0, sizeof(*record));
        record->semantic_id = semantic_id;
        record->node_index = node_index;
        record->region_length = plan->node_count;
        record->dtype_class = numpy_crossjit_dtype_class(
            plan->guards.domain.typenum);
        record->refined_layout_class =
            plan->guards.refined_layout_class;
        record->inplace = inplace;
        record->scalar_mask = plan->guards.scalar_mask;
        record->min_elements = elements;
        record->max_elements = elements;
    }
    record->occurrence_count++;
    if (elements < record->min_elements) {
        record->min_elements = elements;
    }
    if (elements > record->max_elements) {
        record->max_elements = elements;
    }
}

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
numpy_crossjit_safe_array_owner(PyArrayObject *array)
{
    if (!PyArray_CheckExact((PyObject *)array)) {
        return 0;
    }
    /* Suppressed input cleanup cannot invoke a user buffer destructor or
     * custom NumPy data allocator between stages. Exact builtin byte buffers
     * and the default ndarray ownership chain have callback-free cleanup. */
    PyObject *owner=(PyObject *)array;
    for (int depth=0;owner!=NULL;depth++) {
        if (depth>NPY_MAXDIMS) return 0;
        if (!PyArray_CheckExact(owner)) {
            if (!PyBytes_CheckExact(owner) && !PyByteArray_CheckExact(owner)) return 0;
            break;
        }
        PyObject *handler=PyArray_HANDLER((PyArrayObject *)owner);
        if (handler && handler!=PyDataMem_DefaultHandler) return 0;
        /* Holding a live-in until region exit must not postpone Python
         * weakref callbacks that original intermediate cleanup can invoke.
         * Callback-free weak references are still safe under the held GIL. */
        for (PyWeakReference *ref=(PyWeakReference *)
                ((PyArrayObject_fields *)owner)->weakreflist;ref;ref=ref->wr_next) {
            if (ref->wr_callback && ref->wr_callback!=Py_None) return 0;
        }
        owner=PyArray_BASE((PyArrayObject *)owner);
    }
    return 1;
}

static int
numpy_crossjit_array_fact(PyArrayObject *array, NumpyCrossJitArrayFact *fact)
{
    if (!numpy_crossjit_safe_array_owner(array)) return 0;
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

#include "jit_native_aliases.h"
static int numpy_crossjit_stage_c_order(const NumpyCrossJitArrayFact *);

static int
numpy_crossjit_same_callable(PyObject *left, PyObject *right)
{
    if (left==right) return 1;
    return left && right && PyCFunction_CheckExact(left) && PyCFunction_CheckExact(right) &&
        PyCFunction_GET_SELF(left)==PyCFunction_GET_SELF(right) &&
        PyCFunction_GET_FUNCTION(left)==PyCFunction_GET_FUNCTION(right) &&
        PyCFunction_GET_FLAGS(left)==PyCFunction_GET_FLAGS(right);
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
        int alias=numpy_crossjit_alias_index(callable);
        if (alias>=0) op=NUMPY_CROSSJIT_SUM+alias;
        else if (callable == crossjit_state.add) op = NUMPY_CROSSJIT_ADD;
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
        else if (callable == crossjit_state.sqrt) op = NUMPY_CROSSJIT_SQRT;
        else if (callable == crossjit_state.exp) op = NUMPY_CROSSJIT_EXP;
        else if (callable == crossjit_state.tanh) op = NUMPY_CROSSJIT_TANH;
        else if (callable == crossjit_state.dot) {
            if (request->operand_count!=2) return PYJIT_RESULT_MISS;
            for (uint8_t i=0;i<2;i++) {
                PyObject *value=PyJit_ValueGetObservedObject(request->operands[i]);
                if (!value || !PyArray_CheckExact(value) || PyArray_NDIM((PyArrayObject *)value)!=1) return PYJIT_RESULT_MISS;
            }
            op=NUMPY_CROSSJIT_DOT;
        }
        else if (callable == crossjit_state.add_outer ||
                 (PyCFunction_CheckExact(callable) && PyCFunction_CheckExact(crossjit_state.add_outer) &&
                  PyCFunction_GET_FUNCTION(callable)==PyCFunction_GET_FUNCTION(crossjit_state.add_outer) &&
                  PyCFunction_GET_SELF(callable)==crossjit_state.add)) op = NUMPY_CROSSJIT_ADD_OUTER;
        else if (callable == crossjit_state.mean) op = NUMPY_CROSSJIT_MEAN;
        else if (numpy_crossjit_same_callable(callable,crossjit_state.maximum_reduce)) op = NUMPY_CROSSJIT_MAXIMUM_REDUCE;
        else if (numpy_crossjit_same_callable(callable,crossjit_state.add_reduce)) op = NUMPY_CROSSJIT_ADD_REDUCE;
        else return PYJIT_RESULT_MISS;
    }
    else {
        return PYJIT_RESULT_MISS;
    }
    if (request->kind==PYJIT_OPERATION_CALL && request->call_descriptor_version!=PYJIT_CALL_DESCRIPTOR_VERSION) return PYJIT_RESULT_MISS;
    if (op>=NUMPY_CROSSJIT_SUM && !numpy_crossjit_alias_valid(op-NUMPY_CROSSJIT_SUM)) return PYJIT_RESULT_MISS;
    if (numpy_crossjit_is_reduction(op) && request->positional_count<1) return PYJIT_RESULT_MISS;
    if (op==NUMPY_CROSSJIT_DOT_PUBLIC) {
        if (request->operand_count!=2) return PYJIT_RESULT_MISS;
        for (int i=0;i<2;i++) {
            PyObject *object=PyJit_ValueGetObservedObject(request->operands[i]);
            if (!object || !PyArray_CheckExact(object) || PyArray_NDIM((PyArrayObject *)object)!=1) return PYJIT_RESULT_MISS;
        }
    }
    if (request->has_keywords && op!=NUMPY_CROSSJIT_MAXIMUM_REDUCE && op!=NUMPY_CROSSJIT_ADD_REDUCE && op!=NUMPY_CROSSJIT_SUM && op!=NUMPY_CROSSJIT_MAX && op!=NUMPY_CROSSJIT_AMAX) return PYJIT_RESULT_MISS;
    if (request->kind==PYJIT_OPERATION_CALL && op!=NUMPY_CROSSJIT_ADD_REDUCE && op!=NUMPY_CROSSJIT_MAXIMUM_REDUCE && op!=NUMPY_CROSSJIT_SUM && op!=NUMPY_CROSSJIT_MAX && op!=NUMPY_CROSSJIT_AMAX) {
        int arity=op==NUMPY_CROSSJIT_NEGATIVE || op==NUMPY_CROSSJIT_SQRT ||
            op==NUMPY_CROSSJIT_EXP || op==NUMPY_CROSSJIT_TANH || op==NUMPY_CROSSJIT_MEAN ? 1 : 2;
        if (request->operand_count!=arity) return PYJIT_RESULT_MISS;
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
            /* A zero-dimensional ufunc domain normally produces a NumPy
             * scalar. Shape-dependent facts remain conditional on the
             * Provider's eventual array guards, not ordinary Host typing. */
            if (PyArray_NDIM((PyArrayObject *)object) != 0) {
                has_array = 1;
            }
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
    int native_stage = op == NUMPY_CROSSJIT_MATMUL || op == NUMPY_CROSSJIT_ADD_OUTER ||
        op==NUMPY_CROSSJIT_ADD_REDUCE || op==NUMPY_CROSSJIT_MAXIMUM_REDUCE ||
        op==NUMPY_CROSSJIT_SQRT || op==NUMPY_CROSSJIT_EXP || op==NUMPY_CROSSJIT_TANH || op==NUMPY_CROSSJIT_DOT || op>=NUMPY_CROSSJIT_SUM;
    PyTypeObject *native_result_type = NULL;
    if (op == NUMPY_CROSSJIT_ADD_OUTER) native_result_type = &PyArray_Type;
    if ((op==NUMPY_CROSSJIT_SQRT || op==NUMPY_CROSSJIT_EXP || op==NUMPY_CROSSJIT_TANH) && has_array)
        native_result_type=&PyArray_Type;
    if ((op == NUMPY_CROSSJIT_MATMUL || op==NUMPY_CROSSJIT_DOT || op==NUMPY_CROSSJIT_DOT_PUBLIC) && request->operand_count == 2) {
        PyObject *left = PyJit_ValueGetObservedObject(request->operands[0]);
        PyObject *right = PyJit_ValueGetObservedObject(request->operands[1]);
        if (left && right && PyArray_CheckExact(left) && PyArray_CheckExact(right)) {
            PyArrayObject *a=(PyArrayObject *)left, *b=(PyArrayObject *)right;
            if (PyArray_NDIM(a)==1 && PyArray_NDIM(b)==1 && PyArray_TYPE(a)==PyArray_TYPE(b)) {
                if (PyArray_TYPE(a)==NPY_DOUBLE) native_result_type=&PyDoubleArrType_Type;
                if (PyArray_TYPE(a)==NPY_FLOAT) native_result_type=&PyFloatArrType_Type;
            }
            else if (PyArray_NDIM(a)>0 && PyArray_NDIM(b)>0) native_result_type=&PyArray_Type;
        }
    }
    if (numpy_crossjit_is_reduction(op)) {
        PyObject *axis=NULL,*keep=NULL;
        for (uint8_t i=1;i<request->operand_count;i++) {
            PyObject *value=PyJit_ValueGetObservedObject(request->operands[i]);
            const char *name=NULL;
            if (i<request->positional_count) name=i==1?"axis":i==((op==NUMPY_CROSSJIT_MAX || op==NUMPY_CROSSJIT_AMAX)?3:4)?"keepdims":NULL;
            else if (request->keyword_names) {
                PyObject *key=PyTuple_GET_ITEM(request->keyword_names,i-request->positional_count);
                if (PyUnicode_CheckExact(key)) name=PyUnicode_AsUTF8(key);
                if (PyErr_Occurred()) return PYJIT_RESULT_ERROR;
            }
            if (name && strcmp(name,"axis")==0) axis=value;
            if (name && strcmp(name,"keepdims")==0) keep=value;
        }
        PyObject *input=PyJit_ValueGetObservedObject(request->operands[0]);
        if (keep==Py_True && input && PyArray_CheckExact(input) && PyArray_NDIM((PyArrayObject *)input)>0) native_result_type=&PyArray_Type;
        else if (input && PyArray_CheckExact(input)) {
            PyArrayObject *a=(PyArrayObject *)input;
            int removed=(axis==Py_None || (!axis && op>=NUMPY_CROSSJIT_SUM))?PyArray_NDIM(a):axis && PyTuple_CheckExact(axis)?PyTuple_GET_SIZE(axis):1;
            if (PyArray_NDIM(a)>removed) native_result_type=&PyArray_Type;
            else if (PyArray_NDIM(a)==removed) {
                if (PyArray_TYPE(a)==NPY_DOUBLE) native_result_type=&PyDoubleArrType_Type;
                if (PyArray_TYPE(a)==NPY_FLOAT) native_result_type=&PyFloatArrType_Type;
            }
        }
    }
    *summary = (PyJitSemanticSummary){
        .struct_size = sizeof(PyJitSemanticSummary),
        .mutation = inplace ? PYJIT_MUTATION_ARGUMENT :
                              PYJIT_MUTATION_NONE,
        .result_alias = inplace ? PYJIT_ALIAS_MAY_ALIAS_ARGUMENT :
                                  PYJIT_ALIAS_FRESH,
        .materialization = !inplace && (elementwise || native_stage) && !may_call_host ?
            PYJIT_MATERIALIZATION_VIRTUALIZABLE :
            PYJIT_MATERIALIZATION_REQUIRED,
        .replayability = !inplace && !may_call_host ?
            PYJIT_REPLAYABLE : PYJIT_NOT_REPLAYABLE,
        .may_raise = 1,
        .may_call_host = (uint8_t)may_call_host,
        .exact_result_type = elementwise && has_array ? &PyArray_Type : native_result_type,
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
    if (typenum==NPY_INT && sizeof(npy_int)==4) return value>=INT32_MIN && value<=INT32_MAX;
    if ((typenum==NPY_LONG && sizeof(npy_long)==8) ||
        (typenum==NPY_LONGLONG && sizeof(npy_longlong)==8)) return 1;
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

/* Provider-owned proof: a common trailing domain can be traversed as one
 * affine row without changing the original ndarray shape or guard contract.
 * Start conservatively: exact domains, positive strides, no singleton axes or
 * array broadcasting. All original dimensions/strides remain runtime guards.
 */
static void
numpy_crossjit_coalesce_trailing_axes(struct PyJitGuardPlan *guards)
{
    uint8_t ndim = guards->domain.ndim;
    if (ndim <= 2) {
        return;
    }
    uint64_t inner = 1;
    for (uint8_t axis = 0; axis < ndim; axis++) {
        npy_intp dimension = guards->domain.dimensions[axis];
        if (dimension <= 1) {
            return;
        }
        if (axis != 0) {
            if (inner > (uint64_t)NPY_MAX_INTP / (uint64_t)dimension) {
                return;
            }
            inner *= (uint64_t)dimension;
        }
    }
    for (uint8_t i = 0; i < guards->input_count; i++) {
        if (guards->scalar_mask & (1u << i)) {
            continue;
        }
        const NumpyCrossJitArrayFact *input = &guards->inputs[i];
        if (input->ndim != ndim || input->strides[0] <= 0 ||
                input->strides[ndim - 1] <= 0) {
            return;
        }
        for (uint8_t axis = 0; axis < ndim; axis++) {
            if (input->dimensions[axis] != guards->domain.dimensions[axis]) {
                return;
            }
            if (axis > 0 && axis + 1 < ndim) {
                npy_intp step = input->strides[axis + 1];
                npy_intp size = input->dimensions[axis + 1];
                if (step <= 0 || step > NPY_MAX_INTP / size ||
                        input->strides[axis] != step * size) {
                    return;
                }
            }
        }
        /* Output byte-stride multiplication must also fit npy_intp. */
        if (inner > (uint64_t)NPY_MAX_INTP /
                (uint64_t)guards->iteration.output.inner_stride) {
            return;
        }
    }
    NumpyCrossJitIterationDescriptor *descriptor = &guards->iteration;
    descriptor->outer_shape_class = NUMPY_CROSSJIT_OUTER_AFFINE_ROWS;
    descriptor->outer_count = (uint64_t)guards->domain.dimensions[0];
    descriptor->inner_extent = inner;
    descriptor->output.outer_stride =
        (npy_intp)inner * descriptor->output.inner_stride;
    guards->outer_count = descriptor->outer_count;
    guards->inner_extent = inner;
    int all_inner_contiguous = 1;
    for (uint8_t i = 0; i < guards->input_count; i++) {
        if (guards->scalar_mask & (1u << i)) {
            continue;
        }
        NumpyCrossJitOperandDescriptor *operand = &descriptor->operands[i];
        operand->inner_stride = guards->inputs[i].strides[ndim - 1];
        operand->outer_stride = guards->inputs[i].strides[0];
        operand->broadcast_relation = NUMPY_CROSSJIT_BROADCAST_NONE;
        operand->access_kind = operand->inner_stride ==
            descriptor->output.inner_stride ?
            NUMPY_CROSSJIT_ACCESS_CONTIGUOUS :
            NUMPY_CROSSJIT_ACCESS_FIXED_STRIDE;
        guards->access_kinds[i] = operand->access_kind;
        if (operand->access_kind == NUMPY_CROSSJIT_ACCESS_FIXED_STRIDE) {
            descriptor->version = 2;
        }
        all_inner_contiguous &= operand->access_kind ==
            NUMPY_CROSSJIT_ACCESS_CONTIGUOUS;
    }
    if (!guards->scalar_mask && guards->refined_layout_class !=
            NUMPY_CROSSJIT_PROFILE_FULL_CONTIGUOUS) {
        guards->refined_layout_class = all_inner_contiguous ?
            NUMPY_CROSSJIT_PROFILE_INNER_CONTIGUOUS :
            NUMPY_CROSSJIT_PROFILE_FIXED_AFFINE_STRIDE;
    }
}

static void
numpy_crossjit_classify_profile_layout(const PyJitRegionView *region,
                                       struct PyJitGuardPlan *guards)
{
    uint8_t domain_ndim = guards->domain.ndim;
    uint64_t elements = numpy_crossjit_domain_elements(&guards->domain);
    guards->iteration_rank_class = domain_ndim <= 1 ? 1 : 2;
    guards->inner_extent = domain_ndim == 0 ? 1 :
        (uint64_t)guards->domain.dimensions[domain_ndim - 1];
    guards->outer_count = guards->inner_extent == 0 ? 0 :
        elements / guards->inner_extent;

    int any_scalar = 0;
    int any_zero_stride = 0;
    int all_fully_contiguous = 1;
    int all_inner_contiguous = domain_ndim != 0;
    int all_fixed_affine = domain_ndim >= 1 && domain_ndim <= 2;
    npy_intp domain_itemsize = 0;
    for (uint8_t i = 0; i < guards->input_count; i++) {
        if (guards->scalar_mask & (1u << i)) {
            guards->access_kinds[i] = NUMPY_CROSSJIT_ACCESS_SCALAR;
            any_scalar = 1;
            continue;
        }
        const NumpyCrossJitArrayFact *input = &guards->inputs[i];
        const PyJitRegionValue *region_input = PyJit_RegionGetInput(region, i);
        PyArrayObject *array = (PyArrayObject *)
            PyJit_ValueGetObservedObject(region_input->value);
        if (domain_itemsize == 0) {
            domain_itemsize = PyArray_ITEMSIZE(array);
        }
        int leading = (int)domain_ndim - input->ndim;
        int broadcast = 0;
        int exact_domain = input->ndim == domain_ndim;
        for (uint8_t axis = 0; axis < domain_ndim; axis++) {
            int input_axis = (int)axis - leading;
            if (input_axis < 0 ||
                    (input->dimensions[input_axis] == 1 &&
                     guards->domain.dimensions[axis] != 1)) {
                broadcast = 1;
                exact_domain = 0;
            }
            else if (input->dimensions[input_axis] !=
                     guards->domain.dimensions[axis]) {
                exact_domain = 0;
            }
        }
        int inner = input->ndim == 0 ||
            input->dimensions[input->ndim - 1] <= 1 ||
            input->strides[input->ndim - 1] == PyArray_ITEMSIZE(array);
        if (broadcast) {
            guards->access_kinds[i] = NUMPY_CROSSJIT_ACCESS_ZERO_STRIDE;
            any_zero_stride = 1;
        }
        else if (exact_domain && input->c_contiguous) {
            guards->access_kinds[i] = NUMPY_CROSSJIT_ACCESS_CONTIGUOUS;
        }
        else if (domain_ndim == 1) {
            guards->access_kinds[i] = NUMPY_CROSSJIT_ACCESS_FIXED_STRIDE;
        }
        else if (inner) {
            guards->access_kinds[i] = NUMPY_CROSSJIT_ACCESS_CONTIGUOUS;
        }
        else if (domain_ndim == 2 && exact_domain) {
            guards->access_kinds[i] = NUMPY_CROSSJIT_ACCESS_FIXED_STRIDE;
        }
        else {
            guards->access_kinds[i] =
                NUMPY_CROSSJIT_ACCESS_GENERAL_STRIDED;
        }
        all_fully_contiguous &= exact_domain && input->c_contiguous;
        all_inner_contiguous &= inner;
        all_fixed_affine &= guards->access_kinds[i] ==
            NUMPY_CROSSJIT_ACCESS_CONTIGUOUS ||
            guards->access_kinds[i] == NUMPY_CROSSJIT_ACCESS_FIXED_STRIDE;
    }

    if (any_scalar) {
        guards->refined_layout_class =
            NUMPY_CROSSJIT_PROFILE_SCALAR_OPERAND;
    }
    else if (any_zero_stride) {
        guards->refined_layout_class =
            NUMPY_CROSSJIT_PROFILE_ZERO_STRIDE_BROADCAST;
    }
    else if (all_fully_contiguous) {
        guards->refined_layout_class =
            NUMPY_CROSSJIT_PROFILE_FULL_CONTIGUOUS;
    }
    else if (all_inner_contiguous) {
        guards->refined_layout_class =
            NUMPY_CROSSJIT_PROFILE_INNER_CONTIGUOUS;
    }
    else if (all_fixed_affine) {
        guards->refined_layout_class =
            NUMPY_CROSSJIT_PROFILE_FIXED_AFFINE_STRIDE;
    }
    else {
        guards->refined_layout_class =
            NUMPY_CROSSJIT_PROFILE_TRULY_GENERAL_STRIDED;
    }

    NumpyCrossJitIterationDescriptor *descriptor = &guards->iteration;
    memset(descriptor, 0, sizeof(*descriptor));
    descriptor->version = NUMPY_CROSSJIT_ITERATION_DESCRIPTOR_VERSION;
    descriptor->iteration_rank_class = domain_ndim <= 1 ?
        NUMPY_CROSSJIT_ITERATION_FLAT :
        NUMPY_CROSSJIT_ITERATION_OUTER_INNER;
    descriptor->outer_shape_class = domain_ndim <= 1 ?
        NUMPY_CROSSJIT_OUTER_NONE :
        (domain_ndim == 2 ? NUMPY_CROSSJIT_OUTER_AFFINE_ROWS :
         NUMPY_CROSSJIT_OUTER_GENERAL);
    descriptor->operand_count = guards->input_count;
    descriptor->outer_count = guards->outer_count;
    descriptor->inner_extent = guards->inner_extent;
    for (uint8_t i = 0; i < guards->input_count; i++) {
        NumpyCrossJitOperandDescriptor *operand = &descriptor->operands[i];
        operand->logical_input_slot = i;
        if (guards->scalar_mask & (1u << i)) {
            operand->access_kind = NUMPY_CROSSJIT_ACCESS_SCALAR;
            operand->broadcast_relation = NUMPY_CROSSJIT_BROADCAST_SCALAR;
            continue;
        }
        const NumpyCrossJitArrayFact *input = &guards->inputs[i];
        int leading = (int)domain_ndim - input->ndim;
        int inner_axis = (int)domain_ndim - 1 - leading;
        int inner_broadcast = domain_ndim != 0 &&
            (inner_axis < 0 ||
             (input->dimensions[inner_axis] == 1 &&
              guards->domain.dimensions[domain_ndim - 1] != 1));
        int outer_broadcast = 0;
        int broadcast_axes = inner_broadcast;
        for (uint8_t axis = 0; axis + 1 < domain_ndim; axis++) {
            int input_axis = (int)axis - leading;
            int broadcast = input_axis < 0 ||
                (input->dimensions[input_axis] == 1 &&
                 guards->domain.dimensions[axis] != 1);
            outer_broadcast |= broadcast;
            broadcast_axes += broadcast;
        }
        operand->access_kind = guards->access_kinds[i] <=
            NUMPY_CROSSJIT_ACCESS_FIXED_STRIDE ?
            guards->access_kinds[i] : 0;
        operand->broadcast_relation = broadcast_axes > 1 ?
            NUMPY_CROSSJIT_BROADCAST_MULTIPLE :
            (inner_broadcast ? NUMPY_CROSSJIT_BROADCAST_INNER :
             (outer_broadcast ? NUMPY_CROSSJIT_BROADCAST_OUTER :
              NUMPY_CROSSJIT_BROADCAST_NONE));
        operand->inner_stride = inner_broadcast || domain_ndim == 0 ? 0 :
            input->strides[inner_axis];
        if (operand->access_kind == NUMPY_CROSSJIT_ACCESS_FIXED_STRIDE) {
            descriptor->version = 2;
        }
        if (domain_ndim == 2) {
            int outer_axis = -leading;
            operand->outer_stride = outer_axis < 0 || outer_broadcast ? 0 :
                input->strides[outer_axis];
        }
    }
    descriptor->output.access_kind = NUMPY_CROSSJIT_ACCESS_CONTIGUOUS;
    descriptor->output.inner_stride = domain_itemsize;
    descriptor->output.outer_stride = domain_ndim <= 1 ? 0 :
        (npy_intp)(descriptor->inner_extent *
                   (uint64_t)descriptor->output.inner_stride);
    numpy_crossjit_coalesce_trailing_axes(guards);
}

static uint64_t
numpy_crossjit_int64_class(PyObject *object)
{
    if (sizeof(npy_long)==8 && Py_TYPE(object)==&PyLongArrType_Type) return 7;
    if (sizeof(npy_longlong)==8 && Py_TYPE(object)==&PyLongLongArrType_Type) return 8;
    return 0;
}

static npy_int64
numpy_crossjit_int64_value(PyObject *object)
{
    return Py_TYPE(object)==&PyLongArrType_Type?(npy_int64)PyArrayScalar_VAL(object,Long):
        (npy_int64)PyArrayScalar_VAL(object,LongLong);
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
            guards->scalar_class_mask |= UINT64_C(1) << (4 * i);
            continue;
        }
        if (object != NULL && Py_TYPE(object) == &PyDoubleArrType_Type) {
            guards->scalar_mask |= (uint16_t)(1u << i);
            guards->double_scalar_mask |= (uint16_t)(1u << i);
            guards->scalar_class_mask |= UINT64_C(2) << (4 * i);
            continue;
        }
        if (object != NULL && Py_TYPE(object) == &PyFloatArrType_Type) {
            guards->scalar_mask |= (uint16_t)(1u << i);
            guards->float_scalar_mask |= (uint16_t)(1u << i);
            guards->scalar_class_mask |= UINT64_C(6) << (4 * i);
            continue;
        }
        if (object != NULL && numpy_crossjit_int64_class(object)) {
            guards->scalar_mask|=1u<<i;
            guards->numpy_int_scalar_mask|=1u<<i;
            guards->scalar_class_mask|=numpy_crossjit_int64_class(object)<<(4*i);
            continue;
        }
        if (object != NULL && PyLong_CheckExact(object)) {
            guards->scalar_mask |= (uint16_t)(1u << i);
            guards->int_scalar_mask |= (uint16_t)(1u << i);
            long long value = PyLong_AsLongLong(object);
            if (value == -1 && PyErr_Occurred()) {
                PyErr_Clear();
                return numpy_crossjit_lowering_reject(
                    NUMPY_CROSSJIT_REJECT_INPUT_KIND, "integer scalar exceeds conversion class");
            }
            uint64_t cls = value >= -INT64_C(1073741823) &&
                value <= INT64_C(1073741823) ? 3 : 4;
            guards->scalar_class_mask |= cls << (4 * i);
            continue;
        }
        if (object == NULL || !PyArray_CheckExact(object) ||
            !numpy_crossjit_array_fact(
                (PyArrayObject *)object, &guards->inputs[i]) ||
            !numpy_crossjit_stage_c_order(&guards->inputs[i])) {
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
        /* Every intermediate and the final result are allocated in C order.
         * NumPy's default KEEPORDER axis ordering must therefore be compatible
         * with C order for each live-in, and consequently for each DAG node.
         * Nontrivial, nonzero strides must decrease from outer to inner axes.
         * Reject conflicting/Fortran preferences before lowering, including
         * diagnostic generic execution.  Exact runtime stride guards preserve
         * this private proof.  Use unsigned magnitudes to handle INTP_MIN. */
        npy_uintp previous_stride = NPY_MAX_UINTP;
        for (uint8_t axis = 0; axis < fact->ndim; axis++) {
            npy_intp stride = fact->strides[axis];
            if (fact->dimensions[axis] <= 1 || stride == 0) {
                continue;
            }
            npy_uintp magnitude = stride < 0 ?
                (npy_uintp)0 - (npy_uintp)stride : (npy_uintp)stride;
            if (magnitude > previous_stride) {
                return numpy_crossjit_lowering_reject(
                    NUMPY_CROSSJIT_REJECT_DTYPE_LAYOUT,
                    "KEEPORDER output is not proven C-compatible");
            }
            previous_stride = magnitude;
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
    if (domain_ndim == 0) {
        /* Both generated and generic plans publish fresh ndarrays. They do
         * not implement NumPy scalar publication for a zero-dimensional
         * ufunc domain, so preserve the original operation instead. */
        return numpy_crossjit_lowering_reject(
            NUMPY_CROSSJIT_REJECT_DTYPE_LAYOUT,
            "zero-dimensional scalar publication is unsupported");
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
    if ((guards->scalar_mask & ~(guards->int_scalar_mask|guards->numpy_int_scalar_mask)) != 0 && guards->domain.typenum != NPY_FLOAT &&
        guards->domain.typenum != NPY_DOUBLE) {
        return numpy_crossjit_lowering_reject(
            NUMPY_CROSSJIT_REJECT_SCALAR_PROMOTION,
            "Python float would promote the integer array domain");
    }
    /* NumPy scalars use strong dtype promotion.  Unlike Python floats, an
     * exact np.float64 would widen float32, so that domain remains generic. */
    if (guards->double_scalar_mask != 0 &&
            guards->domain.typenum != NPY_DOUBLE) {
        return numpy_crossjit_lowering_reject(
            NUMPY_CROSSJIT_REJECT_SCALAR_PROMOTION,
            "NumPy float64 scalar would change the array result dtype");
    }
    if (guards->float_scalar_mask != 0 && guards->domain.typenum != NPY_FLOAT &&
            guards->domain.typenum != NPY_DOUBLE) {
        return 0;
    }
    if (guards->numpy_int_scalar_mask && guards->domain.typenum!=NPY_DOUBLE &&
        guards->domain.typenum!=NPY_LONG && guards->domain.typenum!=NPY_LONGLONG) return 0;
    for (uint8_t i=0;i<guards->input_count;i++) {
        if (guards->numpy_int_scalar_mask&(1u<<i)) {
            uint64_t cls=(guards->scalar_class_mask>>(4*i))&15;
            if (guards->domain.typenum!=NPY_DOUBLE &&
                guards->domain.typenum!=(cls==7?NPY_LONG:NPY_LONGLONG)) return 0;
        }
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
    numpy_crossjit_classify_profile_layout(region, guards);
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

/* Native selection is Provider-owned and never depends on workload names. */
static int
numpy_crossjit_key_append(char *key, size_t capacity, size_t *used,
                         const char *format, ...)
{
    va_list args;
    va_start(args, format);
    int written = vsnprintf(key + *used, capacity - *used, format, args);
    va_end(args);
    if (written < 0 || (size_t)written >= capacity - *used) {
        return 0;
    }
    *used += (size_t)written;
    return 1;
}

static int
numpy_crossjit_call_key(const PyJitRegionView *region,struct PyJitExecPlan *plan)
{
    size_t used=strlen(plan->native_key);
    for (uint8_t i=0;i<plan->node_count;i++) {
        const PyJitNodeRef *node=PyJit_RegionGetNode(region,i);
        int positional=PyJit_NodeGetPositionalCount(node);
        PyObject *names=PyJit_NodeGetKeywordNames(node);
        if (!positional && !names) continue;
        int count=names?PyTuple_GET_SIZE(names):0;
        if (!numpy_crossjit_key_append(plan->native_key,sizeof(plan->native_key),&used,"|h:%u:%d:%u:%d:",
                i,positional,PyJit_NodeGetCallBindingKind(node),count)) return 0;
        for (int j=0;j<count;j++) {
            PyObject *name=PyTuple_GET_ITEM(names,j);
            Py_ssize_t length=0;
            const char *bytes=PyUnicode_AsUTF8AndSize(name,&length);
            if (!bytes || !numpy_crossjit_key_append(plan->native_key,sizeof(plan->native_key),&used,
                    "%lld:%s,",(long long)length,bytes)) return 0;
        }
    }
    return 1;
}

static int
numpy_crossjit_native_key(struct PyJitExecPlan *plan)
{
    size_t used = 0;
    char *key = plan->native_key;
#define KEY(...) do { if (!numpy_crossjit_key_append( \
    key, sizeof(plan->native_key), &used, __VA_ARGS__)) return 0; } while (0)
    KEY("E1|%d|%u|%u|%u|%u|", plan->guards.domain.typenum,
        plan->guards.domain.ndim, plan->guards.input_count, plan->node_count,
        plan->output_slot);
    for (uint8_t axis = 0; axis < plan->guards.domain.ndim; axis++) {
        KEY("%lld,", (long long)plan->guards.domain.dimensions[axis]);
    }
    for (uint8_t i = 0; i < plan->guards.input_count; i++) {
        if (plan->guards.constant_mask & (1u<<i)) {
            const NumpyCrossJitArrayFact *fact=&plan->guards.inputs[i];
            KEY("|c:%u:%u:",plan->guards.constant_kinds[i],fact->ndim);
            for (uint8_t j=0;j<fact->ndim;j++) KEY("%lld,",(long long)fact->dimensions[j]);
        }
        else if (plan->guards.scalar_mask & (1u << i)) {
            KEY("|s:%llu", (unsigned long long)((plan->guards.scalar_class_mask >> (4*i)) & 15));
        }
        else {
            const NumpyCrossJitArrayFact *fact = &plan->guards.inputs[i];
            KEY("|a:%d:%u:", fact->typenum, fact->ndim);
            for (uint8_t axis = 0; axis < fact->ndim; axis++) {
                KEY("%lld,%lld;", (long long)fact->dimensions[axis],
                    (long long)fact->strides[axis]);
            }
        }
    }
    for (uint8_t i = 0; i < plan->node_count; i++) {
        KEY("|n:%u:%u:%u:%u", plan->nodes[i].op, plan->nodes[i].left,
            plan->nodes[i].right, plan->nodes[i].output);
    }
#undef KEY
    for (uint8_t i=0;i<plan->stage_count;i++) {
        NumpyCrossJitStage *stage=&plan->stages[i];
        if (numpy_crossjit_is_reduction(stage->semantic_op)) {
            if (!numpy_crossjit_key_append(key,sizeof(plan->native_key),&used,"|r:%u:%u:%u:",i,stage->keepdims,stage->axis_count)) return 0;
            for (uint8_t j=0;j<stage->axis_count;j++) {
                if (!numpy_crossjit_key_append(key,sizeof(plan->native_key),&used,"%d,",stage->axes[j])) return 0;
            }
        }
    }
    if (!numpy_crossjit_key_append(key,sizeof(plan->native_key),&used,"|b:3")) return 0;
    return 1;
}

static int
numpy_crossjit_native_measured(const char *key)
{
    size_t low = 0, high = NUMPY_CROSSJIT_NATIVE_POLICY_COUNT;
    while (low < high) {
        size_t mid = low + (high-low)/2;
        int cmp = strcmp(key, numpy_crossjit_native_policy_keys[mid]);
        if (cmp == 0) return 1;
        if (cmp < 0) high = mid;
        else low = mid + 1;
    }
    return 0;
}

/* Opt-in discovery only. No saved object/array pointer becomes a runtime key. */
static int
numpy_crossjit_native_observe(const char *key, const char *event)
{
    if (!crossjit_stats.diagnostics_enabled) return 1;
    if (native_observations == NULL) {
        native_observations = PyDict_New();
        if (native_observations == NULL) return 0;
    }
    PyObject *row = PyDict_GetItemString(native_observations, key);
    if (row == NULL) {
        row = PyDict_New();
        if (row == NULL) return 0;
        int status = PyDict_SetItemString(native_observations, key, row);
        Py_DECREF(row);
        if (status < 0) return 0;
        row = PyDict_GetItemString(native_observations, key);
    }
    PyObject *old = PyDict_GetItemString(row, event);
    unsigned long long count = old == NULL ? 0 : PyLong_AsUnsignedLongLong(old);
    if (PyErr_Occurred()) return 0;
    PyObject *next = PyLong_FromUnsignedLongLong(count + 1);
    if (next == NULL) return 0;
    int status = PyDict_SetItemString(row, event, next);
    Py_DECREF(next);
    return status == 0;
}

#include "jit_execution_audit.h"

static size_t
numpy_crossjit_dtype_width(int dtype)
{
    return dtype==NPY_FLOAT || dtype==NPY_INT?4:8;
}

static PyJitResult numpy_crossjit_record_exec(PyJitResult, uint64_t, uint64_t, uint64_t);
#include "jit_native_loops.h"
#include "jit_native_stages.h"

/* Builds the immutable provider program after Host validation succeeds. */
static PyJitPrepareDecision
numpy_crossjit_build_exec(const PyJitRegionView *region,
                          struct PyJitExecPlan *plan)
{
    memset(plan, 0, sizeof(*plan));
    if (crossjit_stats.diagnostics_enabled) {
        memset(&crossjit_stats.last, 0, sizeof(crossjit_stats.last));
    }
    for (uint8_t i = 0; i < PyJit_RegionGetNodeCount(region); i++) {
        uint64_t semantic = PyJit_NodeGetSemanticId(PyJit_RegionGetNode(region, i));
        if (semantic == NUMPY_CROSSJIT_MATMUL + 1 || semantic == NUMPY_CROSSJIT_ADD_OUTER + 1 ||
                semantic == NUMPY_CROSSJIT_ADD_REDUCE + 1 || semantic == NUMPY_CROSSJIT_MAXIMUM_REDUCE + 1 ||
                semantic == NUMPY_CROSSJIT_SQRT+1 || semantic == NUMPY_CROSSJIT_EXP+1 || semantic == NUMPY_CROSSJIT_TANH+1 || semantic == NUMPY_CROSSJIT_DOT+1 || semantic >= NUMPY_CROSSJIT_SUM+1 ||
                semantic == NUMPY_CROSSJIT_MINIMUM+1 || semantic == NUMPY_CROSSJIT_MAXIMUM+1) {
            return numpy_crossjit_build_stages(region, plan);
        }
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
    if (plan->node_count == 0 ||
            plan->node_count > NUMPY_CROSSJIT_MAX_NODES) {
        numpy_crossjit_lowering_reject(
            NUMPY_CROSSJIT_REJECT_NODE_COUNT, "unsupported node count");
        return PYJIT_PREPARE_UNSUPPORTED_NODE;
    }
    uint64_t elements = numpy_crossjit_domain_elements(&plan->guards.domain);
    uint64_t eliminated = (uint64_t)(plan->node_count - 1);
    /* Single-operation candidates eliminate extension dispatch, not an array.
     * This cheap gate is only a reject filter. Exact measured DecisionDB cells
     * remain mandatory for production execution. */
    uint64_t work_factor = eliminated ? eliminated : 1;
    uint64_t work = elements > UINT64_MAX / work_factor ? UINT64_MAX :
        elements * work_factor;
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
            if (node != NULL) {
                numpy_crossjit_record_unsupported_node(
                    plan, PyJit_NodeGetSemanticId(node), i);
            }
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
        plan->guards.builtin_ufunc_mask|=1u<<lowered->op;
    }
    plan->output_slot = plan->guards.input_count + plan->node_count - 1;
    numpy_crossjit_record_observed_signature(
        plan, NUMPY_CROSSJIT_OBSERVED_PREPARE);
    NumpyCrossJitPolicyDecision policy = NUMPY_CROSSJIT_POLICY_ORIGINAL;
    if (plan->guards.iteration_class ==
            NUMPY_CROSSJIT_CONTIGUOUS_SAME_SHAPE &&
        plan->guards.domain.typenum == NPY_DOUBLE) {
        policy = numpy_crossjit_generated_registry_lookup(
            plan->nodes, plan->node_count, plan->guards.input_count,
            plan->guards.scalar_mask, plan->output_slot, elements,
            &plan->fixed);
    }
    const char *force_profiled_generated = Py_GETENV(
        "NPY_CROSSJIT_PROFILED_FORCE_GENERATED");
    if (policy == NUMPY_CROSSJIT_POLICY_ORIGINAL &&
            plan->guards.domain.typenum == NPY_DOUBLE) {
        int force_profiled = force_profiled_generated != NULL &&
            strcmp(force_profiled_generated, "1") == 0;
        policy = numpy_crossjit_profiled_iteration_registry_lookup(
                plan->nodes, plan->node_count, plan->guards.input_count,
                &plan->guards.iteration, plan->output_slot, elements,
                plan->guards.scalar_class_mask, force_profiled, &plan->fixed);
    }
    /* Existing generated decisions retain priority and identity. Native
     * calibration bypasses only measured admission, never semantic proof. */
    if (policy == NUMPY_CROSSJIT_POLICY_ORIGINAL && numpy_crossjit_native_key(plan)) {
        if (plan->node_count==1) numpy_crossjit_select_native_loop(plan->nodes[0].op,plan->guards.domain.typenum,&plan->loop);
        if (plan->loop.entry) plan->native_key[1]='2';
        if (!numpy_crossjit_call_key(region,plan)) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
        if (!numpy_crossjit_native_observe(plan->native_key, "prepare")) {
            return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
        }
        const char *calibration = Py_GETENV("NPY_CROSSJIT_NATIVE_CALIBRATION");
        if (numpy_crossjit_native_measured(plan->native_key) ||
                (calibration != NULL && strcmp(calibration, "1") == 0)) {
            policy = NUMPY_CROSSJIT_POLICY_GENERIC;
            plan->native_backend = 1;
            plan->fixed.scalar_classes_required = 1;
            crossjit_stats.native_plans_built++;
        }
    }
    int has_calls=0;
    for (uint8_t i=0;i<plan->node_count;i++) {
        const PyJitNodeRef *node=PyJit_RegionGetNode(region,i);
        has_calls|=PyJit_NodeGetPositionalCount(node)!=0 || PyJit_NodeGetKeywordNames(node)!=NULL;
    }
    int new_scalar_classes=plan->guards.float_scalar_mask || plan->guards.numpy_int_scalar_mask;
    if (policy==NUMPY_CROSSJIT_POLICY_GENERATED && (has_calls || new_scalar_classes)) {
        if (!numpy_crossjit_native_key(plan)) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
        plan->native_key[0]='G';
        if (!numpy_crossjit_call_key(region,plan) || !numpy_crossjit_native_observe(plan->native_key,"prepare"))
            return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
        const char *calibration=Py_GETENV("NPY_CROSSJIT_NATIVE_CALIBRATION");
        if (!numpy_crossjit_native_measured(plan->native_key) &&
            !(calibration && strcmp(calibration,"1")==0)) return PYJIT_PREPARE_UNPROFITABLE;
        plan->native_backend=3; /* complete generated plan with new call guards */
        plan->fixed.scalar_classes_required=1;
    }
    plan->guards.guard_scalar_classes = plan->fixed.scalar_classes_required;
    const char *disable_generated = Py_GETENV(
        "NPY_CROSSJIT_DIAGNOSTIC_DISABLE_GENERATED");
    if (policy == NUMPY_CROSSJIT_POLICY_GENERATED &&
            disable_generated != NULL && strcmp(disable_generated, "1") == 0) {
        memset(&plan->fixed, 0, sizeof(plan->fixed));
        policy = NUMPY_CROSSJIT_POLICY_GENERIC;
    }
    const char *force_fallback = Py_GETENV(
        "NPY_CROSSJIT_DIAGNOSTIC_FORCE_FALLBACK");
    if (policy == NUMPY_CROSSJIT_POLICY_ORIGINAL &&
            force_fallback != NULL && strcmp(force_fallback, "1") == 0) {
        policy = NUMPY_CROSSJIT_POLICY_GENERIC;
    }
    if (policy == NUMPY_CROSSJIT_POLICY_GENERATED) {
        crossjit_stats.fixed_fast_plans_built++;
        crossjit_stats.generated_registry_hit++;
    }
    else if (policy == NUMPY_CROSSJIT_POLICY_GENERIC) {
        crossjit_stats.fixed_generic_plans_built++;
        crossjit_stats.generated_registry_miss++;
    }
    else {
        crossjit_stats.generated_registry_miss++;
        numpy_crossjit_lowering_reject(
            NUMPY_CROSSJIT_REJECT_UNPROFITABLE,
            "DecisionDB has no measured-profitable runtime path");
        return PYJIT_PREPARE_UNPROFITABLE;
    }
    numpy_crossjit_record_observed_signature(
        plan, NUMPY_CROSSJIT_OBSERVED_ACCEPTED);
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
    if (PyErr_Occurred()) {
        PyMem_Free(contract);
        return PYJIT_RESULT_ERROR;
    }
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
            ((contract->execution.guards.double_scalar_mask & (1u << i)) ?
                3 : (contract->execution.guards.int_scalar_mask & (1u << i)) ?
                2 : 1) : 0;
        if (contract->execution.guards.guard_scalar_classes &&
                ((contract->execution.guards.int_scalar_mask & (1u << i)) ||
                 (contract->execution.native_backend && (contract->execution.guards.scalar_mask & (1u << i))))) {
            contract->scalar_keys[i] = 1 + (uint8_t)(
                (contract->execution.guards.scalar_class_mask >> (4 * i)) & 15);
        }
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
        if (plan->constant_mask&(1u<<i)) {
            NumpyCrossJitArrayFact current={0};
            int kind=numpy_crossjit_constant_fact(inputs[i],&current);
            const NumpyCrossJitArrayFact *expected=&plan->inputs[i];
            if (!kind || kind!=plan->constant_kinds[i] || current.ndim!=expected->ndim ||
                    memcmp(current.dimensions,expected->dimensions,current.ndim*sizeof(npy_intp))) {
                return numpy_crossjit_record_guard(PYJIT_RESULT_MISS,started);
            }
            continue;
        }
        if (plan->scalar_mask & (1u << i)) {
            if (crossjit_stats.diagnostics_enabled) {
                crossjit_stats.guard_checks++;
            }
            if (plan->numpy_int_scalar_mask & (1u<<i)) {
                if (numpy_crossjit_int64_class(inputs[i]) != ((plan->scalar_class_mask>>(4*i))&15))
                    return numpy_crossjit_record_guard(PYJIT_RESULT_MISS,started);
            }
            else if (plan->int_scalar_mask & (1u << i)) {
                if (!numpy_crossjit_safe_int_scalar(
                        inputs[i], plan->domain.typenum)) {
                    return numpy_crossjit_record_guard(
                        PYJIT_RESULT_MISS, started);
                }
                /* Conversion cost is part of class-sensitive profitability.
                 * A changed magnitude must be rejected before execution. */
                if (plan->guard_scalar_classes) {
                    long long value = PyLong_AsLongLong(inputs[i]);
                    uint64_t cls = value >= -INT64_C(1073741823) &&
                        value <= INT64_C(1073741823) ? 3 : 4;
                    if (((plan->scalar_class_mask >> (4 * i)) & 15) != cls) {
                        return numpy_crossjit_record_guard(PYJIT_RESULT_MISS, started);
                    }
                }
            }
            else if (plan->double_scalar_mask & (1u << i)) {
                if (Py_TYPE(inputs[i]) != &PyDoubleArrType_Type) {
                    return numpy_crossjit_record_guard(
                        PYJIT_RESULT_MISS, started);
                }
            }
            else if (plan->float_scalar_mask & (1u << i)) {
                if (Py_TYPE(inputs[i]) != &PyFloatArrType_Type) {
                    return numpy_crossjit_record_guard(PYJIT_RESULT_MISS, started);
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
        PyArrayObject *array=(PyArrayObject *)inputs[i];
        const NumpyCrossJitArrayFact *expected = &plan->inputs[i];
        int ndim=PyArray_NDIM(array);
        /* Read the immutable contract fields directly. Copying and clearing
         * an NPY_MAXDIMS-sized fact on every HIT or shape MISS is unnecessary.
         * Keep every metadata and callback-free owner check from preparation. */
        if (PyArray_TYPE(array)!=expected->typenum || ndim!=expected->ndim ||
                PyArray_IS_C_CONTIGUOUS(array)!=expected->c_contiguous ||
                PyArray_ISNOTSWAPPED(array)!=expected->native_byteorder ||
                PyArray_ISALIGNED(array)!=expected->aligned ||
                memcmp(PyArray_DIMS(array),expected->dimensions,ndim*sizeof(npy_intp))) {
            return numpy_crossjit_record_guard(
                PYJIT_RESULT_MISS, started);
        }
        if (crossjit_stats.diagnostics_enabled) {
            crossjit_stats.guard_metadata_reads +=
                5 + (uint64_t)ndim * 2;
        }
        if (crossjit_stats.diagnostics_enabled) {
            crossjit_stats.guard_checks += 5 + ndim;
        }
        if (!numpy_crossjit_safe_array_owner(array)) {
            return numpy_crossjit_record_guard(
                PYJIT_RESULT_MISS, started);
        }
        /* Contiguous strides are derived from dtype and dimensions.  The
         * remaining classes retain exact stride guards for their kernels. */
        if ((plan->staged || (plan->iteration_class !=
                NUMPY_CROSSJIT_CONTIGUOUS_SAME_SHAPE &&
            plan->iteration_class != NUMPY_CROSSJIT_CONTIGUOUS_BROADCAST)) &&
            memcmp(PyArray_STRIDES(array), expected->strides,
                   (size_t)ndim * sizeof(npy_intp)) != 0) {
            if (crossjit_stats.diagnostics_enabled) {
                crossjit_stats.guard_checks += ndim;
            }
            return numpy_crossjit_record_guard(
                PYJIT_RESULT_MISS, started);
        }
    }
    /* Reject changed array/scalar contracts before checking shared compute
     * dependencies. No computation or publication precedes any guard. */
    PyObject *allocation_handler=PyDataMem_GetHandler();
    if (!allocation_handler) return numpy_crossjit_record_guard(PYJIT_RESULT_ERROR,started);
    int default_allocation=allocation_handler==PyDataMem_DefaultHandler;
    Py_DECREF(allocation_handler);
    if (!default_allocation) return numpy_crossjit_record_guard(PYJIT_RESULT_MISS,started);
    for (int loop=0;loop<11;loop++) {
        if ((plan->builtin_ufunc_mask&(1u<<loop)) && !numpy_crossjit_builtin_unchanged(loop))
            return numpy_crossjit_record_guard(PYJIT_RESULT_MISS,started);
    }
    for (uint8_t alias=0;alias<4;alias++) {
        if ((plan->alias_mask&(1u<<alias)) && !numpy_crossjit_alias_valid(alias))
            return numpy_crossjit_record_guard(PYJIT_RESULT_MISS,started);
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
numpy_crossjit_execute_validated(PyJitExecPlan *opaque, PyObject *const *inputs,
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
    numpy_crossjit_record_observed_signature(
        plan, NUMPY_CROSSJIT_OBSERVED_EXECUTION);
    if (plan->stage_count) {
        return numpy_crossjit_execute_stages(plan, inputs, result, started);
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
    char fp_barrier;
    int floating = fact->typenum == NPY_FLOAT || fact->typenum == NPY_DOUBLE;
    uint64_t kernel_started = 0;
    int ok = 1;
    if (plan->fixed.entry != NULL || plan->fixed.outer_inner_entry != NULL ||
            plan->fixed.affine_entry != NULL) {
        const void *role_inputs[NUMPY_CROSSJIT_MAX_FAST_INPUTS];
        ptrdiff_t inner_strides[NUMPY_CROSSJIT_MAX_FAST_INPUTS];
        ptrdiff_t outer_strides[NUMPY_CROSSJIT_MAX_FAST_INPUTS];
        double scalar_values[NUMPY_CROSSJIT_MAX_FAST_INPUTS];
        for (uint8_t role = 0; role < plan->fixed.input_count; role++) {
            uint8_t slot = plan->fixed.input_slots[role];
            if (plan->guards.scalar_mask & (1u << slot)) {
                double value = plan->guards.numpy_int_scalar_mask & (1u<<slot) ?
                    (double)numpy_crossjit_int64_value(inputs[slot]) : plan->guards.int_scalar_mask & (1u << slot) ?
                    PyLong_AsDouble(inputs[slot]) :
                    (plan->guards.float_scalar_mask & (1u << slot) ?
                     (double)PyArrayScalar_VAL(inputs[slot], Float) : PyFloat_AS_DOUBLE(inputs[slot]));
                if (value == -1.0 && PyErr_Occurred()) {
                    PyErr_Clear();
                    Py_DECREF(output);
                    return numpy_crossjit_record_exec(
                        PYJIT_RESULT_MISS, started, allocation_ns, kernel_ns);
                }
                scalar_values[role] = value;
                role_inputs[role] = &scalar_values[role];
                outer_strides[role] = 0;
                inner_strides[role] = 0;
            }
            else {
                role_inputs[role] = PyArray_DATA(
                    (PyArrayObject *)inputs[slot]);
                outer_strides[role] =
                    plan->guards.iteration.operands[slot].outer_stride;
                inner_strides[role] =
                    plan->guards.iteration.operands[slot].inner_stride;
            }
        }
        npy_clear_floatstatus_barrier(&fp_barrier);
        kernel_started = numpy_crossjit_clock_ns();
        if (plan->fixed.entry != NULL) {
            plan->fixed.entry(role_inputs, (double *)PyArray_DATA(output),
                              (size_t)PyArray_SIZE(output));
        }
        else if (plan->fixed.affine_entry != NULL) {
            plan->fixed.affine_entry(
                role_inputs, inner_strides, outer_strides,
                (double *)PyArray_DATA(output),
                (size_t)plan->guards.iteration.outer_count,
                (size_t)plan->guards.iteration.inner_extent);
        }
        else {
            plan->fixed.outer_inner_entry(
                role_inputs, outer_strides,
                (double *)PyArray_DATA(output),
                (size_t)plan->guards.iteration.outer_count,
                (size_t)plan->guards.iteration.inner_extent);
        }
        crossjit_stats.fixed_fast_path_hit++;
        crossjit_stats.generated_exec_hit++;
        numpy_crossjit_record_observed_signature(
            plan, NUMPY_CROSSJIT_OBSERVED_GENERATED);
    }
    else {
        const void *data[NUMPY_CROSSJIT_MAX_INPUTS];
        npy_intp input_strides[NUMPY_CROSSJIT_MAX_INPUTS][NPY_MAXDIMS] = {{0}};
        double scalar_double[NUMPY_CROSSJIT_MAX_INPUTS];
        float scalar_float[NUMPY_CROSSJIT_MAX_INPUTS];
        npy_int64 scalar_int[NUMPY_CROSSJIT_MAX_INPUTS];
        npy_int32 scalar_int32[NUMPY_CROSSJIT_MAX_INPUTS];
        int linear_domain = 1;
        for (uint8_t i = 0; i < plan->guards.input_count; i++) {
            if (plan->guards.scalar_mask & (1u << i)) {
                if (fact->typenum!=NPY_FLOAT && fact->typenum!=NPY_DOUBLE) {
                    npy_int64 value=plan->guards.numpy_int_scalar_mask&(1u<<i)?
                        numpy_crossjit_int64_value(inputs[i]):PyLong_AsLongLong(inputs[i]);
                    if (PyErr_Occurred()) {PyErr_Clear();Py_DECREF(output);return numpy_crossjit_record_exec(PYJIT_RESULT_MISS,started,allocation_ns,kernel_ns);}
                    if (fact->typenum==NPY_INT) {scalar_int32[i]=(npy_int32)value;data[i]=&scalar_int32[i];}
                    else {scalar_int[i]=value;data[i]=&scalar_int[i];}
                    continue;
                }
                double value = plan->guards.numpy_int_scalar_mask & (1u<<i) ?
                    (double)numpy_crossjit_int64_value(inputs[i]) : plan->guards.int_scalar_mask & (1u << i) ?
                    PyLong_AsDouble(inputs[i]) :
                    (plan->guards.float_scalar_mask & (1u << i) ?
                     (double)PyArrayScalar_VAL(inputs[i], Float) : PyFloat_AS_DOUBLE(inputs[i]));
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
                    if (input_axis < 0 ||
                        PyArray_DIM(array, input_axis) == 1) {
                        input_strides[i][axis] = 0;
                    }
                    else {
                        input_strides[i][axis] =
                            PyArray_STRIDE(array, input_axis);
                    }
                }
            }
        }
        npy_intp linear_dimension = PyArray_SIZE(output);
        if (linear_domain) {
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
        if (floating) {
            npy_clear_floatstatus_barrier(&fp_barrier);
        }
        kernel_started = numpy_crossjit_clock_ns();
        ok = plan->loop.entry ? numpy_crossjit_native_loop(&plan->loop,&plan->nodes[0],data,input_strides,
            linear_domain?1:fact->ndim,linear_domain?&linear_dimension:fact->dimensions,
            PyArray_BYTES(output),PyArray_SIZE(output),PyArray_ITEMSIZE(output)) : numpy_crossjit_kernel_strided(
            plan->nodes, plan->node_count, plan->guards.input_count,
            plan->guards.scalar_mask, plan->output_slot, fact->typenum,
            data, input_strides, linear_domain ? 1 : fact->ndim,
            linear_domain ? &linear_dimension : fact->dimensions,
            PyArray_DATA(output), PyArray_SIZE(output));
        if (ok) {
            crossjit_stats.fixed_generic_path_hit++;
            crossjit_stats.generic_fallback_hit++;
            numpy_crossjit_record_observed_signature(
                plan, NUMPY_CROSSJIT_OBSERVED_GENERIC);
        }
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
    if (plan->native_backend) {
        if (plan->native_backend!=3) crossjit_stats.native_exec_hit++;
        if (!numpy_crossjit_native_observe(plan->native_key, "execute")) {
            Py_DECREF(output);
            return numpy_crossjit_record_exec(PYJIT_RESULT_ERROR, started, allocation_ns, kernel_ns);
        }
    }
    *result = (PyObject *)output;
    return numpy_crossjit_record_exec(
        PYJIT_RESULT_HIT, started, allocation_ns, kernel_ns);
}

/* Required separate-callback ABI entry; also used by guard-combining ablations.
 * Independently validate inputs before calling the execution implementation. */
static PyJitResult
numpy_crossjit_run_region(PyJitExecPlan *opaque, PyObject *const *inputs,
                          Py_ssize_t input_count, PyObject **result)
{
    struct PyJitExecPlan *plan = (struct PyJitExecPlan *)opaque;
    *result = NULL;
    PyJitResult guard_status = plan == NULL ? PYJIT_RESULT_MISS :
        numpy_crossjit_run_guards(&plan->guards, inputs, input_count);
    if (guard_status != PYJIT_RESULT_HIT) {
        crossjit_stats.regions_executed++;
        return numpy_crossjit_record_exec(guard_status, 0, 0, 0);
    }
    return numpy_crossjit_execute_validated(opaque, inputs, input_count, result);
}

/* The Host supplies its scheduled guard plan at the original region entry.
 * No invocation state or validated object pointer is cached across callbacks. */
static PyJitResult
numpy_crossjit_run_guarded_region(PyJitGuardPlan *guards, PyJitExecPlan *plan,
                                  PyObject *const *inputs, Py_ssize_t input_count,
                                  PyJitResult *guard_status, PyObject **result)
{
    *result = NULL;
    *guard_status = numpy_crossjit_run_guards(guards, inputs, input_count);
    if (*guard_status != PYJIT_RESULT_HIT) {
        return *guard_status;
    }
    return numpy_crossjit_execute_validated(plan, inputs, input_count, result);
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
numpy_crossjit_observed_signature_dict(
    const NumpyCrossJitObservedSignature *observed)
{
    PyObject *result = PyDict_New();
    PyObject *nodes = PyTuple_New(observed->node_count);
    if (result == NULL || nodes == NULL) {
        Py_XDECREF(result);
        Py_XDECREF(nodes);
        return NULL;
    }
#define ADD_OBSERVED(name) do {                                            \
        if (numpy_crossjit_dict_u64(result, #name,                         \
                                    observed->name) < 0) {                 \
            Py_DECREF(result);                                             \
            Py_DECREF(nodes);                                              \
            return NULL;                                                   \
        }                                                                  \
    } while (0)
    ADD_OBSERVED(prepare_count);
    ADD_OBSERVED(accepted_count);
    ADD_OBSERVED(execution_count);
    ADD_OBSERVED(generated_hit_count);
    ADD_OBSERVED(generic_fallback_count);
    ADD_OBSERVED(min_elements);
    ADD_OBSERVED(max_elements);
    ADD_OBSERVED(input_count);
    ADD_OBSERVED(node_count);
    ADD_OBSERVED(dtype_class);
    ADD_OBSERVED(layout_class);
    ADD_OBSERVED(refined_layout_class);
    ADD_OBSERVED(iteration_rank_class);
    ADD_OBSERVED(iteration_descriptor_version);
    ADD_OBSERVED(outer_shape_class);
    ADD_OBSERVED(outer_count);
    ADD_OBSERVED(inner_extent);
    ADD_OBSERVED(scalar_mask);
    ADD_OBSERVED(sample_numpy_float64_scalar_mask);
    ADD_OBSERVED(sample_value_class_mask);
#undef ADD_OBSERVED
    PyObject *strides = PyTuple_New(observed->input_count);
    if (strides == NULL) {
        Py_DECREF(result);
        Py_DECREF(nodes);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < observed->input_count; i++) {
        PyObject *sample = Py_BuildValue("(LL)",
            (long long)observed->sample_inner_strides[i],
            (long long)observed->sample_outer_strides[i]);
        if (sample == NULL) {
            Py_DECREF(strides);
            Py_DECREF(result);
            Py_DECREF(nodes);
            return NULL;
        }
        PyTuple_SET_ITEM(strides, i, sample);
    }
    if (PyDict_SetItemString(result, "sample_operand_strides", strides) < 0) {
        Py_DECREF(strides);
        Py_DECREF(result);
        Py_DECREF(nodes);
        return NULL;
    }
    Py_DECREF(strides);
    PyObject *prepare_buckets = numpy_crossjit_u64_tuple(
        observed->prepare_element_buckets, NUMPY_CROSSJIT_ELEMENT_BUCKETS);
    if (prepare_buckets == NULL ||
            PyDict_SetItemString(result, "prepare_element_buckets",
                                 prepare_buckets) < 0) {
        Py_XDECREF(prepare_buckets);
        Py_DECREF(result);
        Py_DECREF(nodes);
        return NULL;
    }
    Py_DECREF(prepare_buckets);
    PyObject *buckets = numpy_crossjit_u64_tuple(
        observed->element_buckets, NUMPY_CROSSJIT_ELEMENT_BUCKETS);
    if (buckets == NULL ||
            PyDict_SetItemString(result, "element_buckets", buckets) < 0) {
        Py_XDECREF(buckets);
        Py_DECREF(result);
        Py_DECREF(nodes);
        return NULL;
    }
    Py_DECREF(buckets);
    PyObject *access_kinds = PyTuple_New(observed->input_count);
    if (access_kinds == NULL) {
        Py_DECREF(result);
        Py_DECREF(nodes);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < observed->input_count; i++) {
        PyObject *kind = PyLong_FromUnsignedLong(observed->access_kinds[i]);
        if (kind == NULL) {
            Py_DECREF(access_kinds);
            Py_DECREF(result);
            Py_DECREF(nodes);
            return NULL;
        }
        PyTuple_SET_ITEM(access_kinds, i, kind);
    }
    if (PyDict_SetItemString(result, "access_kinds", access_kinds) < 0) {
        Py_DECREF(access_kinds);
        Py_DECREF(result);
        Py_DECREF(nodes);
        return NULL;
    }
    Py_DECREF(access_kinds);
    for (Py_ssize_t i = 0; i < observed->node_count; i++) {
        const NumpyCrossJitKernelNode *node = &observed->nodes[i];
        PyObject *item = Py_BuildValue(
            "{sI,sI,sI,sI}",
            "op", (unsigned int)node->op,
            "left", (unsigned int)node->left,
            "right", (unsigned int)node->right,
            "output", (unsigned int)node->output);
        if (item == NULL) {
            Py_DECREF(result);
            Py_DECREF(nodes);
            return NULL;
        }
        PyTuple_SET_ITEM(nodes, i, item);
    }
    if (PyDict_SetItemString(result, "nodes", nodes) < 0) {
        Py_DECREF(result);
        Py_DECREF(nodes);
        return NULL;
    }
    Py_DECREF(nodes);
    return result;
}

static PyObject *
numpy_crossjit_unsupported_node_dict(
    const NumpyCrossJitUnsupportedNode *record)
{
    return Py_BuildValue(
        "{sK,sK,sK,sK,sI,sI,sI,sI,sI,sI}",
        "semantic_id", (unsigned long long)record->semantic_id,
        "occurrence_count", (unsigned long long)record->occurrence_count,
        "min_elements", (unsigned long long)record->min_elements,
        "max_elements", (unsigned long long)record->max_elements,
        "node_index", (unsigned int)record->node_index,
        "region_length", (unsigned int)record->region_length,
        "dtype_class", (unsigned int)record->dtype_class,
        "refined_layout_class", (unsigned int)record->refined_layout_class,
        "inplace", (unsigned int)record->inplace,
        "scalar_mask", (unsigned int)record->scalar_mask);
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
    PyObject *observed_signatures = NULL;
    PyObject *unsupported_nodes = NULL;
    if (result == NULL) {
        return NULL;
    }
    if (numpy_crossjit_dict_u64(result, "provider_abi_version",
                               PYJIT_PROVIDER_ABI_VERSION) < 0) {
        goto error;
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
    ADD_COUNTER(fixed_fast_plans_built);
    ADD_COUNTER(fixed_generic_plans_built);
    ADD_COUNTER(generated_registry_hit);
    ADD_COUNTER(generated_registry_miss);
    ADD_COUNTER(generated_exec_hit);
    ADD_COUNTER(generic_fallback_hit);
    ADD_COUNTER(native_exec_hit);
    ADD_COUNTER(mixed_exec_hit);
    ADD_COUNTER(native_plans_built);
    ADD_COUNTER(output_allocations);
    ADD_COUNTER(internal_allocations);
    ADD_COUNTER(internal_buffer_bytes);
    ADD_COUNTER(internal_object_allocations);
    ADD_COUNTER(packing_allocations);
    ADD_COUNTER(packing_bytes);
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
    ADD_COUNTER(observed_signature_records_dropped);
    ADD_COUNTER(unsupported_node_records_dropped);
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

    observed_signatures = PyTuple_New(
        (Py_ssize_t)crossjit_stats.observed_signature_count);
    if (observed_signatures == NULL) {
        goto error;
    }
    for (Py_ssize_t i = 0;
         i < (Py_ssize_t)crossjit_stats.observed_signature_count; i++) {
        PyObject *observed = numpy_crossjit_observed_signature_dict(
            &crossjit_stats.observed_signatures[i]);
        if (observed == NULL) {
            goto error;
        }
        PyTuple_SET_ITEM(observed_signatures, i, observed);
    }
    if (PyDict_SetItemString(result, "observed_signatures",
                             observed_signatures) < 0) {
        goto error;
    }
    Py_CLEAR(observed_signatures);

    unsupported_nodes = PyTuple_New(
        (Py_ssize_t)crossjit_stats.unsupported_node_count);
    if (unsupported_nodes == NULL) {
        goto error;
    }
    for (Py_ssize_t i = 0;
         i < (Py_ssize_t)crossjit_stats.unsupported_node_count; i++) {
        PyObject *unsupported = numpy_crossjit_unsupported_node_dict(
            &crossjit_stats.unsupported_nodes[i]);
        if (unsupported == NULL) {
            goto error;
        }
        PyTuple_SET_ITEM(unsupported_nodes, i, unsupported);
    }
    if (PyDict_SetItemString(result, "unsupported_nodes",
                             unsupported_nodes) < 0) {
        goto error;
    }
    Py_CLEAR(unsupported_nodes);

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
    PyObject *native = PyDict_New();
    if (native == NULL) goto error;
    if (native_observations != NULL) {
        Py_ssize_t position = 0;
        PyObject *key, *value;
        while (PyDict_Next(native_observations, &position, &key, &value)) {
            PyObject *row = PyDict_Copy(value);
            if (row == NULL || PyDict_SetItem(native, key, row) < 0) {
                Py_XDECREF(row);
                Py_DECREF(native);
                goto error;
            }
            Py_DECREF(row);
        }
    }
    int native_status = PyDict_SetItemString(result, "native_candidates", native);
    Py_DECREF(native);
    if (native_status < 0) goto error;
    return result;

error:
    Py_XDECREF(last);
    Py_XDECREF(candidates);
    Py_XDECREF(region_lengths);
    Py_XDECREF(layouts);
    Py_XDECREF(dtypes);
    Py_XDECREF(rejections);
    Py_XDECREF(observed_signatures);
    Py_XDECREF(unsupported_nodes);
    Py_DECREF(result);
    return NULL;
}

static void
numpy_crossjit_reset_stats_data(void)
{
    Py_CLEAR(native_observations);
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
    {"_crossjit_register_intrinsic", numpy_crossjit_register_intrinsic, METH_VARARGS, NULL},
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
        crossjit_keys[count++] = (PyJitProviderKey){
            PYJIT_KEY_BINARY, scalar_ops[i], &PyArray_Type, &PyDoubleArrType_Type, NULL};
        crossjit_keys[count++] = (PyJitProviderKey){
            PYJIT_KEY_BINARY, scalar_ops[i], &PyDoubleArrType_Type, &PyArray_Type, NULL};
    }
    for (size_t i = 0; i < sizeof(scalar_ops) / sizeof(scalar_ops[0]); i++) {
        crossjit_keys[count++] = (PyJitProviderKey){
            PYJIT_KEY_BINARY, scalar_ops[i], &PyArray_Type, &PyFloatArrType_Type, NULL};
        crossjit_keys[count++] = (PyJitProviderKey){
            PYJIT_KEY_BINARY, scalar_ops[i], &PyFloatArrType_Type, &PyArray_Type, NULL};
    }
    PyTypeObject *integer_scalars[]={&PyLongArrType_Type,&PyLongLongArrType_Type};
    for (size_t t=0;t<2;t++) {
        for (size_t i=0;i<sizeof(scalar_ops)/sizeof(scalar_ops[0]);i++) {
            crossjit_keys[count++]=(PyJitProviderKey){PYJIT_KEY_BINARY,scalar_ops[i],&PyArray_Type,integer_scalars[t],NULL};
            crossjit_keys[count++]=(PyJitProviderKey){PYJIT_KEY_BINARY,scalar_ops[i],integer_scalars[t],&PyArray_Type,NULL};
        }
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
        add_ufunc_key(module, "matmul", &crossjit_state.matmul, &count) < 0 ||
        add_ufunc_key(module, "sqrt", &crossjit_state.sqrt, &count) < 0 ||
        add_ufunc_key(module, "exp", &crossjit_state.exp, &count) < 0 ||
        add_ufunc_key(module, "tanh", &crossjit_state.tanh, &count) < 0) {
        return -1;
    }
    /* mean is a Python-level convenience in some NumPy builds; semantic
     * support remains available whenever this exact callable is exported. */
    crossjit_state.dot=PyObject_GetAttrString(module,"dot");
    if (!crossjit_state.dot) return -1;
    crossjit_keys[count++]=(PyJitProviderKey){PYJIT_KEY_CALLABLE,0,NULL,NULL,crossjit_state.dot};
    crossjit_state.mean = PyObject_GetAttrString(module, "mean");
    if (crossjit_state.mean != NULL) {
        crossjit_keys[count++] = (PyJitProviderKey){PYJIT_KEY_CALLABLE, 0, NULL, NULL, crossjit_state.mean};
    }
    else {
        PyErr_Clear();
    }
    crossjit_state.add_outer = PyObject_GetAttrString(crossjit_state.add, "outer");
    crossjit_state.maximum_reduce = PyObject_GetAttrString(crossjit_state.maximum, "reduce");
    crossjit_state.add_reduce = PyObject_GetAttrString(crossjit_state.add, "reduce");
    if (crossjit_state.add_outer == NULL || crossjit_state.maximum_reduce == NULL || crossjit_state.add_reduce == NULL) {
        return -1;
    }
    crossjit_keys[count++] = (PyJitProviderKey){PYJIT_KEY_CALLABLE, 0, NULL, NULL, crossjit_state.add_outer};
    crossjit_keys[count++] = (PyJitProviderKey){PYJIT_KEY_CALLABLE, 0, NULL, NULL, crossjit_state.maximum_reduce};
    crossjit_keys[count++] = (PyJitProviderKey){PYJIT_KEY_CALLABLE, 0, NULL, NULL, crossjit_state.add_reduce};
    assert(count<=sizeof(crossjit_keys)/sizeof(crossjit_keys[0]));
    numpy_crossjit_capture_builtins();
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
        .run_guarded_region = numpy_crossjit_run_guarded_region,
#ifdef PYJIT_SINGLE_NODE_REGION_CAPABILITY
        .supports_single_node_regions = 1,
#endif
        .destroy_contract = numpy_crossjit_destroy_contract,
        .destroy_guard_plan = numpy_crossjit_destroy_guard,
        .destroy_exec_plan = numpy_crossjit_destroy_exec,
    };
    /* Causal ablation only; production uses the combined entry by default. */
    const char *separate_guards = Py_GETENV(
        "NPY_CROSSJIT_DIAGNOSTIC_SEPARATE_GUARDS");
    if (separate_guards != NULL && strcmp(separate_guards, "1") == 0) {
        crossjit_provider.run_guarded_region = NULL;
    }
    return PyJit_RegisterProvider(&crossjit_provider);
#endif
}

#else

void numpy_crossjit_native_record_packing(size_t bytes) { (void)bytes; }

/* Keeps non-CrossJIT CPython builds source-compatible without registering. */
int
numpy_crossjit_init(PyObject *NPY_UNUSED(module))
{
    return 0;
}

#endif  /* PYJIT_PROVIDER_ABI_VERSION */
