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
#include <time.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"
#include "numpy/npy_math.h"
#include "numpy/ufuncobject.h"
#include "extobj.h"
#include "mem_overlap.h"
#include "jit_kernel.h"
#include "jit_aot_backend.h"
#include "jit_llvm_backend.h"

#ifndef PYJIT_EXT_ABI_VERSION
#define PYJIT_EXT_ABI_VERSION 7u
#define PYJIT_EXT_CORE_IR_VERSION 1u
#define PYJIT_EXT_REGION_MAX_NODES 6u
#define PYJIT_EXT_REGION_MAX_INPUTS 8u
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
typedef uint64_t PyJitExtDialectId;
typedef enum {
    PYJIT_EXT_OP_DOMAIN_OPAQUE = 0,
    PYJIT_EXT_OP_DOMAIN_ELEMENTWISE = 1,
    PYJIT_EXT_OP_DOMAIN_REDUCTION = 2,
    PYJIT_EXT_OP_DOMAIN_CONTRACTION = 3,
    PYJIT_EXT_OP_DOMAIN_VIEW = 4,
} PyJitExtOpDomain;

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
    PyJitExtDialectId dialect_id;
    uint16_t semantic_version;
    uint8_t domain;
    uint8_t output_count;
    const void *attributes;
    uint32_t attributes_size;
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
    uint16_t ir_version;
    uint16_t reserved;
    uint32_t required_features;
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
            PyObject *receiver;
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
    PyJitExtDialectId dialect_id;
    uint16_t dialect_version;
    uint16_t core_ir_version;
    uint32_t core_ir_features;
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
    void *payload, PyObject *callable, PyObject *receiver,
    PyObject *const *args,
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
    uint8_t op_domain;
    uint8_t output_count;
    uint16_t semantic_version;
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
    uint64_t region_op_signature;
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
    void *provider_data;
    void (*free_provider_data)(void *provider_data);
};

#define PYJIT_EXT_PLAN_REQUIRES_GIL          (1u << 0)
#define PYJIT_EXT_PLAN_NO_PYTHON_CALLBACK    (1u << 2)
#define PYJIT_EXT_PLAN_REGION_ANCHOR_ONLY    (1u << 7)
#define PYJIT_EXT_PLAN_ENTRY_PROFITABLE      (1u << 8)
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

#define NUMPY_JIT_MUL_ADDSUB_REGION_MIN_ELEMENTS 256
#define NUMPY_JIT_VARIANT_PROBATION_SAMPLES 4
#define NUMPY_JIT_VARIANT_DEFAULT_MIN_GAIN_PERCENT 3

typedef enum {
    NUMPY_JIT_VARIANT_PROBATION = 1,
    NUMPY_JIT_VARIANT_COMMITTED_ATOMIC,
    NUMPY_JIT_VARIANT_COMMITTED_CANDIDATE,
} NumpyJitVariantDecision;

/* Stable public dialect identity for the NumPy Region operation schema.
   It is deliberately independent of module addresses and Python objects so
   producers written in another implementation language can emit compatible
   Core IR. */
#define NUMPY_JIT_DIALECT_ID UINT64_C(0x6e756d7079000001)

static PyJitExtResult numpy_jit_prepare_binary_contig_f64(
    void *payload, PyObject *left, PyObject *right, void *run_state);

#include "jit_stencils_generated.h"

extern PyTypeObject PyUFunc_Type;
NPY_NO_EXPORT int PyUFunc_GiveFloatingpointErrors(
    const char *name, int fpe_errors);

typedef enum {
    NUMPY_JIT_OP_ADD,
    NUMPY_JIT_OP_SUBTRACT,
    NUMPY_JIT_OP_MULTIPLY,
    NUMPY_JIT_OP_TRUE_DIVIDE,
    NUMPY_JIT_OP_POWER,
    NUMPY_JIT_OP_NEGATIVE,
    NUMPY_JIT_OP_MATMUL,
    NUMPY_JIT_OP_OUTER_ADD,
    NUMPY_JIT_OP_MINIMUM,
    NUMPY_JIT_OP_LESS,
    NUMPY_JIT_OP_LESS_EQUAL,
    NUMPY_JIT_OP_LOGICAL_AND,
    NUMPY_JIT_OP_BOOLEAN_SUBSCRIPT,
    NUMPY_JIT_OP_MEAN,
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

typedef struct NumpyJitPlanPayload NumpyJitPlanPayload;
typedef struct NumpyJitDispatchEntry NumpyJitDispatchEntry;

typedef struct {
    NumpyJitPlanPayload *next;
} NumpyJitTelemetryState;

typedef struct {
    uint8_t atomic_samples;
    uint8_t candidate_samples;
    uint64_t atomic_elapsed_ns;
    uint64_t candidate_elapsed_ns;
} NumpyJitProfileState;

typedef PyJitExtResult (*NumpyJitRegionDispatchExecuteFn)(
    NumpyJitPlanPayload *payload, NumpyJitDispatchEntry *entry,
    PyObject *const *args, Py_ssize_t nargs, PyObject **result);

struct NumpyJitDispatchEntry {
    uint64_t signature;
    PyJitExtLoweredRegion *lowered;
    uint8_t negative;
    uint8_t unsupported_layout_mask;
    uint8_t size_bucket;
    uint8_t decision;
    /* The one authoritative target for this PropertyKey. */
    uint8_t selected_variant;
    NumpyJitKernelFunc selected_function;
    NumpyJitProfileState profile;
    /* Installed only after probation.  The hot path must not repeat
       candidate selection, environment parsing, or sampling. */
    NumpyJitRegionDispatchExecuteFn selected_execute;
};

typedef struct {
    uint32_t execution_count;
    uint8_t count;
    uint8_t saturated;
    uint8_t fast_entry;
    NumpyJitDispatchEntry entries[PYJIT_EXT_REGION_PIC_SIZE];
    const NumpyJitAOTEntry *precompiled_entry;
    uint64_t precompiled_lookup_ns;
    uint32_t precompiled_lookup_hit;
    uint32_t precompiled_lookup_miss;
    uint32_t precompiled_uses[3];
    uint8_t precompiled_lookup_done;
    NumpyJitLLVMEntry *llvm_entry;
    uint64_t llvm_visited_elements;
    uint64_t llvm_old_samples[3][8];
    uint64_t llvm_native_samples[3][8];
    uint32_t llvm_eligible_hits;
    uint32_t llvm_guard_misses;
    uint32_t llvm_native_uses[3];
    int16_t llvm_dtype;
    uint8_t llvm_scalar_mask;
    uint8_t llvm_property_valid;
    uint8_t llvm_decision;
    uint8_t llvm_bucket_decision[3];
    uint8_t llvm_old_count[3];
    uint8_t llvm_native_count[3];
    uint8_t llvm_cache_hit;
    uint8_t llvm_probation_old_active;
} NumpyJitDispatchTable;

typedef enum {
    NUMPY_JIT_VARIANT_ATOMIC = 1,
    NUMPY_JIT_VARIANT_DEDICATED,
    NUMPY_JIT_VARIANT_GENERIC,
    NUMPY_JIT_VARIANT_PRECOMPILED,
    NUMPY_JIT_VARIANT_LLVM,
} NumpyJitVariantKind;

/* Candidate producers fill this small value object.  Only the selector is
 * allowed to choose one as a dispatch target. */
typedef struct {
    NumpyJitVariantKind kind;
    NumpyJitKernelFunc function;
    uint8_t measure_probation;
} NumpyJitVariant;

#define NUMPY_JIT_REGION_PROGRAM_FIELDS \
    NumpyJitOp region_first_op; \
    NumpyJitOp region_second_op; \
    int region_second_consumes_right; \
    int region_runtime_other_first; \
    uint8_t region_generic; \
    uint8_t region_matmul_epilogue; \
    uint8_t region_outer_minimum; \
    uint8_t region_compare_logical; \
    uint8_t region_masked_mean; \
    uint8_t region_compiled_kind; \
    uint8_t region_runtime_miss_on_decline; \
    uint8_t region_node_count; \
    uint8_t region_input_count; \
    uint8_t region_output_slot; \
    int8_t region_mutation_input; \
    PyJitExtValueId region_input_ids[PYJIT_EXT_REGION_MAX_INPUTS]; \
    NumpyJitOp region_ops[PYJIT_EXT_REGION_MAX_NODES]; \
    uint8_t region_left[PYJIT_EXT_REGION_MAX_NODES]; \
    uint8_t region_right[PYJIT_EXT_REGION_MAX_NODES]; \
    PyJitExtPlan *region_atomic_plans[PYJIT_EXT_REGION_MAX_NODES]; \
    NumpyJitKernelRegionNode region_nodes[PYJIT_EXT_REGION_MAX_NODES]; \
    PyJitExtPlan *region_owner_plan; \
    uint64_t region_graph_hash

typedef struct {
    NUMPY_JIT_REGION_PROGRAM_FIELDS;
} NumpyJitRegionProgram;

struct NumpyJitPlanPayload {
    NumpyJitOp op;
    NumpyJitLayoutKind layout;
    int left_dtype;
    int right_dtype;
    int result_dtype;
    int itemsize;
    npy_intp min_elements;
    PyObject *expected_callable;
    PyObject *expected_receiver;
    PyCFunction expected_call_target;
    const NumpyJitCandidate *candidate;
    PyJitExtSiteState *site;
    int fused_op;
    int inplace;
    uint32_t fused_flags;
    const char *fused_pattern;
    union {
        NumpyJitRegionProgram program;
        struct {
            NUMPY_JIT_REGION_PROGRAM_FIELDS;
        };
    };
    NumpyJitDispatchTable dispatch;
    NumpyJitResultCache result_cache;
    NumpyJitResultCache *shared_result_cache;
    NumpyJitTelemetryState telemetry;
};

/* Transitional member aliases keep the evaluator readable while making the
   ownership boundary explicit: all PropertyKey decisions live in the one
   DispatchTable, not in the immutable Region description. */
#define region_execution_count dispatch.execution_count
#define region_pic_count dispatch.count
#define region_pic_saturated dispatch.saturated
#define region_fast_pic dispatch.fast_entry
#define region_pic dispatch.entries
#define region_aot_entry dispatch.precompiled_entry
#define region_aot_lookup_ns dispatch.precompiled_lookup_ns
#define region_aot_lookup_hit dispatch.precompiled_lookup_hit
#define region_aot_lookup_miss dispatch.precompiled_lookup_miss
#define region_aot_native_uses dispatch.precompiled_uses
#define region_aot_lookup_done dispatch.precompiled_lookup_done
#define region_llvm_entry dispatch.llvm_entry
#define region_llvm_visited_elements dispatch.llvm_visited_elements
#define region_llvm_old_samples dispatch.llvm_old_samples
#define region_llvm_native_samples dispatch.llvm_native_samples
#define region_llvm_eligible_hits dispatch.llvm_eligible_hits
#define region_llvm_guard_misses dispatch.llvm_guard_misses
#define region_llvm_native_uses dispatch.llvm_native_uses
#define region_llvm_dtype dispatch.llvm_dtype
#define region_llvm_scalar_mask dispatch.llvm_scalar_mask
#define region_llvm_property_valid dispatch.llvm_property_valid
#define region_llvm_decision dispatch.llvm_decision
#define region_llvm_bucket_decision dispatch.llvm_bucket_decision
#define region_llvm_old_count dispatch.llvm_old_count
#define region_llvm_native_count dispatch.llvm_native_count
#define region_llvm_cache_hit dispatch.llvm_cache_hit
#define region_llvm_probation_old_active dispatch.llvm_probation_old_active
#define stats_next telemetry.next

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
static int numpy_jit_region_layout_class(PyArrayObject *array);
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
    .dialect_id = NUMPY_JIT_DIALECT_ID,
    .dialect_version = 1,
    .core_ir_version = PYJIT_EXT_CORE_IR_VERSION,
    .core_ir_features = 0,
};

static NumpyJitOpDef numpy_jit_ops[] = {
    { NUMPY_JIT_OP_ADD, NB_ADD, NULL, 2, "add" },
    { NUMPY_JIT_OP_SUBTRACT, NB_SUBTRACT, NULL, 2, "subtract" },
    { NUMPY_JIT_OP_MULTIPLY, NB_MULTIPLY, NULL, 2, "multiply" },
    { NUMPY_JIT_OP_TRUE_DIVIDE, NB_TRUE_DIVIDE, NULL, 2, "true_divide" },
    { NUMPY_JIT_OP_POWER, NB_POWER, NULL, 2, "power" },
    { NUMPY_JIT_OP_MATMUL, NB_MATRIX_MULTIPLY, NULL, 2, "matmul" },
    { NUMPY_JIT_OP_MINIMUM, -1, NULL, 2, "minimum" },
    { NUMPY_JIT_OP_LOGICAL_AND, -1, NULL, 2, "logical_and" },
};

static PyObject *numpy_jit_mean_descriptor;

static int
numpy_jit_itemsize_for_dtype(int dtype)
{
    if (dtype == NPY_FLOAT) {
        return (int)sizeof(float);
    }
    if (dtype == NPY_DOUBLE) {
        return (int)sizeof(double);
    }
    if (dtype == NPY_LONG || dtype == NPY_LONGLONG) {
        return (int)sizeof(npy_int64);
    }
    if (dtype == NPY_INT) {
        return (int)sizeof(npy_int32);
    }
    return 0;
}

static int
numpy_jit_is_signed_integer_scalar_type(PyTypeObject *type)
{
    return type == &PyLongArrType_Type ||
           type == &PyLongLongArrType_Type;
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
    void *payload, PyObject *callable, PyObject *receiver,
    PyObject *const *args,
    Py_ssize_t nargs, PyObject **result);
static PyJitExtResult numpy_outer_add_call_eval(
    void *payload, PyObject *callable, PyObject *receiver,
    PyObject *const *args, Py_ssize_t nargs, PyObject **result);
static PyJitExtResult numpy_passthrough_binary_call_eval(
    void *payload, PyObject *callable, PyObject *receiver,
    PyObject *const *args, Py_ssize_t nargs, PyObject **result);
static PyJitExtResult numpy_compare_binary_eval(
    void *payload, PyObject *left, PyObject *right, PyObject **result);
static PyJitExtResult numpy_boolean_subscript_binary_eval(
    void *payload, PyObject *left, PyObject *right, PyObject **result);
static PyJitExtResult numpy_mean_call_eval(
    void *payload, PyObject *callable, PyObject *receiver,
    PyObject *const *args, Py_ssize_t nargs, PyObject **result);
static PyJitExtResult numpy_fused_binary2_eval(
    void *payload, PyObject *first_left, PyObject *first_right,
    PyObject *second_other, PyObject **result);
static PyJitExtResult numpy_region_eval(
    void *payload, PyObject *const *args, Py_ssize_t nargs,
    PyObject **result);
static PyJitExtResult numpy_region_fallback_eval(
    void *payload, PyObject *const *args, Py_ssize_t nargs,
    PyObject **result);
static PyJitExtResult numpy_outer_minimum_region_eval(
    NumpyJitPlanPayload *payload, PyObject *const *args,
    Py_ssize_t nargs, PyObject **result);
static PyJitExtResult numpy_compare_logical_region_eval(
    NumpyJitPlanPayload *payload, PyObject *const *args,
    Py_ssize_t nargs, PyObject **result);
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
static int numpy_jit_env_aot_mode = -1;
static int numpy_jit_env_llvm_mode = -1;
static uint64_t numpy_jit_env_llvm_min_hits = UINT64_MAX;
static uint64_t numpy_jit_env_llvm_min_visited_elements = UINT64_MAX;
static uint64_t numpy_jit_env_variant_min_gain_percent = UINT64_MAX;
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
typedef struct NumpyJitLLVMRetiredStats {
    NumpyJitPlanPayload payload;
    uint32_t site_id;
    struct NumpyJitLLVMRetiredStats *next;
} NumpyJitLLVMRetiredStats;
static NumpyJitLLVMRetiredStats *numpy_jit_llvm_retired_stats = NULL;

static int numpy_jit_stats_enabled(void);
static uint64_t numpy_jit_median8(uint64_t values[8]);

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

enum {
    NUMPY_JIT_LLVM_MODE_OFF = 0,
    NUMPY_JIT_LLVM_MODE_AUTO = 1,
    NUMPY_JIT_LLVM_MODE_FORCE = 2,
};

static int
numpy_jit_aot_mode(void)
{
    if (numpy_jit_env_aot_mode >= 0) {
        return numpy_jit_env_aot_mode;
    }
    const char *value = Py_GETENV("NUMPY_JIT_AOT");
    int mode = NUMPY_JIT_LLVM_MODE_AUTO;
    if (value != NULL && value[0] != '\0') {
        if (strcmp(value, "0") == 0 || strcmp(value, "off") == 0) {
            mode = NUMPY_JIT_LLVM_MODE_OFF;
        }
        else if (strcmp(value, "force") == 0) {
            mode = NUMPY_JIT_LLVM_MODE_FORCE;
        }
    }
    if (!numpy_jit_aot_backend_available()) {
        mode = NUMPY_JIT_LLVM_MODE_OFF;
    }
    numpy_jit_env_aot_mode = mode;
    return mode;
}

static int
numpy_jit_llvm_mode(void)
{
    if (numpy_jit_env_llvm_mode >= 0) {
        return numpy_jit_env_llvm_mode;
    }
    const char *value = Py_GETENV("NUMPY_JIT_LLVM");
    int mode = NUMPY_JIT_LLVM_MODE_AUTO;
    if (value != NULL && value[0] != '\0') {
        if (strcmp(value, "0") == 0 || strcmp(value, "off") == 0) {
            mode = NUMPY_JIT_LLVM_MODE_OFF;
        }
        else if (strcmp(value, "force") == 0) {
            mode = NUMPY_JIT_LLVM_MODE_FORCE;
        }
    }
    if (!numpy_jit_llvm_backend_available()) {
        mode = NUMPY_JIT_LLVM_MODE_OFF;
    }
    numpy_jit_env_llvm_mode = mode;
    return mode;
}

static uint64_t
numpy_jit_env_u64(const char *name, uint64_t *cache, uint64_t fallback)
{
    if (*cache != UINT64_MAX) {
        return *cache;
    }
    uint64_t result = fallback;
    const char *value = Py_GETENV(name);
    if (value != NULL && value[0] != '\0') {
        char *end = NULL;
        unsigned long long parsed = strtoull(value, &end, 10);
        if (end != value) {
            result = (uint64_t)parsed;
        }
    }
    *cache = result;
    return result;
}

static uint64_t
numpy_jit_llvm_min_hits(void)
{
    return numpy_jit_env_u64(
        "NUMPY_JIT_LLVM_MIN_HITS", &numpy_jit_env_llvm_min_hits, 32);
}

static uint64_t
numpy_jit_llvm_min_visited_elements(void)
{
    return numpy_jit_env_u64(
        "NUMPY_JIT_LLVM_MIN_VISITED_ELEMENTS",
        &numpy_jit_env_llvm_min_visited_elements, 65536);
}

static uint64_t
numpy_jit_variant_min_gain_percent(void)
{
    uint64_t gain = numpy_jit_env_u64(
        "NUMPY_JIT_VARIANT_MIN_GAIN_PERCENT",
        &numpy_jit_env_variant_min_gain_percent,
        NUMPY_JIT_VARIANT_DEFAULT_MIN_GAIN_PERCENT);
    /* A value outside this range makes the multiplication-based comparison
     * either meaningless or vulnerable to overflow for long-running sites. */
    return gain > 99 ? 99 : gain;
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
numpy_jit_record_result_cache_stats(NumpyJitPlanPayload *payload,
                                    NumpyJitResultCache *cache)
{
    if (payload == NULL || cache == NULL) {
        return;
    }
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

static void
numpy_jit_free_site_result_cache(void *provider_data)
{
    NumpyJitResultCache *cache =
        (NumpyJitResultCache *)provider_data;
    if (cache == NULL) {
        return;
    }
    Py_XDECREF(cache->result);
    PyMem_Free(cache);
}

static NumpyJitResultCache *
numpy_jit_result_cache_for_payload(NumpyJitPlanPayload *payload)
{
    if (payload->shared_result_cache != NULL) {
        return payload->shared_result_cache;
    }

    PyJitExtSiteState *site = payload->site;
    if (site == NULL || payload->region_owner_plan == NULL ||
        !(payload->region_owner_plan->flags &
          PYJIT_EXT_PLAN_ENTRY_PROFITABLE)) {
        return &payload->result_cache;
    }
    if (site->provider_data == NULL) {
        NumpyJitResultCache *cache =
            PyMem_Calloc(1, sizeof(NumpyJitResultCache));
        if (cache == NULL) {
            /* Result caching is only an optimization.  Preserve execution
               through the plan-local cache when site allocation fails. */
            return &payload->result_cache;
        }
        site->provider_data = cache;
        site->free_provider_data = numpy_jit_free_site_result_cache;
    }
    if (site->free_provider_data == numpy_jit_free_site_result_cache) {
        payload->shared_result_cache =
            (NumpyJitResultCache *)site->provider_data;
        return payload->shared_result_cache;
    }
    /* A bytecode site has one provider owner.  If a foreign provider already
       claimed this optional slot, keep the local cache instead of assuming
       its payload layout. */
    return &payload->result_cache;
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
numpy_jit_result_cache_maybe_disable(NumpyJitResultCache *cache)
{
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
    if (payload == NULL || !numpy_jit_result_cache_enabled() || size < 0) {
        return (PyArrayObject *)PyArray_EMPTY(ndim, dims, dtype, 0);
    }

    NumpyJitResultCache *cache =
        numpy_jit_result_cache_for_payload(payload);
    if (cache->disabled) {
        return (PyArrayObject *)PyArray_EMPTY(ndim, dims, dtype, 0);
    }

    int itemsize = numpy_jit_itemsize_for_dtype(dtype);
    if (itemsize <= 0) {
        itemsize = payload->itemsize;
    }
    if (itemsize <= 0 ||
        (uint64_t)size > UINT64_MAX / (uint64_t)itemsize) {
        return (PyArrayObject *)PyArray_EMPTY(ndim, dims, dtype, 0);
    }
    uint64_t bytes = (uint64_t)size * (uint64_t)itemsize;
    if (bytes > numpy_jit_result_cache_max_bytes()) {
        return (PyArrayObject *)PyArray_EMPTY(ndim, dims, dtype, 0);
    }

    if (cache->result != NULL) {
        if (Py_REFCNT(cache->result) != 1) {
            cache->miss++;
            cache->refcnt_miss++;
            Py_CLEAR(cache->result);
            cache->bytes = 0;
            numpy_jit_result_cache_maybe_disable(cache);
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
            numpy_jit_result_cache_maybe_disable(cache);
            if (cache->disabled) {
                return (PyArrayObject *)PyArray_EMPTY(ndim, dims, dtype, 0);
            }
        }
    }
    else {
        cache->miss++;
        numpy_jit_result_cache_maybe_disable(cache);
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

static PyArrayObject *
numpy_jit_get_or_alloc_workspace(NumpyJitPlanPayload *payload, int ndim,
                                 npy_intp *dims, int dtype,
                                 npy_intp size)
{
    int itemsize = numpy_jit_itemsize_for_dtype(dtype);
    if (payload == NULL || itemsize <= 0 || size < 0 ||
        (uint64_t)size > UINT64_MAX / (uint64_t)itemsize) {
        return NULL;
    }
    uint64_t bytes = (uint64_t)size * (uint64_t)itemsize;
    /* Workspaces are never exposed to Python and remove a full temporary
       allocation from contraction prologues.  Bound them independently of
       returned-result caches, whose much smaller default limits retained
       user-visible buffers at ordinary elementwise sites. */
    if (bytes > 64ull * 1024ull * 1024ull) {
        return NULL;
    }
    NumpyJitResultCache *cache = &payload->result_cache;
    if (cache->result != NULL &&
        !numpy_jit_result_cache_shape_matches(
            cache->result, ndim, dims, dtype)) {
        cache->shape_miss++;
        Py_CLEAR(cache->result);
        cache->bytes = 0;
    }
    if (cache->result == NULL) {
        cache->miss++;
        cache->result = (PyArrayObject *)PyArray_EMPTY(
            ndim, dims, dtype, 0);
        if (cache->result == NULL) {
            return NULL;
        }
        cache->bytes = bytes;
    }
    else {
        cache->hit++;
    }
    return cache->result;
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
numpy_jit_retire_llvm_stats(NumpyJitPlanPayload *payload)
{
    if (!numpy_jit_stats_enabled() || payload == NULL ||
        !payload->region_generic ||
        (payload->region_llvm_eligible_hits == 0 &&
         payload->region_llvm_entry == NULL &&
         !payload->region_aot_lookup_done)) {
        return;
    }
    NumpyJitLLVMRetiredStats *retired = malloc(sizeof(*retired));
    if (retired == NULL) {
        return;
    }
    memcpy(&retired->payload, payload, sizeof(retired->payload));
    retired->site_id = payload->site != NULL ?
        payload->site->site_id : UINT32_MAX;
    retired->next = numpy_jit_llvm_retired_stats;
    numpy_jit_llvm_retired_stats = retired;
}

static void
numpy_jit_dump_aot_row(FILE *fp, NumpyJitPlanPayload *payload,
                       uint32_t site_id)
{
    if (!payload->region_aot_lookup_done) {
        return;
    }
    NumpyJitAOTEntryInfo info = {0};
    numpy_jit_aot_entry_info(payload->region_aot_entry, &info);
    fprintf(fp,
        "{\"site_id\":%" PRIu32
        ",\"graph_hash\":%" PRIu64
        ",\"backend\":\"aot-stencil\",\"lookup_hit\":%" PRIu32
        ",\"lookup_miss\":%" PRIu32
        ",\"lookup_ns\":%" PRIu64
        ",\"name\":\"%s\",\"accepted_bucket_mask\":%u"
        ",\"vector_bits\":%u,\"code_size\":%u"
        ",\"native_uses\":[%" PRIu32 ",%" PRIu32 ",%" PRIu32 "]}\n",
        site_id, payload->region_graph_hash,
        payload->region_aot_lookup_hit, payload->region_aot_lookup_miss,
        payload->region_aot_lookup_ns, info.name,
        (unsigned)info.accepted_bucket_mask, (unsigned)info.vector_bits,
        (unsigned)info.code_size, payload->region_aot_native_uses[0],
        payload->region_aot_native_uses[1], payload->region_aot_native_uses[2]);
}

static void
numpy_jit_dump_llvm_row(FILE *fp, NumpyJitPlanPayload *payload,
                        uint32_t site_id)
{
    NumpyJitLLVMEntryInfo info = {0};
    numpy_jit_llvm_entry_info(payload->region_llvm_entry, &info);
    for (unsigned bucket = 0; bucket < 3; bucket++) {
        if (bucket != 0 && payload->region_llvm_old_count[bucket] == 0 &&
            payload->region_llvm_native_count[bucket] == 0 &&
            payload->region_llvm_native_uses[bucket] == 0) {
            continue;
        }
        uint64_t old_median = payload->region_llvm_old_count[bucket] == 8 ?
            numpy_jit_median8(payload->region_llvm_old_samples[bucket]) : 0;
        uint64_t native_median =
            payload->region_llvm_native_count[bucket] == 8 ?
            numpy_jit_median8(payload->region_llvm_native_samples[bucket]) : 0;
        uint8_t bucket_decision =
            payload->region_llvm_bucket_decision[bucket];
        const char *decision = payload->region_llvm_decision == 3 ?
            "rejected-hard" : bucket_decision == 2 ?
            "accepted" : bucket_decision == 3 ?
            "rejected" : bucket_decision == 1 ?
            "probation" : "warming";
        fprintf(fp,
            "{\"site_id\":%" PRIu32
            ",\"graph_hash\":%" PRIu64
            ",\"backend\":\"%s\",\"dtype\":%d,"
            "\"scalar_mask\":%u,\"size_bucket\":%u,"
            "\"eligible_hits\":%" PRIu32
            ",\"guard_misses\":%" PRIu32
            ",\"visited_elements\":%" PRIu64
            ",\"entry_state\":%d,\"cache_hit\":%u"
            ",\"queue_ns\":%" PRIu64
            ",\"compile_ns\":%" PRIu64
            ",\"optimization_ns\":%" PRIu64
            ",\"codegen_ns\":%" PRIu64
            ",\"object_bytes\":%" PRIu64
            ",\"cache_uses\":%" PRIu64
            ",\"vector_bits\":%u,\"decision\":\"%s\","
            "\"old_samples\":%u,\"native_samples\":%u,"
            "\"old_median_ns_per_element_x1024\":%" PRIu64
            ",\"native_median_ns_per_element_x1024\":%" PRIu64
            ",\"native_uses\":%" PRIu32 "}\n",
            site_id, payload->region_graph_hash,
            numpy_jit_llvm_backend_name(),
            (int)payload->region_llvm_dtype,
            (unsigned)payload->region_llvm_scalar_mask,
            bucket, payload->region_llvm_eligible_hits,
            payload->region_llvm_guard_misses,
            payload->region_llvm_visited_elements,
            info.state, (unsigned)payload->region_llvm_cache_hit,
            info.queue_ns, info.compile_ns, info.optimization_ns,
            info.codegen_ns, info.object_bytes, info.cache_uses,
            info.vector_bits, decision,
            (unsigned)payload->region_llvm_old_count[bucket],
            (unsigned)payload->region_llvm_native_count[bucket],
            old_median, native_median,
            payload->region_llvm_native_uses[bucket]);
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
            NumpyJitResultCache *cache = payload->shared_result_cache != NULL ?
                payload->shared_result_cache : &payload->result_cache;
            if (payload->shared_result_cache != NULL) {
                int already_dumped = 0;
                for (NumpyJitPlanPayload *prior = numpy_jit_plan_stats_list;
                     prior != payload; prior = prior->stats_next) {
                    if (prior->shared_result_cache == cache) {
                        already_dumped = 1;
                        break;
                    }
                }
                if (already_dumped) {
                    continue;
                }
            }
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

    snprintf(path, sizeof(path), "%s/%s", dir,
             "crossjit_numpy_llvm.ndjson");
    fp = fopen(path, "w");
    if (fp != NULL) {
        for (NumpyJitPlanPayload *payload = numpy_jit_plan_stats_list;
             payload != NULL; payload = payload->stats_next) {
            if (!payload->region_generic ||
                (payload->region_llvm_eligible_hits == 0 &&
                 payload->region_llvm_entry == NULL)) {
                continue;
            }
            numpy_jit_dump_llvm_row(
                fp, payload,
                payload->site != NULL ?
                    payload->site->site_id : UINT32_MAX);
        }
        for (NumpyJitLLVMRetiredStats *retired =
                 numpy_jit_llvm_retired_stats;
             retired != NULL; retired = retired->next) {
            numpy_jit_dump_llvm_row(
                fp, &retired->payload, retired->site_id);
        }
        fclose(fp);
    }

    snprintf(path, sizeof(path), "%s/%s", dir,
             "crossjit_numpy_aot.ndjson");
    fp = fopen(path, "w");
    if (fp != NULL) {
        for (NumpyJitPlanPayload *payload = numpy_jit_plan_stats_list;
             payload != NULL; payload = payload->stats_next) {
            numpy_jit_dump_aot_row(
                fp, payload, payload->site != NULL ?
                payload->site->site_id : UINT32_MAX);
        }
        for (NumpyJitLLVMRetiredStats *retired =
                 numpy_jit_llvm_retired_stats;
             retired != NULL; retired = retired->next) {
            numpy_jit_dump_aot_row(fp, &retired->payload, retired->site_id);
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
#if PYJIT_EXT_ABI_VERSION >= 6
    /* ABI v6 uses the Core IR-aware generic external-op emitter.  The v5
       hand-authored r23 blob remains available to old runtimes, but must not
       be mixed with the enlarged plan/provider contracts. */
    return NULL;
#else
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
#endif
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
        Py_XDECREF(payload->expected_receiver);
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
    plan->op_domain = candidate->op == NUMPY_JIT_OP_MATMUL ?
        PYJIT_EXT_OP_DOMAIN_CONTRACTION :
        PYJIT_EXT_OP_DOMAIN_ELEMENTWISE;
    plan->output_count = 1;
    plan->semantic_version = 1;
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
    if (request->u.binary.binary_oparg == NB_SUBSCR &&
        (request->u.binary.left_type == NULL ||
         request->u.binary.left_type == &PyArray_Type) &&
        (request->u.binary.right_type == NULL ||
         request->u.binary.right_type == &PyArray_Type)) {
        NumpyJitPlanPayload *payload = PyMem_Calloc(1, sizeof(*payload));
        PyJitExtPlan *plan = PyMem_Calloc(1, sizeof(*plan));
        if (payload == NULL || plan == NULL) {
            PyMem_Free(payload);
            PyMem_Free(plan);
            PyErr_NoMemory();
            return NULL;
        }
        payload->op = NUMPY_JIT_OP_BOOLEAN_SUBSCRIPT;
        payload->site = site;
        numpy_jit_link_plan_stats(payload);
        plan->abi_version = PYJIT_EXT_ABI_VERSION;
        plan->struct_size = sizeof(*plan);
        plan->kind = PYJIT_EXT_PLAN_BINARY;
        plan->provider = &numpy_jit_provider;
        plan->flags = PYJIT_EXT_PLAN_REQUIRES_GIL |
                      PYJIT_EXT_PLAN_NO_PYTHON_CALLBACK |
                      PYJIT_EXT_PLAN_ALLOCATES_RESULT |
                      PYJIT_EXT_PLAN_REGION_ANCHOR_ONLY;
        /* Anchor restoration uses op_id as the original NB_* argument. */
        plan->op_id = NB_SUBSCR;
        plan->payload = payload;
        plan->eval.binary_eval = numpy_boolean_subscript_binary_eval;
        plan->effect_flags = PYJIT_EXT_EFFECT_PURE |
                             PYJIT_EXT_EFFECT_NO_PYTHON_CALLBACK;
        plan->mutation_input = -1;
        plan->region_input_count = 2;
        plan->op_domain = PYJIT_EXT_OP_DOMAIN_VIEW;
        plan->output_count = 1;
        plan->semantic_version = 1;
        plan->debug_name = "numpy.boolean_subscript.region_anchor";
        return plan;
    }
    int richcompare = request->u.binary.binary_oparg -
                      PYJIT_EXT_BINARY_RICHCOMPARE_BASE;
    if (richcompare == Py_LT || richcompare == Py_LE) {
        NumpyJitPlanPayload *payload = PyMem_Calloc(1, sizeof(*payload));
        PyJitExtPlan *plan = PyMem_Calloc(1, sizeof(*plan));
        if (payload == NULL || plan == NULL) {
            PyMem_Free(payload);
            PyMem_Free(plan);
            PyErr_NoMemory();
            return NULL;
        }
        payload->op = richcompare == Py_LT ?
            NUMPY_JIT_OP_LESS : NUMPY_JIT_OP_LESS_EQUAL;
        payload->site = site;
        numpy_jit_link_plan_stats(payload);
        plan->abi_version = PYJIT_EXT_ABI_VERSION;
        plan->struct_size = sizeof(*plan);
        plan->kind = PYJIT_EXT_PLAN_BINARY;
        plan->provider = &numpy_jit_provider;
        plan->flags = PYJIT_EXT_PLAN_REQUIRES_GIL |
                      PYJIT_EXT_PLAN_NO_PYTHON_CALLBACK |
                      PYJIT_EXT_PLAN_ALLOCATES_RESULT |
                      PYJIT_EXT_PLAN_REGION_ANCHOR_ONLY;
        /* Keep the original COMPARE_OP encoding for anchor restoration;
           the payload remains the semantic Core IR operation. */
        plan->op_id = (uint32_t)(richcompare << 5);
        plan->payload = payload;
        plan->eval.binary_eval = numpy_compare_binary_eval;
        plan->effect_flags = PYJIT_EXT_EFFECT_PURE |
                             PYJIT_EXT_EFFECT_NO_PYTHON_CALLBACK;
        plan->mutation_input = -1;
        plan->region_input_count = 2;
        plan->op_domain = PYJIT_EXT_OP_DOMAIN_ELEMENTWISE;
        plan->output_count = 1;
        plan->semantic_version = 1;
        plan->debug_name = "numpy.richcompare.region_anchor";
        return plan;
    }
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
              request->u.binary.right_type == &PyDoubleArrType_Type ||
              numpy_jit_is_signed_integer_scalar_type(
                  request->u.binary.right_type))) {
        if (!numpy_jit_array_scalar_enabled()) {
            return NULL;
        }
        layout = NUMPY_JIT_LAYOUT_C_CONTIG_ARRAY_SCALAR;
    }
    else if ((request->u.binary.left_type == &PyFloat_Type ||
              request->u.binary.left_type == &PyLong_Type ||
              request->u.binary.left_type == &PyFloatArrType_Type ||
              request->u.binary.left_type == &PyDoubleArrType_Type ||
              numpy_jit_is_signed_integer_scalar_type(
                  request->u.binary.left_type)) &&
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
    plan->op_domain = PYJIT_EXT_OP_DOMAIN_ELEMENTWISE;
    plan->output_count = 1;
    plan->semantic_version = 1;
    plan->debug_name = "numpy.negative.region_anchor";
    return plan;
}

static PyJitExtPlan *
numpy_jit_try_call_plan(const PyJitExtTraceRequest *request,
                        PyJitExtSiteState *site)
{
    if (request->u.call.nargs == 0 &&
        !(request->u.call.call_flags & PYJIT_CALL_HAS_KWARGS) &&
        request->u.call.callable == numpy_jit_mean_descriptor) {
        NumpyJitPlanPayload *payload = PyMem_Calloc(1, sizeof(*payload));
        PyJitExtPlan *plan = PyMem_Calloc(1, sizeof(*plan));
        if (payload == NULL || plan == NULL) {
            PyMem_Free(payload);
            PyMem_Free(plan);
            PyErr_NoMemory();
            return NULL;
        }
        payload->op = NUMPY_JIT_OP_MEAN;
        payload->expected_callable = request->u.call.callable;
        Py_INCREF(payload->expected_callable);
        payload->site = site;
        numpy_jit_link_plan_stats(payload);
        plan->abi_version = PYJIT_EXT_ABI_VERSION;
        plan->struct_size = sizeof(*plan);
        plan->kind = PYJIT_EXT_PLAN_CALL;
        plan->provider = &numpy_jit_provider;
        plan->flags = PYJIT_EXT_PLAN_REQUIRES_GIL |
                      PYJIT_EXT_PLAN_NO_PYTHON_CALLBACK |
                      PYJIT_EXT_PLAN_ALLOCATES_RESULT |
                      PYJIT_EXT_PLAN_REGION_ANCHOR_ONLY;
        plan->op_id = NUMPY_JIT_OP_MEAN;
        plan->payload = payload;
        plan->eval.call_eval = numpy_mean_call_eval;
        plan->effect_flags = PYJIT_EXT_EFFECT_PURE |
                             PYJIT_EXT_EFFECT_NO_PYTHON_CALLBACK;
        plan->mutation_input = -1;
        /* For a method call the receiver is the semantic Region operand. */
        plan->region_input_count = 1;
        plan->op_domain = PYJIT_EXT_OP_DOMAIN_REDUCTION;
        plan->output_count = 1;
        plan->semantic_version = 1;
        plan->debug_name = "numpy.ndarray.mean.region_anchor";
        return plan;
    }
    if (request->u.call.nargs != 2 ||
        (request->u.call.call_flags & PYJIT_CALL_HAS_KWARGS)) {
        return NULL;
    }

    PyObject *resolved_callable = request->u.call.callable;
    PyObject *owned_resolved_callable = NULL;
    if (resolved_callable == NULL) {
        owned_resolved_callable = PyObject_GetAttrString(
            (PyObject *)&PyUFunc_Type, "outer");
        resolved_callable = owned_resolved_callable;
    }
    if (resolved_callable == NULL) {
        return NULL;
    }

    if (PyCFunction_Check(resolved_callable) ||
        request->u.call.receiver ==
            numpy_jit_known_ufunc(NUMPY_JIT_OP_ADD)) {
        PyObject *name = PyObject_GetAttrString(
            resolved_callable, "__name__");
        int is_outer = name != NULL && PyUnicode_Check(name) &&
            PyUnicode_CompareWithASCIIString(name, "outer") == 0;
        Py_XDECREF(name);
        if (!is_outer) {
            Py_XDECREF(owned_resolved_callable);
            PyErr_Clear();
            return NULL;
        }
        NumpyJitPlanPayload *payload =
            PyMem_Calloc(1, sizeof(*payload));
        PyJitExtPlan *plan = PyMem_Calloc(1, sizeof(*plan));
        if (payload == NULL || plan == NULL) {
            PyMem_Free(payload);
            PyMem_Free(plan);
            Py_XDECREF(owned_resolved_callable);
            PyErr_NoMemory();
            return NULL;
        }
        payload->op = NUMPY_JIT_OP_OUTER_ADD;
        payload->expected_callable = resolved_callable;
        payload->expected_receiver =
            numpy_jit_known_ufunc(NUMPY_JIT_OP_ADD);
        Py_INCREF(payload->expected_callable);
        Py_INCREF(payload->expected_receiver);
        if (PyCFunction_Check(resolved_callable)) {
            payload->expected_call_target =
                PyCFunction_GET_FUNCTION(resolved_callable);
        }
        payload->site = site;
        numpy_jit_link_plan_stats(payload);
        plan->abi_version = PYJIT_EXT_ABI_VERSION;
        plan->struct_size = sizeof(*plan);
        plan->kind = PYJIT_EXT_PLAN_CALL;
        plan->provider = &numpy_jit_provider;
        plan->flags = PYJIT_EXT_PLAN_REQUIRES_GIL |
                      PYJIT_EXT_PLAN_NO_PYTHON_CALLBACK |
                      PYJIT_EXT_PLAN_ALLOCATES_RESULT |
                      PYJIT_EXT_PLAN_REGION_ANCHOR_ONLY;
        plan->op_id = NUMPY_JIT_OP_OUTER_ADD;
        plan->payload = payload;
        plan->eval.call_eval = numpy_outer_add_call_eval;
        plan->effect_flags = PYJIT_EXT_EFFECT_PURE |
                             PYJIT_EXT_EFFECT_NO_PYTHON_CALLBACK;
        plan->mutation_input = -1;
        plan->region_input_count = 2;
        plan->op_domain = PYJIT_EXT_OP_DOMAIN_ELEMENTWISE;
        plan->output_count = 1;
        plan->semantic_version = 1;
        plan->debug_name = "numpy.add.outer.region_anchor";
        Py_XDECREF(owned_resolved_callable);
        return plan;
    }

    Py_XDECREF(owned_resolved_callable);

    NumpyJitOp op;
    if (!resolve_known_ufunc(request->u.call.callable, &op)) {
        return NULL;
    }
    if (op == NUMPY_JIT_OP_MINIMUM || op == NUMPY_JIT_OP_LOGICAL_AND) {
        NumpyJitPlanPayload *payload = PyMem_Calloc(1, sizeof(*payload));
        PyJitExtPlan *plan = PyMem_Calloc(1, sizeof(*plan));
        if (payload == NULL || plan == NULL) {
            PyMem_Free(payload);
            PyMem_Free(plan);
            PyErr_NoMemory();
            return NULL;
        }
        payload->op = op;
        payload->expected_callable = request->u.call.callable;
        Py_INCREF(payload->expected_callable);
        payload->site = site;
        numpy_jit_link_plan_stats(payload);
        plan->abi_version = PYJIT_EXT_ABI_VERSION;
        plan->struct_size = sizeof(*plan);
        plan->kind = PYJIT_EXT_PLAN_CALL;
        plan->provider = &numpy_jit_provider;
        plan->flags = PYJIT_EXT_PLAN_REQUIRES_GIL |
                      PYJIT_EXT_PLAN_NO_PYTHON_CALLBACK |
                      PYJIT_EXT_PLAN_ALLOCATES_RESULT |
                      PYJIT_EXT_PLAN_REGION_ANCHOR_ONLY;
        plan->op_id = op;
        plan->payload = payload;
        plan->eval.call_eval = numpy_passthrough_binary_call_eval;
        plan->effect_flags = PYJIT_EXT_EFFECT_PURE |
                             PYJIT_EXT_EFFECT_NO_PYTHON_CALLBACK;
        plan->mutation_input = -1;
        plan->region_input_count = 2;
        plan->op_domain = PYJIT_EXT_OP_DOMAIN_ELEMENTWISE;
        plan->output_count = 1;
        plan->semantic_version = 1;
        plan->debug_name = op == NUMPY_JIT_OP_MINIMUM ?
            "numpy.minimum.region_anchor" :
            "numpy.logical_and.region_anchor";
        return plan;
    }
    const NumpyJitCandidate *candidate = find_candidate(
        op, NPY_DOUBLE, NPY_DOUBLE, NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE);
    if (candidate == NULL) {
        return NULL;
    }
    return make_numpy_plan(candidate, request->u.call.callable, site);
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

/* Lower ``(array * scalar) @ other`` (and the reflected scalar multiply)
   while preserving NumPy's exact operation order.  The multiply ufunc still
   performs the original elementwise computation, but writes into a bounded
   provider workspace reused by the Region plan; matmul then consumes that
   workspace directly. */
static PyJitExtPlan *
numpy_jit_build_matmul_prologue_region(
        const PyJitExtRegionRequest *request, PyJitExtSiteState *site)
{
    /* Scaling one matmul operand into a provider workspace is semantically
     * safe, but it is not consistently profitable: Gemver S regresses while
     * M gains materially.  Keep the experimental lowering available for
     * targeted study, but require an explicit opt-in until Core IR can carry
     * a pre-admission size property rather than installing a Region that
     * later has to fall back. */
    const char *enabled = getenv("NUMPY_JIT_ENABLE_MATMUL_PROLOGUE_REGION");
    if (enabled == NULL || enabled[0] != '1') {
        return NULL;
    }
    if (request->node_count != 2 || request->input_count != 3) {
        return NULL;
    }
    PyJitExtPlan *multiply_plan =
        (PyJitExtPlan *)request->nodes[0].atomic_plan;
    PyJitExtPlan *matmul_plan =
        (PyJitExtPlan *)request->nodes[1].atomic_plan;
    if (multiply_plan == NULL || matmul_plan == NULL ||
        multiply_plan->provider != &numpy_jit_provider ||
        matmul_plan->provider != &numpy_jit_provider ||
        multiply_plan->payload == NULL || matmul_plan->payload == NULL ||
        ((NumpyJitPlanPayload *)multiply_plan->payload)->op !=
            NUMPY_JIT_OP_MULTIPLY ||
        ((NumpyJitPlanPayload *)matmul_plan->payload)->op !=
            NUMPY_JIT_OP_MATMUL ||
        request->nodes[0].input_count != 2 ||
        request->nodes[1].input_count != 2) {
        return NULL;
    }
    int consumes_left =
        request->nodes[1].inputs[0] == request->nodes[0].output;
    int consumes_right =
        request->nodes[1].inputs[1] == request->nodes[0].output;
    if (consumes_left == consumes_right) {
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
    payload->region_matmul_epilogue = 3;
    payload->region_runtime_miss_on_decline = 1;
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
    payload->region_ops[0] = NUMPY_JIT_OP_MULTIPLY;
    payload->region_ops[1] = NUMPY_JIT_OP_MATMUL;
    payload->region_owner_plan = plan;
    numpy_jit_link_plan_stats(payload);

    plan->abi_version = PYJIT_EXT_ABI_VERSION;
    plan->struct_size = sizeof(*plan);
    plan->kind = PYJIT_EXT_PLAN_REGION;
    plan->provider = &numpy_jit_provider;
    plan->flags = PYJIT_EXT_PLAN_REQUIRES_GIL |
                  PYJIT_EXT_PLAN_NO_PYTHON_CALLBACK |
                  PYJIT_EXT_PLAN_ALLOCATES_RESULT |
                  PYJIT_EXT_PLAN_ENTRY_PROFITABLE;
    plan->op_id = 4200u;
    plan->payload = payload;
    plan->eval.region_eval = numpy_region_eval;
    plan->region_fallback_eval = numpy_region_fallback_eval;
    plan->effect_flags = PYJIT_EXT_EFFECT_PURE |
                         PYJIT_EXT_EFFECT_NO_PYTHON_CALLBACK |
                         PYJIT_EXT_EFFECT_PRESERVES_FP_POLICY |
                         PYJIT_EXT_EFFECT_OWNS_ATOMIC_PLANS;
    plan->mutation_input = -1;
    plan->region_input_count = 3;
    plan->op_domain = PYJIT_EXT_OP_DOMAIN_CONTRACTION;
    plan->output_count = 1;
    plan->semantic_version = 1;
    plan->debug_name = "numpy.region.matmul_prologue";
    return plan;
}

/* Lower ``matmul(a, b) + (array * scalar)`` without materializing the
   scaled-array and add temporaries.  Matmul still observes its original
   operands and evaluation order; only the independent elementwise branch is
   scheduled into the fresh matmul output. */
static PyJitExtPlan *
numpy_jit_build_matmul_scaled_add_region(
        const PyJitExtRegionRequest *request, PyJitExtSiteState *site)
{
    if (request->node_count != 3 || request->input_count != 4) {
        return NULL;
    }
    PyJitExtPlan *atomic[3] = {
        (PyJitExtPlan *)request->nodes[0].atomic_plan,
        (PyJitExtPlan *)request->nodes[1].atomic_plan,
        (PyJitExtPlan *)request->nodes[2].atomic_plan,
    };
    for (int i = 0; i < 3; i++) {
        if (atomic[i] == NULL || atomic[i]->provider != &numpy_jit_provider ||
            atomic[i]->kind != PYJIT_EXT_PLAN_BINARY ||
            atomic[i]->payload == NULL) {
            return NULL;
        }
    }
    NumpyJitPlanPayload *p0 = (NumpyJitPlanPayload *)atomic[0]->payload;
    NumpyJitPlanPayload *p1 = (NumpyJitPlanPayload *)atomic[1]->payload;
    NumpyJitPlanPayload *p2 = (NumpyJitPlanPayload *)atomic[2]->payload;
    const PyJitExtRegionNode *matmul = &request->nodes[0];
    const PyJitExtRegionNode *multiply = &request->nodes[1];
    const PyJitExtRegionNode *add = &request->nodes[2];
    int add_joins =
        ((add->inputs[0] == matmul->output &&
          add->inputs[1] == multiply->output) ||
         (add->inputs[1] == matmul->output &&
          add->inputs[0] == multiply->output));
    if (p0->op != NUMPY_JIT_OP_MATMUL ||
        p1->op != NUMPY_JIT_OP_MULTIPLY ||
        p2->op != NUMPY_JIT_OP_ADD || !add_joins ||
        matmul->input_count != 2 || multiply->input_count != 2 ||
        add->input_count != 2) {
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
    payload->region_matmul_epilogue = 2;
    payload->region_runtime_miss_on_decline = 1;
    payload->region_node_count = 3;
    payload->region_input_count = 4;
    payload->region_output_slot = 6;
    payload->region_mutation_input = -1;
    payload->min_elements = 1;
    payload->site = site;
    for (int i = 0; i < request->input_count; i++) {
        payload->region_input_ids[i] = request->inputs[i].value_id;
    }
    for (int i = 0; i < 3; i++) {
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
        payload->region_atomic_plans[i] = atomic[i];
    }
    payload->region_ops[0] = NUMPY_JIT_OP_MATMUL;
    payload->region_ops[1] = NUMPY_JIT_OP_MULTIPLY;
    payload->region_ops[2] = NUMPY_JIT_OP_ADD;
    payload->region_owner_plan = plan;
    numpy_jit_link_plan_stats(payload);

    plan->abi_version = PYJIT_EXT_ABI_VERSION;
    plan->struct_size = sizeof(*plan);
    plan->kind = PYJIT_EXT_PLAN_REGION;
    plan->provider = &numpy_jit_provider;
    plan->flags = PYJIT_EXT_PLAN_REQUIRES_GIL |
                  PYJIT_EXT_PLAN_NO_PYTHON_CALLBACK |
                  PYJIT_EXT_PLAN_ALLOCATES_RESULT |
                  PYJIT_EXT_PLAN_ENTRY_PROFITABLE;
    plan->op_id = 4100u;
    plan->payload = payload;
    plan->eval.region_eval = numpy_region_eval;
    plan->region_fallback_eval = numpy_region_fallback_eval;
    plan->effect_flags = PYJIT_EXT_EFFECT_PURE |
                         PYJIT_EXT_EFFECT_NO_PYTHON_CALLBACK |
                         PYJIT_EXT_EFFECT_PRESERVES_FP_POLICY |
                         PYJIT_EXT_EFFECT_OWNS_ATOMIC_PLANS;
    plan->mutation_input = -1;
    plan->region_input_count = 4;
    plan->op_domain = PYJIT_EXT_OP_DOMAIN_CONTRACTION;
    plan->output_count = 1;
    plan->semantic_version = 1;
    plan->debug_name = "numpy.region.matmul_scaled_add";
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
    payload->region_graph_hash = request->graph_hash;
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
    int all_add = 1;
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
        if (atomic_payload->op != NUMPY_JIT_OP_ADD) {
            all_add = 0;
        }
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
    /* Keep short in-place updates on the ordinary NumPy path: their
     * allocation/aliasing cost dominates the saved dispatch, and wrapping
     * them in a PIC adds two guards on every iteration. */
    if (request->node_count == 2 && request->input_count == 3) {
        PyJitExtPlan *last = (PyJitExtPlan *)request->nodes[1].atomic_plan;
        if (last != NULL &&
            (last->effect_flags & PYJIT_EXT_EFFECT_MUTATES_INPUT)) {
            return NULL;
        }
        /* A binary followed by an in-place binary is encoded by the Core IR
         * producer with the stable ``100 + op`` semantic id.  Reject the
         * corresponding short multiply/update form as well; both forms are
         * update chains where the external guard cost exceeds the dispatch
         * saved by the stencil. */
        PyJitExtPlan *first = (PyJitExtPlan *)request->nodes[0].atomic_plan;
        if (first != NULL && last != NULL &&
            ((first->op_id == 0 && last->op_id == 100) ||
             (first->op_id == 2 && last->op_id == 4))) {
            return NULL;
        }
    }

    /* In-place two-node DAGs are frequently just a short-lived temporary
     * update (for example ``out += a + b``).  The generic region wrapper
     * adds PIC/guard bookkeeping but cannot eliminate the surrounding
     * allocation and aliasing work, so keep the existing NumPy path for this
     * shape.  Non-mutating two-node regions (the common scalar/vector
     * affine kernels) remain eligible for fusion. */
    if (payload->inplace && request->node_count == 2 &&
        request->input_count == 3) {
        PyMem_Free(payload);
        return NULL;
    }

    /* The all-array add chain is one of the most common Region DAGs.  Use
       the existing add3 traversal as a normal schedule (rather than
       treating it as a separate fused architecture path). */
    if (!payload->inplace && all_add && payload->region_node_count == 2 &&
        payload->region_input_count == 3) {
        int t0 = payload->region_input_count;
        NumpyJitKernelRegionNode *tail = &payload->region_nodes[1];
        int consumes_internal = tail->left == t0 || tail->right == t0;
        if (consumes_internal) {
            payload->region_compiled_kind = 7;
        }
    }

    /* High-frequency elementwise multiply followed by add/subtract. */
    if (!payload->inplace && payload->region_node_count == 2 &&
        payload->region_input_count == 3 &&
        payload->region_nodes[0].op == NUMPY_JIT_KERNEL_REGION_MUL &&
        (payload->region_nodes[1].op == NUMPY_JIT_KERNEL_REGION_ADD ||
         payload->region_nodes[1].op == NUMPY_JIT_KERNEL_REGION_SUB) &&
        payload->region_nodes[1].left == payload->region_input_count) {
        payload->region_compiled_kind =
            payload->region_nodes[1].op == NUMPY_JIT_KERNEL_REGION_ADD ? 8 : 9;
    }

    /* Compile the common affine DAG

           t0 = a * alpha
           t1 = b * beta
           t2 = t0 + t1
           out = t2 + gamma

       to a schedule-specific vectorizable loop.  Scalar/array roles remain
       dynamic guards in the evaluator; this is structural lowering only. */
    if (!payload->inplace && payload->region_node_count == 4 &&
        payload->region_input_count == 5 &&
        payload->region_nodes[0].op == NUMPY_JIT_KERNEL_REGION_MUL &&
        payload->region_nodes[1].op == NUMPY_JIT_KERNEL_REGION_MUL &&
        payload->region_nodes[2].op == NUMPY_JIT_KERNEL_REGION_ADD &&
        payload->region_nodes[3].op == NUMPY_JIT_KERNEL_REGION_ADD) {
        int t0 = payload->region_input_count;
        int t1 = t0 + 1;
        int t2 = t1 + 1;
        NumpyJitKernelRegionNode *add = &payload->region_nodes[2];
        NumpyJitKernelRegionNode *tail = &payload->region_nodes[3];
        int joins_products =
            (add->left == t0 && add->right == t1) ||
            (add->left == t1 && add->right == t0);
        int consumes_sum = tail->left == t2 || tail->right == t2;
        int product_inputs_external =
            payload->region_nodes[0].left < payload->region_input_count &&
            payload->region_nodes[0].right < payload->region_input_count &&
            payload->region_nodes[1].left < payload->region_input_count &&
            payload->region_nodes[1].right < payload->region_input_count;
        if (joins_products && consumes_sum && product_inputs_external) {
            payload->region_compiled_kind = 1;
        }
    }

    /* Compile the five-node stencil shape

           t0 = a * scale
           t1 = b + c
           t2 = t1 + d
           t3 = t2 + e
           out = t0 - t3          (or the reversed subtraction)

       as one traversal.  The graph test is deliberately structural: names,
       bytecode locations and concrete array properties are not part of the
       lowering decision. */
    if (!payload->inplace && payload->region_node_count == 5 &&
        payload->region_input_count == 6 &&
        payload->region_nodes[0].op == NUMPY_JIT_KERNEL_REGION_MUL &&
        payload->region_nodes[1].op == NUMPY_JIT_KERNEL_REGION_ADD &&
        payload->region_nodes[2].op == NUMPY_JIT_KERNEL_REGION_ADD &&
        payload->region_nodes[3].op == NUMPY_JIT_KERNEL_REGION_ADD &&
        payload->region_nodes[4].op == NUMPY_JIT_KERNEL_REGION_SUB) {
        int t0 = payload->region_input_count;
        int t1 = t0 + 1;
        int t2 = t1 + 1;
        int t3 = t2 + 1;
        NumpyJitKernelRegionNode *add0 = &payload->region_nodes[1];
        NumpyJitKernelRegionNode *add1 = &payload->region_nodes[2];
        NumpyJitKernelRegionNode *add2 = &payload->region_nodes[3];
        NumpyJitKernelRegionNode *sub = &payload->region_nodes[4];
        int shape_ok =
            payload->region_nodes[0].left < payload->region_input_count &&
            payload->region_nodes[0].right < payload->region_input_count &&
            add0->left < payload->region_input_count &&
            add0->right < payload->region_input_count &&
            ((add1->left == t1 &&
              add1->right < payload->region_input_count) ||
             (add1->right == t1 &&
              add1->left < payload->region_input_count)) &&
            ((add2->left == t2 &&
              add2->right < payload->region_input_count) ||
             (add2->right == t2 &&
              add2->left < payload->region_input_count)) &&
            ((sub->left == t0 && sub->right == t3) ||
             (sub->right == t0 && sub->left == t3));
        if (shape_ok) {
            payload->region_compiled_kind = 2;
        }
    }

    /* Reuse the provider's row/strided fused kernel for
       ``(a +/- b) * scalar`` and ``(a +/- b) / scalar``.  This admits the
       same semantic schedule for positive-strided stencil views that the
       older atomic-layout matcher could only recognize for C-contiguous
       operands. */
    if (!payload->inplace && payload->region_node_count == 2 &&
        payload->region_input_count == 3 &&
        (payload->region_nodes[0].op == NUMPY_JIT_KERNEL_REGION_ADD ||
         payload->region_nodes[0].op == NUMPY_JIT_KERNEL_REGION_SUB) &&
        (payload->region_nodes[1].op == NUMPY_JIT_KERNEL_REGION_MUL ||
         payload->region_nodes[1].op == NUMPY_JIT_KERNEL_REGION_DIV)) {
        int t0 = payload->region_input_count;
        NumpyJitKernelRegionNode *first = &payload->region_nodes[0];
        NumpyJitKernelRegionNode *tail = &payload->region_nodes[1];
        int consumes_first = tail->left == t0 &&
            tail->right < payload->region_input_count;
        int consumes_second = tail->right == t0 &&
            tail->left < payload->region_input_count;
        int shape_ok = first->left < payload->region_input_count &&
            first->right < payload->region_input_count &&
            (consumes_first ||
             (tail->op == NUMPY_JIT_KERNEL_REGION_MUL && consumes_second));
        /* ``a * (b - c)`` has the same opcode-only fingerprint as the
         * scalar affine form, but puts the temporary on the right.  It is a
         * three-array expression, not an affine scalar tail; installing a
         * generic Region would replace two already-hot atomic sites with a
         * Region entry that can never use the schedule-5 kernel.  Reject it
         * while building the plan so CPython keeps the independent atomic
         * specialization path. */
        if (consumes_second) {
            PyMem_Free(payload);
            return NULL;
        }
        if (shape_ok) {
            payload->region_compiled_kind = 5;
        }
    }

    /* Compile ``(a * b) * scalar`` (including scalar-first tail order). */
    if (!payload->inplace && payload->region_node_count == 2 &&
        payload->region_input_count == 3 &&
        payload->region_nodes[0].op == NUMPY_JIT_KERNEL_REGION_MUL &&
        payload->region_nodes[1].op == NUMPY_JIT_KERNEL_REGION_MUL) {
        int t0 = payload->region_input_count;
        NumpyJitKernelRegionNode *first = &payload->region_nodes[0];
        NumpyJitKernelRegionNode *tail = &payload->region_nodes[1];
        int shape_ok = first->left < payload->region_input_count &&
            first->right < payload->region_input_count &&
            ((tail->left == t0 &&
              tail->right < payload->region_input_count) ||
             (tail->right == t0 &&
              tail->left < payload->region_input_count));
        if (shape_ok) {
            payload->region_compiled_kind = 6;
        }
    }

    /* Compile ``square(array) +/- other`` as one traversal.  The exponent
       remains an explicit IR input and is guarded as exact 2 by the common
       Region validator; this schedule only removes node dispatch. */
    if (!payload->inplace && payload->region_node_count == 2 &&
        payload->region_input_count == 3 &&
        payload->region_nodes[0].op == NUMPY_JIT_KERNEL_REGION_SQUARE &&
        (payload->region_nodes[1].op == NUMPY_JIT_KERNEL_REGION_ADD ||
         payload->region_nodes[1].op == NUMPY_JIT_KERNEL_REGION_SUB)) {
        int t0 = payload->region_input_count;
        NumpyJitKernelRegionNode *square = &payload->region_nodes[0];
        NumpyJitKernelRegionNode *tail = &payload->region_nodes[1];
        int shape_ok = square->left < payload->region_input_count &&
            square->right < payload->region_input_count &&
            ((tail->left == t0 &&
              tail->right < payload->region_input_count) ||
             (tail->right == t0 &&
              tail->left < payload->region_input_count));
        if (shape_ok) {
            payload->region_compiled_kind = 3;
        }
    }

    /* Compile ``a**2 + b**2 + c**2`` as one traversal.  This is a semantic
       DAG shape rather than a source-level pattern: all three exponent
       inputs remain explicit and are guarded by the common square validator. */
    if (!payload->inplace && payload->region_node_count == 5 &&
        payload->region_input_count == 6 &&
        payload->region_nodes[0].op == NUMPY_JIT_KERNEL_REGION_SQUARE &&
        payload->region_nodes[1].op == NUMPY_JIT_KERNEL_REGION_SQUARE &&
        payload->region_nodes[2].op == NUMPY_JIT_KERNEL_REGION_ADD &&
        payload->region_nodes[3].op == NUMPY_JIT_KERNEL_REGION_SQUARE &&
        payload->region_nodes[4].op == NUMPY_JIT_KERNEL_REGION_ADD) {
        int t0 = payload->region_input_count;
        int t1 = t0 + 1;
        int t2 = t1 + 1;
        int t3 = t2 + 1;
        NumpyJitKernelRegionNode *square0 = &payload->region_nodes[0];
        NumpyJitKernelRegionNode *square1 = &payload->region_nodes[1];
        NumpyJitKernelRegionNode *add0 = &payload->region_nodes[2];
        NumpyJitKernelRegionNode *square2 = &payload->region_nodes[3];
        NumpyJitKernelRegionNode *add1 = &payload->region_nodes[4];
        int shape_ok =
            square0->left < payload->region_input_count &&
            square0->right < payload->region_input_count &&
            square1->left < payload->region_input_count &&
            square1->right < payload->region_input_count &&
            ((add0->left == t0 && add0->right == t1) ||
             (add0->left == t1 && add0->right == t0)) &&
            square2->left < payload->region_input_count &&
            square2->right < payload->region_input_count &&
            ((add1->left == t2 && add1->right == t3) ||
             (add1->left == t3 && add1->right == t2));
        if (shape_ok) {
            payload->region_compiled_kind = 4;
        }
    }

    if (numpy_jit_stats_enabled()) {
        const char *dir = getenv("PYTHON_JIT_EXT_STATS_DIR");
        if (dir == NULL || dir[0] == '\0') {
            dir = "/tmp";
        }
        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", dir,
                 "crossjit_numpy_region_candidates.ndjson");
        FILE *fp = fopen(path, "a");
        if (fp != NULL) {
            fprintf(fp,
                    "{\"graph_hash\":%" PRIu64
                    ",\"node_count\":%u,\"input_count\":%u,"
                    "\"schedule_id\":%u,\"compiled_kind\":%u,\"nodes\":[",
                    request->graph_hash, (unsigned)request->node_count,
                    (unsigned)request->input_count,
                    (unsigned)payload->region_compiled_kind,
                    (unsigned)payload->region_compiled_kind);
            for (int i = 0; i < payload->region_node_count; i++) {
                NumpyJitKernelRegionNode *node = &payload->region_nodes[i];
                fprintf(fp,
                        "%s{\"op\":%u,\"left\":%u,\"right\":%u,"
                        "\"out\":%u}",
                        i == 0 ? "" : ",", (unsigned)node->op,
                        (unsigned)node->left, (unsigned)node->right,
                        (unsigned)node->out);
            }
            fputs("]}\n", fp);
            fclose(fp);
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
    if (defer_to_atomic_prefix && payload->region_compiled_kind == 0) {
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
    if (payload->region_compiled_kind != 0) {
        plan->flags |= PYJIT_EXT_PLAN_ENTRY_PROFITABLE;
        /* ENTRY_PROFITABLE is a speculative admission decision made from
           structural IR before concrete arrays are visible.  If runtime
           specialization rejects the observed shape/layout, report MISS so
           CPython invalidates and negatively caches the site.  Returning
           COLD here would retain the entry executor indefinitely while it
           replays the slower provider DAG fallback on every call. */
        payload->region_runtime_miss_on_decline = 1;
    }
    else {
        /* A generic Region may share an entry executor with a later
           profitable schedule.  If its runtime properties cannot select a
           lowering, invalidate it instead of retaining provider fallback on
           every entry. */
        payload->region_runtime_miss_on_decline = 1;
        for (int i = 0; i < payload->region_node_count; i++) {
            if (payload->region_ops[i] == NUMPY_JIT_OP_POWER) {
                /* A non-square exponent is a property miss, not structural.
                 * Preserve the site for a later exponent=2 PIC entry. */
                payload->region_runtime_miss_on_decline = 0;
                break;
            }
        }
    }
    for (int i = 0; i < payload->region_node_count; i++) {
        if (payload->region_ops[i] == NUMPY_JIT_OP_POWER) {
            payload->region_runtime_miss_on_decline = 0;
            break;
        }
    }
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
numpy_jit_build_masked_mean_region(
        const PyJitExtRegionRequest *request, PyJitExtSiteState *site)
{
    if (request->node_count != 5 || request->input_count > 8) {
        return NULL;
    }
    PyJitExtPlan *atomic[5];
    NumpyJitPlanPayload *atomic_payload[5];
    for (int i = 0; i < 5; i++) {
        atomic[i] = (PyJitExtPlan *)request->nodes[i].atomic_plan;
        if (atomic[i] == NULL || atomic[i]->payload == NULL ||
            atomic[i]->provider != &numpy_jit_provider) {
            return NULL;
        }
        atomic_payload[i] = (NumpyJitPlanPayload *)atomic[i]->payload;
    }
    int joins_compares =
        ((request->nodes[2].inputs[0] == request->nodes[0].output &&
          request->nodes[2].inputs[1] == request->nodes[1].output) ||
         (request->nodes[2].inputs[1] == request->nodes[0].output &&
          request->nodes[2].inputs[0] == request->nodes[1].output));
    int subscript_consumes_mask =
        request->nodes[3].inputs[0] == request->nodes[2].output ||
        request->nodes[3].inputs[1] == request->nodes[2].output;
    int mean_consumes_values = request->nodes[4].input_count == 1 &&
        request->nodes[4].inputs[0] == request->nodes[3].output;
    if (request->nodes[0].input_count != 2 ||
        request->nodes[1].input_count != 2 ||
        request->nodes[2].input_count != 2 ||
        request->nodes[3].input_count != 2 ||
        (atomic_payload[0]->op != NUMPY_JIT_OP_LESS &&
         atomic_payload[0]->op != NUMPY_JIT_OP_LESS_EQUAL) ||
        (atomic_payload[1]->op != NUMPY_JIT_OP_LESS &&
         atomic_payload[1]->op != NUMPY_JIT_OP_LESS_EQUAL) ||
        atomic_payload[2]->op != NUMPY_JIT_OP_LOGICAL_AND ||
        atomic_payload[3]->op != NUMPY_JIT_OP_BOOLEAN_SUBSCRIPT ||
        atomic_payload[4]->op != NUMPY_JIT_OP_MEAN ||
        !joins_compares || !subscript_consumes_mask ||
        !mean_consumes_values) {
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
    payload->region_masked_mean = 1;
    payload->region_runtime_miss_on_decline = 1;
    payload->region_node_count = 5;
    payload->region_input_count = request->input_count;
    payload->region_output_slot = request->input_count + 4;
    payload->region_mutation_input = -1;
    payload->min_elements = 1;
    payload->site = site;
    for (int i = 0; i < request->input_count; i++) {
        payload->region_input_ids[i] = request->inputs[i].value_id;
    }
    for (int i = 0; i < 5; i++) {
        int left = numpy_region_value_slot(
            payload, request, request->nodes[i].inputs[0], i);
        int right = request->nodes[i].input_count == 2 ?
            numpy_region_value_slot(
                payload, request, request->nodes[i].inputs[1], i) : left;
        if (left < 0 || right < 0) {
            PyMem_Free(payload);
            PyMem_Free(plan);
            return NULL;
        }
        payload->region_left[i] = (uint8_t)left;
        payload->region_right[i] = (uint8_t)right;
        payload->region_ops[i] = atomic_payload[i]->op;
        payload->region_atomic_plans[i] = atomic[i];
    }
    payload->region_owner_plan = plan;
    numpy_jit_link_plan_stats(payload);
    plan->abi_version = PYJIT_EXT_ABI_VERSION;
    plan->struct_size = sizeof(*plan);
    plan->kind = PYJIT_EXT_PLAN_REGION;
    plan->provider = &numpy_jit_provider;
    plan->flags = PYJIT_EXT_PLAN_REQUIRES_GIL |
                  PYJIT_EXT_PLAN_NO_PYTHON_CALLBACK |
                  PYJIT_EXT_PLAN_ALLOCATES_RESULT |
                  PYJIT_EXT_PLAN_ENTRY_PROFITABLE;
    plan->op_id = 4500u;
    plan->payload = payload;
    plan->eval.region_eval = numpy_region_eval;
    plan->region_fallback_eval = NULL;
    plan->effect_flags = PYJIT_EXT_EFFECT_PURE |
                         PYJIT_EXT_EFFECT_NO_PYTHON_CALLBACK |
                         PYJIT_EXT_EFFECT_OWNS_ATOMIC_PLANS;
    plan->mutation_input = -1;
    plan->region_input_count = request->input_count;
    plan->op_domain = PYJIT_EXT_OP_DOMAIN_REDUCTION;
    plan->output_count = 1;
    plan->semantic_version = 1;
    plan->debug_name = "numpy.region.masked_mean";
    return plan;
}

static PyJitExtPlan *
numpy_jit_build_compare_logical_region(
        const PyJitExtRegionRequest *request, PyJitExtSiteState *site)
{
    if (request->node_count != 3 || request->input_count > 8) {
        return NULL;
    }
    PyJitExtPlan *atomic[3] = {
        (PyJitExtPlan *)request->nodes[0].atomic_plan,
        (PyJitExtPlan *)request->nodes[1].atomic_plan,
        (PyJitExtPlan *)request->nodes[2].atomic_plan,
    };
    for (int i = 0; i < 3; i++) {
        if (atomic[i] == NULL || atomic[i]->payload == NULL ||
            atomic[i]->provider != &numpy_jit_provider ||
            request->nodes[i].input_count != 2) {
            return NULL;
        }
    }
    NumpyJitPlanPayload *p0 = (NumpyJitPlanPayload *)atomic[0]->payload;
    NumpyJitPlanPayload *p1 = (NumpyJitPlanPayload *)atomic[1]->payload;
    NumpyJitPlanPayload *p2 = (NumpyJitPlanPayload *)atomic[2]->payload;
    int joins =
        ((request->nodes[2].inputs[0] == request->nodes[0].output &&
          request->nodes[2].inputs[1] == request->nodes[1].output) ||
         (request->nodes[2].inputs[1] == request->nodes[0].output &&
          request->nodes[2].inputs[0] == request->nodes[1].output));
    if ((p0->op != NUMPY_JIT_OP_LESS &&
         p0->op != NUMPY_JIT_OP_LESS_EQUAL) ||
        (p1->op != NUMPY_JIT_OP_LESS &&
         p1->op != NUMPY_JIT_OP_LESS_EQUAL) ||
        p2->op != NUMPY_JIT_OP_LOGICAL_AND || !joins) {
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
    payload->region_compare_logical = 1;
    payload->region_runtime_miss_on_decline = 1;
    payload->region_node_count = 3;
    payload->region_input_count = request->input_count;
    payload->region_output_slot = request->input_count + 2;
    payload->region_mutation_input = -1;
    payload->min_elements = 1;
    payload->site = site;
    for (int i = 0; i < request->input_count; i++) {
        payload->region_input_ids[i] = request->inputs[i].value_id;
    }
    for (int i = 0; i < 3; i++) {
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
        payload->region_ops[i] =
            ((NumpyJitPlanPayload *)atomic[i]->payload)->op;
        payload->region_atomic_plans[i] = atomic[i];
    }
    payload->region_owner_plan = plan;
    numpy_jit_link_plan_stats(payload);
    plan->abi_version = PYJIT_EXT_ABI_VERSION;
    plan->struct_size = sizeof(*plan);
    plan->kind = PYJIT_EXT_PLAN_REGION;
    plan->provider = &numpy_jit_provider;
    plan->flags = PYJIT_EXT_PLAN_REQUIRES_GIL |
                  PYJIT_EXT_PLAN_NO_PYTHON_CALLBACK |
                  PYJIT_EXT_PLAN_ALLOCATES_RESULT |
                  PYJIT_EXT_PLAN_ENTRY_PROFITABLE;
    plan->op_id = 4400u;
    plan->payload = payload;
    plan->eval.region_eval = numpy_region_eval;
    plan->region_fallback_eval = NULL;
    plan->effect_flags = PYJIT_EXT_EFFECT_PURE |
                         PYJIT_EXT_EFFECT_NO_PYTHON_CALLBACK |
                         PYJIT_EXT_EFFECT_OWNS_ATOMIC_PLANS;
    plan->mutation_input = -1;
    plan->region_input_count = request->input_count;
    plan->op_domain = PYJIT_EXT_OP_DOMAIN_ELEMENTWISE;
    plan->output_count = 1;
    plan->semantic_version = 1;
    plan->debug_name = "numpy.region.compare_logical_and";
    return plan;
}

static PyJitExtPlan *
numpy_jit_build_outer_minimum_region(
        const PyJitExtRegionRequest *request, PyJitExtSiteState *site)
{
    if (request->node_count != 2 || request->input_count != 7) {
        return NULL;
    }
    PyJitExtPlan *outer =
        (PyJitExtPlan *)request->nodes[0].atomic_plan;
    PyJitExtPlan *minimum =
        (PyJitExtPlan *)request->nodes[1].atomic_plan;
    if (outer == NULL || minimum == NULL || outer->payload == NULL ||
        minimum->payload == NULL || outer->provider != &numpy_jit_provider ||
        minimum->provider != &numpy_jit_provider ||
        request->nodes[0].input_count != 2 ||
        request->nodes[1].input_count != 2) {
        return NULL;
    }
    NumpyJitPlanPayload *outer_payload =
        (NumpyJitPlanPayload *)outer->payload;
    NumpyJitPlanPayload *minimum_payload =
        (NumpyJitPlanPayload *)minimum->payload;
    if (outer_payload->op != NUMPY_JIT_OP_OUTER_ADD ||
        minimum_payload->op != NUMPY_JIT_OP_MINIMUM ||
        (request->nodes[1].inputs[0] != request->nodes[0].output &&
         request->nodes[1].inputs[1] != request->nodes[0].output)) {
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
    payload->region_outer_minimum = 1;
    payload->region_runtime_miss_on_decline = 1;
    payload->region_node_count = 2;
    payload->region_input_count = request->input_count;
    payload->region_output_slot = request->input_count + 1;
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
    payload->region_ops[0] = NUMPY_JIT_OP_OUTER_ADD;
    payload->region_ops[1] = NUMPY_JIT_OP_MINIMUM;
    payload->region_owner_plan = plan;
    numpy_jit_link_plan_stats(payload);

    plan->abi_version = PYJIT_EXT_ABI_VERSION;
    plan->struct_size = sizeof(*plan);
    plan->kind = PYJIT_EXT_PLAN_REGION;
    plan->provider = &numpy_jit_provider;
    plan->flags = PYJIT_EXT_PLAN_REQUIRES_GIL |
                  PYJIT_EXT_PLAN_NO_PYTHON_CALLBACK |
                  PYJIT_EXT_PLAN_ALLOCATES_RESULT |
                  PYJIT_EXT_PLAN_ENTRY_PROFITABLE;
    plan->op_id = 4300u;
    plan->payload = payload;
    plan->eval.region_eval = numpy_region_eval;
    plan->region_fallback_eval = NULL;
    plan->effect_flags = PYJIT_EXT_EFFECT_PURE |
                         PYJIT_EXT_EFFECT_NO_PYTHON_CALLBACK |
                         PYJIT_EXT_EFFECT_OWNS_ATOMIC_PLANS;
    plan->mutation_input = -1;
    plan->region_input_count = request->input_count;
    plan->op_domain = PYJIT_EXT_OP_DOMAIN_ELEMENTWISE;
    plan->output_count = 1;
    plan->semantic_version = 1;
    plan->debug_name = "numpy.region.outer_add_minimum";
    return plan;
}

static PyJitExtPlan *
numpy_jit_prepare_region(const PyJitExtRegionRequest *request,
                     PyJitExtSiteState *site)
{
    if (request == NULL || request->provider != &numpy_jit_provider ||
        request->nodes == NULL || request->inputs == NULL ||
        request->ir_version != PYJIT_EXT_CORE_IR_VERSION) {
        return NULL;
    }
    for (int i = 0; i < request->node_count; i++) {
        const PyJitExtRegionNode *node = &request->nodes[i];
        if (node->dialect_id != NUMPY_JIT_DIALECT_ID ||
            node->semantic_version != 1 || node->output_count != 1) {
            return NULL;
        }
    }
    PyJitExtPlan *outer_minimum =
        numpy_jit_build_outer_minimum_region(request, site);
    if (outer_minimum != NULL || PyErr_Occurred()) {
        return outer_minimum;
    }
    PyJitExtPlan *masked_mean =
        numpy_jit_build_masked_mean_region(request, site);
    if (masked_mean != NULL || PyErr_Occurred()) {
        return masked_mean;
    }
    PyJitExtPlan *compare_logical =
        numpy_jit_build_compare_logical_region(request, site);
    if (compare_logical != NULL || PyErr_Occurred()) {
        return compare_logical;
    }
    PyJitExtPlan *matmul_epilogue =
        numpy_jit_build_matmul_epilogue_region(request, site);
    if (matmul_epilogue != NULL || PyErr_Occurred()) {
        return matmul_epilogue;
    }
    PyJitExtPlan *matmul_prologue =
        numpy_jit_build_matmul_prologue_region(request, site);
    if (matmul_prologue != NULL || PyErr_Occurred()) {
        return matmul_prologue;
    }
    PyJitExtPlan *matmul_scaled_add =
        numpy_jit_build_matmul_scaled_add_region(request, site);
    if (matmul_scaled_add != NULL || PyErr_Occurred()) {
        return matmul_scaled_add;
    }
    if (request->node_count == 4 && request->input_count == 5) {
        NumpyJitPlanPayload *p0 = request->nodes[0].atomic_plan != NULL ?
            (NumpyJitPlanPayload *)request->nodes[0].atomic_plan->payload : NULL;
        NumpyJitPlanPayload *p1 = request->nodes[1].atomic_plan != NULL ?
            (NumpyJitPlanPayload *)request->nodes[1].atomic_plan->payload : NULL;
        NumpyJitPlanPayload *p2 = request->nodes[2].atomic_plan != NULL ?
            (NumpyJitPlanPayload *)request->nodes[2].atomic_plan->payload : NULL;
        NumpyJitPlanPayload *p3 = request->nodes[3].atomic_plan != NULL ?
            (NumpyJitPlanPayload *)request->nodes[3].atomic_plan->payload : NULL;
        int joins_products =
            (request->nodes[2].inputs[0] == request->nodes[0].output &&
             request->nodes[2].inputs[1] == request->nodes[1].output) ||
            (request->nodes[2].inputs[1] == request->nodes[0].output &&
             request->nodes[2].inputs[0] == request->nodes[1].output);
        int consumes_sum =
            request->nodes[3].inputs[0] == request->nodes[2].output ||
            request->nodes[3].inputs[1] == request->nodes[2].output;
        if (p0 != NULL && p1 != NULL && p2 != NULL && p3 != NULL &&
            p0->op == NUMPY_JIT_OP_MULTIPLY &&
            p1->op == NUMPY_JIT_OP_MULTIPLY &&
            p2->op == NUMPY_JIT_OP_ADD && p3->op == NUMPY_JIT_OP_ADD &&
            joins_products && consumes_sum) {
            return numpy_jit_build_generic_region(request, site);
        }
    }
    if (request->node_count == 5 && request->input_count == 6) {
        static const NumpyJitOp expected[5] = {
            NUMPY_JIT_OP_MULTIPLY,
            NUMPY_JIT_OP_ADD,
            NUMPY_JIT_OP_ADD,
            NUMPY_JIT_OP_ADD,
            NUMPY_JIT_OP_SUBTRACT,
        };
        int matches = 1;
        for (int i = 0; i < 5; i++) {
            NumpyJitPlanPayload *atomic =
                request->nodes[i].atomic_plan != NULL ?
                (NumpyJitPlanPayload *)
                    request->nodes[i].atomic_plan->payload : NULL;
            if (atomic == NULL || atomic->op != expected[i]) {
                matches = 0;
                break;
            }
        }
        if (matches) {
            PyJitExtPlan *compiled =
                numpy_jit_build_generic_region(request, site);
            if (compiled != NULL &&
                ((NumpyJitPlanPayload *)compiled->payload)
                    ->region_compiled_kind == 2) {
                return compiled;
            }
            if (compiled != NULL) {
                NumpyJitPlanPayload *compiled_payload =
                    (NumpyJitPlanPayload *)compiled->payload;
                memset(compiled_payload->region_atomic_plans, 0,
                       sizeof(compiled_payload->region_atomic_plans));
                compiled_payload->region_node_count = 0;
                numpy_jit_free_plan(compiled);
            }
        }
    }
    /* Every structural candidate, including two-node affine expressions,
       is represented by the same immutable Region program.  Dedicated
       kernels are variants produced from that program, never a separate
       fused-binary2 plan. */
    return numpy_jit_build_generic_region(request, site);

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
        NumpyJitResultCache *shared_cache = payload->shared_result_cache;
        if (shared_cache == NULL) {
            numpy_jit_record_result_cache_stats(
                payload, &payload->result_cache);
        }
        else {
            int has_other_live_owner = 0;
            for (NumpyJitPlanPayload *other = numpy_jit_plan_stats_list;
                 other != NULL; other = other->stats_next) {
                if (other != payload &&
                    other->shared_result_cache == shared_cache) {
                    has_other_live_owner = 1;
                    break;
                }
            }
            if (!has_other_live_owner) {
                /* The site retains the buffer for the next executor
                   generation.  Move only counter deltas to dead-plan stats
                   so later generations neither lose nor double-count them. */
                numpy_jit_record_result_cache_stats(payload, shared_cache);
                shared_cache->hit = 0;
                shared_cache->miss = 0;
                shared_cache->refcnt_miss = 0;
                shared_cache->shape_miss = 0;
            }
        }
        numpy_jit_unlink_plan_stats(payload);
        numpy_jit_retire_llvm_stats(payload);
        if (payload->shared_result_cache == NULL) {
            Py_XDECREF(payload->result_cache.result);
        }
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

static int
numpy_jit_is_supported_region_dtype(int dtype)
{
    return numpy_jit_is_supported_float_dtype(dtype) ||
           dtype == NPY_INT || dtype == NPY_LONG || dtype == NPY_LONGLONG;
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
    else if (PyArray_IsScalar(scalar, SignedInteger)) {
        npy_int64 integer = (npy_int64)PyLong_AsLongLong(scalar);
        if (integer == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            numpy_jit_guard_fail(
                payload, NUMPY_JIT_GUARD_SCALAR_CONVERT_FAILED);
            return 0;
        }
        /* Numeric scalar slots are language-neutral doubles in Core IR v1.
           Only values represented exactly are admitted to an integer Region;
           larger values safely fall back to NumPy. */
        if (integer < -9007199254740992LL ||
            integer > 9007199254740992LL) {
            numpy_jit_guard_fail(
                payload, NUMPY_JIT_GUARD_SCALAR_CONVERT_FAILED);
            return 0;
        }
        value = (double)integer;
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
numpy_compare_binary_eval(void *payload, PyObject *left, PyObject *right,
                          PyObject **result)
{
    NumpyJitPlanPayload *plan = (NumpyJitPlanPayload *)payload;
    if ((!PyArray_CheckExact(left) && !PyArray_CheckExact(right)) ||
        (plan->op != NUMPY_JIT_OP_LESS &&
         plan->op != NUMPY_JIT_OP_LESS_EQUAL)) {
        return PYJIT_EXT_MISS;
    }
    int richcompare = plan->op == NUMPY_JIT_OP_LESS ? Py_LT : Py_LE;
    PyObject *value = PyObject_RichCompare(left, right, richcompare);
    if (value == NULL) {
        return PYJIT_EXT_ERROR;
    }
    *result = value;
    return PYJIT_EXT_HIT;
}

static PyJitExtResult
numpy_boolean_subscript_binary_eval(void *NPY_UNUSED(payload), PyObject *left,
                                    PyObject *right, PyObject **result)
{
    if (!PyArray_CheckExact(left) || !PyArray_CheckExact(right)) {
        return PYJIT_EXT_MISS;
    }
    PyObject *value = PyObject_GetItem(left, right);
    if (value == NULL) {
        return PYJIT_EXT_ERROR;
    }
    *result = value;
    return PYJIT_EXT_HIT;
}

static PyJitExtResult
numpy_mean_call_eval(void *payload, PyObject *callable, PyObject *receiver,
                     PyObject *const *NPY_UNUSED(args), Py_ssize_t nargs,
                     PyObject **result)
{
    NumpyJitPlanPayload *plan = (NumpyJitPlanPayload *)payload;
    if (nargs != 0 || receiver == NULL ||
        callable != plan->expected_callable ||
        !PyArray_CheckExact(receiver)) {
        return PYJIT_EXT_MISS;
    }
    PyObject *value = PyObject_CallOneArg(callable, receiver);
    if (value == NULL) {
        return PYJIT_EXT_ERROR;
    }
    *result = value;
    return PYJIT_EXT_HIT;
}

static PyJitExtResult
numpy_outer_add_call_eval(void *payload, PyObject *callable,
                          PyObject *receiver, PyObject *const *args,
                          Py_ssize_t nargs, PyObject **result)
{
    NumpyJitPlanPayload *plan = (NumpyJitPlanPayload *)payload;
    int bound_match = receiver == NULL && PyCFunction_Check(callable) &&
        PyCFunction_GET_SELF(callable) == plan->expected_receiver &&
        PyCFunction_GET_FUNCTION(callable) == plan->expected_call_target;
    int descriptor_match = receiver == plan->expected_receiver &&
        callable == plan->expected_callable;
    if ((!bound_match && !descriptor_match) || nargs != 2 ||
        !PyArray_CheckExact(args[0]) || !PyArray_CheckExact(args[1])) {
        return PYJIT_EXT_MISS;
    }
    PyObject *call_args[3] = {receiver, args[0], args[1]};
    PyObject *value = bound_match ?
        PyObject_Vectorcall(callable, &call_args[1], 2, NULL) :
        PyObject_Vectorcall(callable, call_args, 3, NULL);
    if (value == NULL) {
        return PYJIT_EXT_ERROR;
    }
    *result = value;
    return PYJIT_EXT_HIT;
}

static PyJitExtResult
numpy_passthrough_binary_call_eval(void *payload, PyObject *callable,
                                   PyObject *receiver,
                                   PyObject *const *args,
                                   Py_ssize_t nargs, PyObject **result)
{
    NumpyJitPlanPayload *plan = (NumpyJitPlanPayload *)payload;
    if (receiver != NULL || callable != plan->expected_callable ||
        nargs != 2 || !PyArray_CheckExact(args[0]) ||
        !PyArray_CheckExact(args[1])) {
        return PYJIT_EXT_MISS;
    }
    PyObject *value = PyObject_Vectorcall(callable, args, 2, NULL);
    if (value == NULL) {
        return PYJIT_EXT_ERROR;
    }
    *result = value;
    return PYJIT_EXT_HIT;
}

static PyJitExtResult
numpy_ufunc2_f64_call_eval(void *payload, PyObject *callable,
                           PyObject *receiver, PyObject *const *args,
                           Py_ssize_t nargs,
                           PyObject **result)
{
    NumpyJitPlanPayload *plan = (NumpyJitPlanPayload *)payload;
    if (receiver != NULL || nargs != 2 ||
        callable != plan->expected_callable) {
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
        else if (ndim == 2 && PyArray_ISALIGNED(a) &&
                 PyArray_ISALIGNED(b) &&
                 PyArray_STRIDE(a, 0) > 0 && PyArray_STRIDE(a, 1) > 0 &&
                 PyArray_STRIDE(b, 0) > 0 && PyArray_STRIDE(b, 1) > 0) {
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
    if (!PyArray_CheckExact(second_other) || size < payload->min_elements) {
        return PYJIT_EXT_MISS;
    }

    PyArrayObject *c = (PyArrayObject *)second_other;
    int ndim = PyArray_NDIM(a);
    int c_layout = numpy_jit_region_layout_class(c);
    if (PyArray_TYPE(c) != dtype || PyArray_NDIM(c) != ndim ||
        c_layout == 0) {
        return PYJIT_EXT_MISS;
    }
    for (int i = 0; i < ndim; i++) {
        if (PyArray_DIM(c, i) != PyArray_DIM(a, i)) {
            return PYJIT_EXT_MISS;
        }
    }

    if (payload->inplace) {
        if (!PyArray_ISWRITEABLE(c) ||
            solve_may_share_memory(a, c, NPY_MAY_SHARE_BOUNDS) != 0 ||
            solve_may_share_memory(b, c, NPY_MAY_SHARE_BOUNDS) != 0) {
            return PYJIT_EXT_MISS;
        }
        int kernel_ok = layout == NUMPY_JIT_EVAL_LAYOUT_C_CONTIG &&
                        PyArray_IS_C_CONTIGUOUS(c) ?
            numpy_jit_kernel_add3_contig(
                dtype, PyArray_BYTES(a), PyArray_BYTES(b), PyArray_BYTES(c),
                PyArray_BYTES(c), size, 1) :
            numpy_jit_kernel_add3_strided(
                dtype, ndim, PyArray_DIMS(a), PyArray_BYTES(a),
                PyArray_STRIDES(a), PyArray_BYTES(b), PyArray_STRIDES(b),
                PyArray_BYTES(c), PyArray_STRIDES(c), PyArray_BYTES(c),
                size, 1);
        if (!kernel_ok) {
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
    int order = (payload->fused_flags &
                 PYJIT_EXT_FUSED_BINARY2_SECOND_CONSUMES_RIGHT) ? 2 : 0;
    int kernel_ok = layout == NUMPY_JIT_EVAL_LAYOUT_C_CONTIG &&
                    PyArray_IS_C_CONTIGUOUS(c) ?
        numpy_jit_kernel_add3_contig(
            dtype, PyArray_BYTES(a), PyArray_BYTES(b), PyArray_BYTES(c),
            PyArray_BYTES(out), size, order) :
        numpy_jit_kernel_add3_strided(
            dtype, ndim, PyArray_DIMS(a), PyArray_BYTES(a),
            PyArray_STRIDES(a), PyArray_BYTES(b), PyArray_STRIDES(b),
            PyArray_BYTES(c), PyArray_STRIDES(c), PyArray_BYTES(out),
            size, order);
    if (!kernel_ok) {
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
numpy_region_matmul_prologue_eval(NumpyJitPlanPayload *payload,
                                  PyObject *const *args, Py_ssize_t nargs,
                                  PyObject **result)
{
    if (nargs != 3) {
        return PYJIT_EXT_MISS;
    }
    int mul_left = payload->region_left[0];
    int mul_right = payload->region_right[0];
    int internal_slot = payload->region_input_count;
    int matmul_left = payload->region_left[1];
    int matmul_right = payload->region_right[1];
    int internal_is_left = matmul_left == internal_slot;
    int other_slot = internal_is_left ? matmul_right : matmul_left;
    if (mul_left < 0 || mul_left >= nargs ||
        mul_right < 0 || mul_right >= nargs ||
        other_slot < 0 || other_slot >= nargs ||
        (internal_is_left == (matmul_right == internal_slot))) {
        return PYJIT_EXT_MISS;
    }
    int array_slot;
    int scalar_slot;
    int scalar_first;
    if (PyArray_CheckExact(args[mul_left]) &&
        !PyArray_CheckExact(args[mul_right])) {
        array_slot = mul_left;
        scalar_slot = mul_right;
        scalar_first = 0;
    }
    else if (!PyArray_CheckExact(args[mul_left]) &&
             PyArray_CheckExact(args[mul_right])) {
        array_slot = mul_right;
        scalar_slot = mul_left;
        scalar_first = 1;
    }
    else {
        return PYJIT_EXT_MISS;
    }
    if (!PyArray_CheckExact(args[other_slot])) {
        return PYJIT_EXT_MISS;
    }
    PyArrayObject *array = (PyArrayObject *)args[array_slot];
    PyArrayObject *other = (PyArrayObject *)args[other_slot];
    int dtype = PyArray_TYPE(array);
    if (!numpy_jit_is_supported_float_dtype(dtype) ||
        PyArray_TYPE(other) != dtype ||
        !PyArray_IS_C_CONTIGUOUS(array) ||
        !PyArray_IS_C_CONTIGUOUS(other)) {
        return PYJIT_EXT_MISS;
    }
    double scalar;
    if (!exact_python_scalar_as_double(
            payload, args[scalar_slot], &scalar)) {
        return PYJIT_EXT_MISS;
    }
    npy_intp size = PyArray_SIZE(array);
    PyArrayObject *workspace = numpy_jit_get_or_alloc_workspace(
        payload, PyArray_NDIM(array), PyArray_DIMS(array), dtype, size);
    if (workspace == NULL) {
        if (PyErr_Occurred()) {
            return PYJIT_EXT_ERROR;
        }
        return PYJIT_EXT_MISS;
    }
    PyObject *matmul = numpy_jit_known_ufunc(NUMPY_JIT_OP_MATMUL);
    if (matmul == NULL) {
        return PYJIT_EXT_MISS;
    }
    if (!numpy_jit_kernel_array_scalar_contig(
            NUMPY_JIT_KERNEL_OP_MULTIPLY, dtype, PyArray_BYTES(array),
            scalar, scalar_first, PyArray_BYTES(workspace), size)) {
        return PYJIT_EXT_MISS;
    }
    PyObject *left = internal_is_left ?
        (PyObject *)workspace : (PyObject *)other;
    PyObject *right = internal_is_left ?
        (PyObject *)other : (PyObject *)workspace;
    PyObject *out = PyObject_CallFunctionObjArgs(
        matmul, left, right, NULL);
    if (out == NULL) {
        return PYJIT_EXT_ERROR;
    }
    *result = out;
    return PYJIT_EXT_HIT;
}

static PyJitExtResult
numpy_region_matmul_scaled_add_eval(NumpyJitPlanPayload *payload,
                                    PyObject *const *args, Py_ssize_t nargs,
                                    PyObject **result)
{
    if (nargs != 4) {
        return PYJIT_EXT_MISS;
    }
    int a_slot = payload->region_left[0];
    int b_slot = payload->region_right[0];
    int mul_left = payload->region_left[1];
    int mul_right = payload->region_right[1];
    if (a_slot < 0 || a_slot >= nargs || b_slot < 0 || b_slot >= nargs ||
        mul_left < 0 || mul_left >= nargs ||
        mul_right < 0 || mul_right >= nargs ||
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
    int array_slot;
    int scalar_slot;
    int scalar_first;
    if (PyArray_CheckExact(args[mul_left]) &&
        !PyArray_CheckExact(args[mul_right])) {
        array_slot = mul_left;
        scalar_slot = mul_right;
        scalar_first = 0;
    }
    else if (!PyArray_CheckExact(args[mul_left]) &&
             PyArray_CheckExact(args[mul_right])) {
        array_slot = mul_right;
        scalar_slot = mul_left;
        scalar_first = 1;
    }
    else {
        return PYJIT_EXT_MISS;
    }
    PyArrayObject *array = (PyArrayObject *)args[array_slot];
    npy_intp dims[2] = {PyArray_DIM(a, 0), PyArray_DIM(b, 1)};
    if (PyArray_TYPE(array) != dtype || PyArray_NDIM(array) != 2 ||
        PyArray_DIM(array, 0) != dims[0] ||
        PyArray_DIM(array, 1) != dims[1] ||
        !PyArray_IS_C_CONTIGUOUS(array)) {
        return PYJIT_EXT_MISS;
    }
    double scalar;
    if (!exact_python_scalar_as_double(
            payload, args[scalar_slot], &scalar)) {
        return PYJIT_EXT_MISS;
    }
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(2, dims, dtype);
    if (out == NULL) {
        return PYJIT_EXT_ERROR;
    }
    PyObject *matmul = numpy_jit_known_ufunc(NUMPY_JIT_OP_MATMUL);
    if (matmul == NULL) {
        Py_DECREF(out);
        return PYJIT_EXT_MISS;
    }
    PyObject *matmul_result = PyObject_CallFunctionObjArgs(
        matmul, (PyObject *)a, (PyObject *)b, (PyObject *)out, NULL);
    if (matmul_result == NULL) {
        Py_DECREF(out);
        return PYJIT_EXT_ERROR;
    }
    Py_DECREF(matmul_result);
    npy_intp size = dims[0] * dims[1];
    if (!numpy_jit_kernel_scaled_array_add_inplace_contig(
            dtype, PyArray_BYTES(out), PyArray_BYTES(array), scalar,
            scalar_first, size)) {
        Py_DECREF(out);
        return PYJIT_EXT_MISS;
    }
    *result = (PyObject *)out;
    return PYJIT_EXT_HIT;
}

static int
numpy_jit_llvm_supported_op(uint8_t op, int dtype)
{
    switch (op) {
        case NUMPY_JIT_KERNEL_REGION_ADD:
        case NUMPY_JIT_KERNEL_REGION_SUB:
        case NUMPY_JIT_KERNEL_REGION_MUL:
        case NUMPY_JIT_KERNEL_REGION_SQUARE:
        case NUMPY_JIT_KERNEL_REGION_NEGATIVE:
            return 1;
        case NUMPY_JIT_KERNEL_REGION_DIV:
            return dtype == NPY_FLOAT || dtype == NPY_DOUBLE;
        default:
            return 0;
    }
}

static uint64_t
numpy_jit_llvm_scalar_bits(int dtype, double value)
{
    uint64_t bits = 0;
    if (dtype == NPY_DOUBLE) {
        memcpy(&bits, &value, sizeof(value));
    }
    else if (dtype == NPY_FLOAT) {
        float narrowed = (float)value;
        uint32_t narrow_bits;
        memcpy(&narrow_bits, &narrowed, sizeof(narrowed));
        bits = narrow_bits;
    }
    else if (dtype == NPY_INT) {
        bits = (uint32_t)(npy_int32)value;
    }
    else {
        bits = (npy_uint64)(npy_int64)value;
    }
    return bits;
}

static int
numpy_jit_llvm_region_eligible(NumpyJitPlanPayload *payload, int dtype,
                               uint8_t scalar_mask, int all_c_contiguous)
{
    if (payload == NULL || !payload->region_generic || payload->inplace ||
        !all_c_contiguous || payload->region_node_count < 2 ||
        payload->region_node_count > NUMPY_JIT_LLVM_MAX_NODES ||
        payload->region_input_count == 0 ||
        payload->region_input_count > NUMPY_JIT_LLVM_MAX_INPUTS ||
        (scalar_mask & ((1u << payload->region_input_count) - 1u)) ==
            ((1u << payload->region_input_count) - 1u) ||
        !(dtype == NPY_FLOAT || dtype == NPY_DOUBLE || dtype == NPY_INT ||
          dtype == NPY_LONG || dtype == NPY_LONGLONG)) {
        return 0;
    }
    for (int i = 0; i < payload->region_node_count; i++) {
        if (!numpy_jit_llvm_supported_op(payload->region_nodes[i].op, dtype)) {
            return 0;
        }
    }
    return 1;
}

static void
numpy_jit_build_compiled_kernel_spec(const NumpyJitPlanPayload *payload,
                                     int dtype, uint8_t scalar_mask,
                                     NumpyJitLLVMKernelSpec *spec)
{
    memset(spec, 0, sizeof(*spec));
    spec->graph_hash = payload->region_graph_hash;
    spec->dialect_id = NUMPY_JIT_DIALECT_ID;
    spec->dialect_version = 1;
    spec->node_count = payload->region_node_count;
    spec->input_count = payload->region_input_count;
    spec->output_slot = payload->region_output_slot;
    spec->scalar_mask = scalar_mask;
    spec->dtype = (int16_t)dtype;
    memcpy(spec->nodes, payload->region_nodes,
           payload->region_node_count * sizeof(spec->nodes[0]));
}

static void
numpy_jit_llvm_maybe_request(NumpyJitPlanPayload *payload, int dtype,
                             uint8_t scalar_mask, npy_intp size)
{
    if (payload->region_llvm_entry != NULL &&
        numpy_jit_llvm_entry_state(payload->region_llvm_entry) ==
            NUMPY_JIT_LLVM_ORPHANED) {
        payload->region_llvm_entry = NULL;
        payload->region_llvm_decision = 0;
        payload->region_llvm_cache_hit = 0;
    }
    if (!payload->region_llvm_property_valid) {
        payload->region_llvm_property_valid = 1;
        payload->region_llvm_dtype = (int16_t)dtype;
        payload->region_llvm_scalar_mask = scalar_mask;
    }
    else if (payload->region_llvm_dtype != dtype ||
             payload->region_llvm_scalar_mask != scalar_mask) {
        /* Keep the direct native entry monomorphic.  Other properties remain
         * fully supported by the existing four-entry Region PIC. */
        return;
    }
    if (payload->region_llvm_eligible_hits != UINT32_MAX) {
        payload->region_llvm_eligible_hits++;
    }
    if (size > 0 && payload->region_llvm_visited_elements <=
                        UINT64_MAX - (uint64_t)size) {
        payload->region_llvm_visited_elements += (uint64_t)size;
    }
    if (payload->region_llvm_entry != NULL) {
        return;
    }
    int force = numpy_jit_llvm_mode() == NUMPY_JIT_LLVM_MODE_FORCE;
    if (!force &&
        ((uint64_t)payload->region_llvm_eligible_hits <
             numpy_jit_llvm_min_hits() ||
         payload->region_llvm_visited_elements <
             numpy_jit_llvm_min_visited_elements() ||
         payload->region_llvm_guard_misses > 4u +
             payload->region_llvm_eligible_hits / 8u)) {
        return;
    }
    NumpyJitLLVMKernelSpec spec;
    numpy_jit_build_compiled_kernel_spec(payload, dtype, scalar_mask, &spec);
    int cache_hit = 0;
    payload->region_llvm_entry = numpy_jit_llvm_request(&spec, &cache_hit);
    payload->region_llvm_cache_hit = cache_hit != 0;
    if (payload->region_llvm_entry == NULL) {
        payload->region_llvm_decision = 3;
    }
}

static uint64_t
numpy_jit_elapsed_ns(const struct timespec *begin,
                     const struct timespec *end)
{
    return (uint64_t)(end->tv_sec - begin->tv_sec) * UINT64_C(1000000000) +
           (uint64_t)(end->tv_nsec - begin->tv_nsec);
}

static uint64_t
numpy_jit_median8(uint64_t values[8])
{
    uint64_t sorted[8];
    memcpy(sorted, values, sizeof(sorted));
    for (int i = 1; i < 8; i++) {
        uint64_t value = sorted[i];
        int j = i - 1;
        while (j >= 0 && sorted[j] > value) {
            sorted[j + 1] = sorted[j];
            j--;
        }
        sorted[j + 1] = value;
    }
    return sorted[3] / 2u + sorted[4] / 2u +
           ((sorted[3] & 1u) && (sorted[4] & 1u));
}

static unsigned
numpy_jit_llvm_size_bucket(npy_intp size)
{
    if (size < 256) {
        return 0;
    }
    if (size < 4096) {
        return 1;
    }
    return 2;
}

/* Property keys must distinguish the sizes at which a Region crosses over
 * from its atomic sequence.  This deliberately has one more large bucket
 * than the LLVM controller: Region admission is independent of LLVM. */
static unsigned
numpy_jit_region_size_bucket(npy_intp size)
{
    if (size < 256) {
        return 0;
    }
    if (size < 4096) {
        return 1;
    }
    if (size < 65536) {
        return 2;
    }
    return 3;
}

static NumpyJitKernelFunc
numpy_jit_precompiled_variant_produce(NumpyJitPlanPayload *payload, int dtype,
                                      uint8_t scalar_mask, npy_intp size)
{
    if (numpy_jit_aot_mode() == NUMPY_JIT_LLVM_MODE_OFF) {
        return NULL;
    }
    if (!payload->region_aot_lookup_done) {
        NumpyJitLLVMKernelSpec spec;
        struct timespec begin, end;
        int timed = clock_gettime(CLOCK_MONOTONIC, &begin) == 0;
        numpy_jit_build_compiled_kernel_spec(payload, dtype, scalar_mask, &spec);
        payload->region_aot_entry = numpy_jit_aot_lookup(&spec);
        if (timed && clock_gettime(CLOCK_MONOTONIC, &end) == 0) {
            payload->region_aot_lookup_ns = numpy_jit_elapsed_ns(&begin, &end);
        }
        payload->region_aot_lookup_done = 1;
        if (payload->region_aot_entry != NULL) {
            payload->region_aot_lookup_hit++;
        }
        else {
            payload->region_aot_lookup_miss++;
        }
    }
    if (payload->region_aot_entry == NULL) {
        return NULL;
    }
    NumpyJitAOTEntryInfo info;
    numpy_jit_aot_entry_info(payload->region_aot_entry, &info);
    if ((info.accepted_bucket_mask & (1u << numpy_jit_llvm_size_bucket(size))) == 0) {
        return NULL;
    }
    return numpy_jit_aot_entry_function(payload->region_aot_entry);
}

static int
numpy_jit_llvm_select_native(NumpyJitPlanPayload *payload,
                             NumpyJitKernelFunc *function,
                             int *measure_probation, npy_intp size)
{
    *function = NULL;
    *measure_probation = 0;
    if (payload->region_llvm_entry == NULL ||
        payload->region_llvm_decision == 3) {
        return 0;
    }
    int state = numpy_jit_llvm_entry_state(payload->region_llvm_entry);
    if (state == NUMPY_JIT_LLVM_FAILED ||
        state == NUMPY_JIT_LLVM_CAPACITY) {
        payload->region_llvm_decision = 3;
        return 0;
    }
    if (state == NUMPY_JIT_LLVM_ORPHANED) {
        return 0;
    }
    if (state != NUMPY_JIT_LLVM_READY) {
        return 0;
    }
    *function = numpy_jit_llvm_entry_function(payload->region_llvm_entry);
    if (*function == NULL) {
        return 0;
    }
    unsigned bucket = numpy_jit_llvm_size_bucket(size);
    if (numpy_jit_llvm_mode() == NUMPY_JIT_LLVM_MODE_FORCE) {
        payload->region_llvm_bucket_decision[bucket] = 2;
    }
    if (payload->region_llvm_bucket_decision[bucket] == 2) {
        return 1;
    }
    if (payload->region_llvm_bucket_decision[bucket] == 3) {
        return 0;
    }
    payload->region_llvm_bucket_decision[bucket] = 1;
    *measure_probation = 1;
    /* Alternate real executions.  Start with native, then use the old path,
     * so neither candidate receives a systematically colder cache state. */
    return payload->region_llvm_native_count[bucket] <=
           payload->region_llvm_old_count[bucket];
}

/* AOT and LLVM are sibling candidate producers.  Keeping this decision in
 * one place prevents the old "AOT layer, else LLVM layer" execution model
 * from leaking into the Region evaluator. */
static int
numpy_jit_variant_selector(NumpyJitPlanPayload *payload, int dtype,
                           uint8_t scalar_mask, npy_intp size,
                           NumpyJitVariant *selected)
{
    memset(selected, 0, sizeof(*selected));
    NumpyJitKernelFunc function = numpy_jit_precompiled_variant_produce(
        payload, dtype, scalar_mask, size);
    if (function != NULL) {
        selected->kind = NUMPY_JIT_VARIANT_PRECOMPILED;
        selected->function = function;
        return 1;
    }
    int measure_probation = 0;
    if (!numpy_jit_llvm_select_native(
            payload, &function, &measure_probation, size)) {
        return 0;
    }
    selected->kind = NUMPY_JIT_VARIANT_LLVM;
    selected->function = function;
    selected->measure_probation = (uint8_t)measure_probation;
    return 1;
}

static void
numpy_jit_llvm_record_probation(NumpyJitPlanPayload *payload,
                                int used_native, uint64_t elapsed_ns,
                                npy_intp size)
{
    if (payload->region_llvm_decision == 3 || size <= 0) {
        return;
    }
    unsigned bucket = numpy_jit_llvm_size_bucket(size);
    if (payload->region_llvm_bucket_decision[bucket] != 1) {
        return;
    }
    uint64_t normalized = elapsed_ns <= UINT64_MAX / 1024u ?
        elapsed_ns * 1024u / (uint64_t)size : elapsed_ns / (uint64_t)size;
    if (used_native && payload->region_llvm_native_count[bucket] < 8) {
        payload->region_llvm_native_samples[bucket]
            [payload->region_llvm_native_count[bucket]++] = normalized;
    }
    else if (!used_native && payload->region_llvm_old_count[bucket] < 8) {
        payload->region_llvm_old_samples[bucket]
            [payload->region_llvm_old_count[bucket]++] = normalized;
    }
    if (payload->region_llvm_native_count[bucket] == 8 &&
        payload->region_llvm_old_count[bucket] == 8) {
        uint64_t native_median = numpy_jit_median8(
            payload->region_llvm_native_samples[bucket]);
        uint64_t old_median = numpy_jit_median8(
            payload->region_llvm_old_samples[bucket]);
        payload->region_llvm_bucket_decision[bucket] =
            native_median <= old_median * 97u / 100u ? 2 : 3;
    }
}

typedef struct {
    NumpyJitKernelArgs kernel_args;
    PyArrayObject *shape_source;
    int dtype;
    int errmask;
    uint8_t scalar_mask;
} NumpyJitLLVMDirectCall;

static void
numpy_jit_llvm_record_guard_miss(NumpyJitPlanPayload *payload)
{
    if (payload->region_llvm_guard_misses != UINT32_MAX) {
        payload->region_llvm_guard_misses++;
    }
    if (!payload->region_llvm_property_valid &&
        payload->region_llvm_entry == NULL &&
        payload->region_llvm_guard_misses >= 8) {
        /* Stable sites outside the first-phase C-contiguous subset must not
         * pay a permanent argument scan merely because LLVM is enabled. */
        payload->region_llvm_decision = 3;
    }
}

/* Build the complete, address-independent native guard.  This helper is
 * deliberately shared by generic Regions and structural schedules: the
 * selected backend is an implementation detail of one Region DAG. */
static int
numpy_jit_llvm_prepare_direct(NumpyJitPlanPayload *payload,
                              PyObject *const *args, Py_ssize_t nargs,
                              NumpyJitLLVMDirectCall *call,
                              int allow_producer_update)
{
    memset(call, 0, sizeof(*call));
    if ((numpy_jit_llvm_mode() == NUMPY_JIT_LLVM_MODE_OFF &&
         numpy_jit_aot_mode() == NUMPY_JIT_LLVM_MODE_OFF) ||
        payload == NULL || !payload->region_generic || payload->inplace ||
        nargs != payload->region_input_count) {
        return 0;
    }
    npy_intp size = -1;
    for (Py_ssize_t i = 0; i < nargs; i++) {
        if (PyArray_CheckExact(args[i])) {
            PyArrayObject *array = (PyArrayObject *)args[i];
            int dtype = PyArray_TYPE(array);
            if (!PyArray_IS_C_CONTIGUOUS(array) ||
                !numpy_jit_is_supported_region_dtype(dtype)) {
                numpy_jit_llvm_record_guard_miss(payload);
                return 0;
            }
            if (call->shape_source == NULL) {
                call->shape_source = array;
                call->dtype = dtype;
                size = PyArray_SIZE(array);
            }
            else if (dtype != call->dtype ||
                     PyArray_NDIM(array) !=
                         PyArray_NDIM(call->shape_source)) {
                numpy_jit_llvm_record_guard_miss(payload);
                return 0;
            }
            else {
                for (int dim = 0; dim < PyArray_NDIM(array); dim++) {
                    if (PyArray_DIM(array, dim) !=
                        PyArray_DIM(call->shape_source, dim)) {
                        numpy_jit_llvm_record_guard_miss(payload);
                        return 0;
                    }
                }
            }
            call->kernel_args.data[i] = PyArray_BYTES(array);
        }
        else {
            double scalar;
            if (!exact_python_scalar_as_double(payload, args[i], &scalar)) {
                numpy_jit_llvm_record_guard_miss(payload);
                return 0;
            }
            call->scalar_mask |= (uint8_t)(1u << i);
        }
    }
    if (call->shape_source == NULL || size < payload->min_elements ||
        !numpy_jit_llvm_region_eligible(
            payload, call->dtype, call->scalar_mask, 1)) {
        numpy_jit_llvm_record_guard_miss(payload);
        return 0;
    }
    /* Scalar slots may precede the first array, so encode them only after
     * the array dtype has established the specialization. */
    for (Py_ssize_t i = 0; i < nargs; i++) {
        if ((call->scalar_mask & (1u << i)) != 0) {
            if (call->dtype == NPY_INT || call->dtype == NPY_LONG ||
                call->dtype == NPY_LONGLONG) {
                if (!(PyLong_CheckExact(args[i]) ||
                      PyArray_IsScalar(args[i], SignedInteger))) {
                    numpy_jit_llvm_record_guard_miss(payload);
                    return 0;
                }
                npy_int64 integer = (npy_int64)PyLong_AsLongLong(args[i]);
                if (integer == -1 && PyErr_Occurred()) {
                    PyErr_Clear();
                    numpy_jit_llvm_record_guard_miss(payload);
                    return 0;
                }
                if (call->dtype == NPY_INT &&
                    (integer < NPY_MIN_INT32 || integer > NPY_MAX_INT32)) {
                    numpy_jit_llvm_record_guard_miss(payload);
                    return 0;
                }
                call->kernel_args.scalars[i] = (npy_uint64)integer;
                continue;
            }
            double scalar;
            if (!exact_python_scalar_as_double(payload, args[i], &scalar)) {
                numpy_jit_llvm_record_guard_miss(payload);
                return 0;
            }
            call->kernel_args.scalars[i] = numpy_jit_llvm_scalar_bits(
                call->dtype, scalar);
        }
    }
    if (payload->region_llvm_property_valid &&
        (payload->region_llvm_dtype != call->dtype ||
         payload->region_llvm_scalar_mask != call->scalar_mask)) {
        numpy_jit_llvm_record_guard_miss(payload);
        return 0;
    }
    if (_get_errmask_fast(&call->errmask) < 0) {
        return -1;
    }
    for (int shift = 0; shift <= 9; shift += 3) {
        if (((call->errmask >> shift) & 7) > 1) {
            numpy_jit_llvm_record_guard_miss(payload);
            return 0;
        }
    }
    call->kernel_args.size = size;
    /* Static stencils are resolved on the first guarded invocation.  Only an
     * AOT miss (or a size bucket deliberately outside its manifest) enters
     * the asynchronous LLVM tier. */
    if (allow_producer_update &&
        numpy_jit_precompiled_variant_produce(
            payload, call->dtype, call->scalar_mask, size) == NULL &&
        /* An AOT manifest match with this bucket excluded is an explicit
         * admission decision, not a reason to route the same graph into the
         * experimental LLVM tier.  In particular this avoids comparing LLVM
         * against an old path inflated by direct-call preparation. */
        payload->region_aot_entry == NULL) {
        numpy_jit_llvm_maybe_request(
            payload, call->dtype, call->scalar_mask, size);
    }
    return 1;
}

static PyJitExtResult
numpy_jit_compiled_execute_direct(NumpyJitPlanPayload *payload,
                              NumpyJitLLVMDirectCall *call,
                              NumpyJitKernelFunc function,
                              PyObject **result, int is_aot)
{
    PyArrayObject *out = numpy_jit_get_or_alloc_result(
        payload, PyArray_NDIM(call->shape_source),
        PyArray_DIMS(call->shape_source), call->dtype,
        call->kernel_args.size);
    if (out == NULL) {
        return PYJIT_EXT_ERROR;
    }
    char fp_barrier;
    if (call->errmask != 0) {
        npy_clear_floatstatus_barrier(&fp_barrier);
    }
    call->kernel_args.output = PyArray_BYTES(out);
    function(&call->kernel_args);
    *result = (PyObject *)out;
    if (call->errmask != 0) {
        int fpe_errors = npy_get_floatstatus_barrier(&fp_barrier);
        if (fpe_errors != 0 &&
            PyUFunc_GiveFloatingpointErrors(
                "CrossJIT LLVM region", fpe_errors) < 0) {
            Py_CLEAR(*result);
            return PYJIT_EXT_ERROR;
        }
    }
    unsigned bucket = numpy_jit_llvm_size_bucket(call->kernel_args.size);
    uint32_t *uses = is_aot ? &payload->region_aot_native_uses[bucket] :
                              &payload->region_llvm_native_uses[bucket];
    if (*uses != UINT32_MAX) {
        (*uses)++;
    }
    return PYJIT_EXT_HIT;
}

/* Stable native execution is owned by the same PropertyKey entry as atomic
 * and Region candidates.  The producer/cache lookup is deliberately disabled
 * here: a committed hit performs only the specialized direct-call guard and
 * invokes the function already selected for this entry. */
static PyJitExtResult
numpy_jit_region_execute_native(NumpyJitPlanPayload *payload,
                                NumpyJitDispatchEntry *entry,
                                PyObject *const *args, Py_ssize_t nargs,
                                PyObject **result)
{
    if (entry == NULL || entry->selected_function == NULL) {
        return PYJIT_EXT_MISS;
    }
    NumpyJitLLVMDirectCall call;
    int prepared = numpy_jit_llvm_prepare_direct(
        payload, args, nargs, &call, 0);
    if (prepared < 0) {
        return PYJIT_EXT_ERROR;
    }
    if (!prepared) {
        return PYJIT_EXT_MISS;
    }
    return numpy_jit_compiled_execute_direct(
        payload, &call, entry->selected_function, result,
        entry->selected_variant == NUMPY_JIT_VARIANT_PRECOMPILED);
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
            int layout = numpy_jit_region_layout_class(array);
            int layout4_fused_supported = !payload->region_generic &&
                (payload->fused_op == NUMPY_JIT_FUSED_ADD_ARRAY_MUL_SCALAR ||
                 payload->fused_op == NUMPY_JIT_FUSED_SUB_ARRAY_MUL_SCALAR ||
                 payload->fused_op == NUMPY_JIT_FUSED_ADD_ARRAY_DIV_SCALAR ||
                 payload->fused_op == NUMPY_JIT_FUSED_SUB_ARRAY_DIV_SCALAR ||
                 payload->fused_op == NUMPY_JIT_FUSED_ADD_THREE_ARRAYS);
            if (!numpy_jit_is_supported_region_dtype(current_dtype) ||
                layout == 0 ||
                (layout == 4 && payload->region_compiled_kind != 5 &&
                 !layout4_fused_supported)) {
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
    if (dtype == NPY_INT || dtype == NPY_LONG || dtype == NPY_LONGLONG) {
        for (int node = 0; node < payload->region_node_count; node++) {
            if (payload->region_nodes[node].op ==
                    NUMPY_JIT_KERNEL_REGION_DIV) {
                /* Integer true_divide has a floating result, which Core IR
                   v1's single-dtype Region does not encode. */
                return PYJIT_EXT_MISS;
            }
        }
        for (Py_ssize_t i = 0; i < nargs; i++) {
            if (input_is_scalar[i] &&
                !(PyLong_CheckExact(args[i]) ||
                  PyArray_IsScalar(args[i], SignedInteger))) {
                return PYJIT_EXT_MISS;
            }
        }
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
    struct timespec region_t0, region_t1;
    int measure_region = numpy_jit_stats_enabled() &&
        !payload->region_llvm_probation_old_active &&
        clock_gettime(CLOCK_MONOTONIC, &region_t0) == 0;
    if (payload->region_compiled_kind == 1 && all_c_contiguous &&
        PyArray_IS_C_CONTIGUOUS(out)) {
        NumpyJitKernelRegionNode *mul0 = &payload->region_nodes[0];
        NumpyJitKernelRegionNode *mul1 = &payload->region_nodes[1];
        NumpyJitKernelRegionNode *tail = &payload->region_nodes[3];
        int a_slot = input_is_scalar[mul0->left] ?
            mul0->right : mul0->left;
        int alpha_slot = input_is_scalar[mul0->left] ?
            mul0->left : mul0->right;
        int b_slot = input_is_scalar[mul1->left] ?
            mul1->right : mul1->left;
        int beta_slot = input_is_scalar[mul1->left] ?
            mul1->left : mul1->right;
        int sum_slot = payload->region_input_count + 2;
        int gamma_slot = tail->left == sum_slot ? tail->right : tail->left;
        int roles_ok = a_slot < nargs && alpha_slot < nargs &&
            b_slot < nargs && beta_slot < nargs && gamma_slot < nargs &&
            !input_is_scalar[a_slot] && input_is_scalar[alpha_slot] &&
            !input_is_scalar[b_slot] && input_is_scalar[beta_slot] &&
            input_is_scalar[gamma_slot];
        kernel_ok = roles_ok && numpy_jit_kernel_axpby_plus_scalar_contig(
            dtype, input_data[a_slot], scalar_values[alpha_slot],
            input_data[b_slot], scalar_values[beta_slot],
            scalar_values[gamma_slot], PyArray_BYTES(out), size);
    }
    else if (payload->region_compiled_kind == 2 && all_c_contiguous &&
             PyArray_IS_C_CONTIGUOUS(out)) {
        NumpyJitKernelRegionNode *mul = &payload->region_nodes[0];
        NumpyJitKernelRegionNode *add0 = &payload->region_nodes[1];
        NumpyJitKernelRegionNode *add1 = &payload->region_nodes[2];
        NumpyJitKernelRegionNode *add2 = &payload->region_nodes[3];
        NumpyJitKernelRegionNode *sub = &payload->region_nodes[4];
        int a_slot = input_is_scalar[mul->left] ? mul->right : mul->left;
        int scale_slot = input_is_scalar[mul->left] ? mul->left : mul->right;
        int b_slot = add0->left;
        int c_slot = add0->right;
        int t1 = payload->region_input_count + 1;
        int t2 = t1 + 1;
        int d_slot = add1->left == t1 ? add1->right : add1->left;
        int e_slot = add2->left == t2 ? add2->right : add2->left;
        int t0 = payload->region_input_count;
        int t3 = payload->region_input_count + 3;
        int scaled_first = sub->left == t0 && sub->right == t3;
        int roles_ok = a_slot < nargs && scale_slot < nargs &&
            b_slot < nargs && c_slot < nargs && d_slot < nargs &&
            e_slot < nargs && !input_is_scalar[a_slot] &&
            input_is_scalar[scale_slot] && !input_is_scalar[b_slot] &&
            !input_is_scalar[c_slot] && !input_is_scalar[d_slot] &&
            !input_is_scalar[e_slot] &&
            ((sub->left == t0 && sub->right == t3) ||
             (sub->right == t0 && sub->left == t3));
        kernel_ok = roles_ok && numpy_jit_kernel_scale_sub_sum4_contig(
            dtype, input_data[a_slot], scalar_values[scale_slot],
            input_data[b_slot], input_data[c_slot], input_data[d_slot],
            input_data[e_slot], scaled_first, PyArray_BYTES(out), size);
    }
    else if (payload->region_compiled_kind == 3 && all_c_contiguous &&
             PyArray_IS_C_CONTIGUOUS(out)) {
        NumpyJitKernelRegionNode *square = &payload->region_nodes[0];
        NumpyJitKernelRegionNode *tail = &payload->region_nodes[1];
        int t0 = payload->region_input_count;
        int array_slot = square->left;
        int exponent_slot = square->right;
        int other_slot = tail->left == t0 ? tail->right : tail->left;
        int internal_first = tail->left == t0;
        int roles_ok = array_slot < nargs && exponent_slot < nargs &&
            other_slot < nargs && !input_is_scalar[array_slot] &&
            input_is_scalar[exponent_slot];
        kernel_ok = roles_ok && numpy_jit_kernel_square_addsub_contig(
            dtype, input_data[array_slot], input_data[other_slot],
            scalar_values[other_slot], input_is_scalar[other_slot],
            internal_first,
            tail->op == NUMPY_JIT_KERNEL_REGION_SUB,
            PyArray_BYTES(out), size);
    }
    else if (payload->region_compiled_kind == 3) {
        kernel_ok = 0;
    }
    else if (payload->region_compiled_kind == 4 && all_c_contiguous &&
             PyArray_IS_C_CONTIGUOUS(out)) {
        int array_slots[3] = {-1, -1, -1};
        int exponent_slots[3] = {-1, -1, -1};
        int square_nodes[3] = {0, 1, 3};
        int roles_ok = 1;
        for (int i = 0; i < 3; i++) {
            NumpyJitKernelRegionNode *square =
                &payload->region_nodes[square_nodes[i]];
            array_slots[i] = square->left;
            exponent_slots[i] = square->right;
            roles_ok &= array_slots[i] < nargs &&
                        exponent_slots[i] < nargs &&
                        !input_is_scalar[array_slots[i]] &&
                        input_is_scalar[exponent_slots[i]];
        }
        kernel_ok = roles_ok && numpy_jit_kernel_sum3_squares_contig(
            dtype, input_data[array_slots[0]], input_data[array_slots[1]],
            input_data[array_slots[2]], PyArray_BYTES(out), size);
    }
    else if (payload->region_compiled_kind == 4) {
        kernel_ok = 0;
    }
    else if (payload->region_compiled_kind == 7) {
        NumpyJitKernelRegionNode *first = &payload->region_nodes[0];
        NumpyJitKernelRegionNode *tail = &payload->region_nodes[1];
        int t0 = payload->region_input_count;
        int third = tail->left == t0 ? tail->right : tail->left;
        int roles_ok = first->left < nargs && first->right < nargs &&
            third < nargs && !input_is_scalar[first->left] &&
            !input_is_scalar[first->right] && !input_is_scalar[third];
        int order = tail->left == t0 ? 0 : 2;
        kernel_ok = roles_ok &&
            ((all_c_contiguous && PyArray_IS_C_CONTIGUOUS(out)) ?
             numpy_jit_kernel_add3_contig(
                 dtype, input_data[first->left], input_data[first->right],
                 input_data[third], PyArray_BYTES(out), size, order) :
             numpy_jit_kernel_add3_strided(
                 dtype, PyArray_NDIM(shape_source), PyArray_DIMS(shape_source),
                 (const char *)input_data[first->left],
                 input_strides[first->left],
                 (const char *)input_data[first->right],
                 input_strides[first->right], (const char *)input_data[third],
                 input_strides[third], PyArray_BYTES(out), size, order));
        if (!roles_ok) {
            /* The structural schedule is selected before scalar/array roles
             * are known.  Preserve the generic Region fallback for scalar
             * variants instead of turning a schedule mismatch into a miss. */
            kernel_ok = all_c_contiguous && PyArray_IS_C_CONTIGUOUS(out) ?
                numpy_jit_kernel_region_contig(
                    payload->region_nodes, payload->region_node_count,
                    payload->region_input_count, payload->region_output_slot,
                    dtype, input_data, scalar_values, input_is_scalar,
                    PyArray_BYTES(out), size) :
                numpy_jit_kernel_region_strided(
                    payload->region_nodes, payload->region_node_count,
                    payload->region_input_count, payload->region_output_slot,
                    dtype, PyArray_NDIM(shape_source), PyArray_DIMS(shape_source),
                    input_data, input_strides, scalar_values, input_is_scalar,
                    PyArray_BYTES(out), PyArray_STRIDES(out), size);
        }
    }
    else if ((payload->region_compiled_kind == 8 ||
              payload->region_compiled_kind == 9) &&
             size >= NUMPY_JIT_MUL_ADDSUB_REGION_MIN_ELEMENTS) {
        NumpyJitKernelRegionNode *first = &payload->region_nodes[0];
        NumpyJitKernelRegionNode *tail = &payload->region_nodes[1];
        int cslot = tail->right;
        int roles_ok = first->left < nargs && first->right < nargs &&
            cslot < nargs && !input_is_scalar[first->left] &&
            !input_is_scalar[first->right] && !input_is_scalar[cslot];
        kernel_ok = roles_ok && numpy_jit_kernel_mul_addsub3(
            dtype, payload->region_compiled_kind == 9,
            PyArray_NDIM(shape_source), PyArray_DIMS(shape_source),
            (const char *)input_data[first->left], input_strides[first->left],
            (const char *)input_data[first->right], input_strides[first->right],
            (const char *)input_data[cslot], input_strides[cslot],
            PyArray_BYTES(out), PyArray_STRIDES(out), size,
            all_c_contiguous && PyArray_IS_C_CONTIGUOUS(out));
        if (!roles_ok) {
            kernel_ok = all_c_contiguous && PyArray_IS_C_CONTIGUOUS(out) ?
                numpy_jit_kernel_region_contig(
                payload->region_nodes, payload->region_node_count,
                payload->region_input_count, payload->region_output_slot, dtype,
                input_data, scalar_values, input_is_scalar,
                PyArray_BYTES(out), size) :
                numpy_jit_kernel_region_strided(
                payload->region_nodes, payload->region_node_count,
                payload->region_input_count, payload->region_output_slot, dtype,
                PyArray_NDIM(shape_source), PyArray_DIMS(shape_source),
                input_data, input_strides, scalar_values, input_is_scalar,
                PyArray_BYTES(out), PyArray_STRIDES(out), size);
        }
    }
    else if (payload->region_compiled_kind == 5) {
        NumpyJitKernelRegionNode *first = &payload->region_nodes[0];
        NumpyJitKernelRegionNode *tail = &payload->region_nodes[1];
        int a_slot = first->left;
        int b_slot = first->right;
        int internal_slot = payload->region_input_count;
        int scalar_slot = tail->left == internal_slot ?
            tail->right : tail->left;
        int roles_ok = a_slot < nargs && b_slot < nargs &&
            scalar_slot < nargs && !input_is_scalar[a_slot] &&
            !input_is_scalar[b_slot] && input_is_scalar[scalar_slot] &&
            (tail->left == internal_slot ||
             (tail->op == NUMPY_JIT_KERNEL_REGION_MUL &&
              tail->right == internal_slot));
        int fused_op = first->op == NUMPY_JIT_KERNEL_REGION_ADD ?
            (tail->op == NUMPY_JIT_KERNEL_REGION_MUL ?
                NUMPY_JIT_FUSED_ADD_ARRAY_MUL_SCALAR :
                NUMPY_JIT_FUSED_ADD_ARRAY_DIV_SCALAR) :
            (tail->op == NUMPY_JIT_KERNEL_REGION_MUL ?
                NUMPY_JIT_FUSED_SUB_ARRAY_MUL_SCALAR :
                NUMPY_JIT_FUSED_SUB_ARRAY_DIV_SCALAR);
        kernel_ok = roles_ok && numpy_jit_kernel_fused_contig_or_strided(
            fused_op, dtype, !all_c_contiguous,
            PyArray_NDIM(shape_source), size, PyArray_DIMS(shape_source),
            input_strides[a_slot], input_strides[b_slot],
            (const char *)input_data[a_slot],
            (const char *)input_data[b_slot], scalar_values[scalar_slot],
            0.0, PyArray_BYTES(out));
    }
    else if (payload->region_compiled_kind == 6 && all_c_contiguous &&
             PyArray_IS_C_CONTIGUOUS(out)) {
        NumpyJitKernelRegionNode *first = &payload->region_nodes[0];
        NumpyJitKernelRegionNode *tail = &payload->region_nodes[1];
        int t0 = payload->region_input_count;
        int scalar_slot = tail->left == t0 ? tail->right : tail->left;
        int roles_ok = first->left < nargs && first->right < nargs &&
            scalar_slot < nargs && !input_is_scalar[first->left] &&
            !input_is_scalar[first->right] && input_is_scalar[scalar_slot];
        kernel_ok = roles_ok && numpy_jit_kernel_mul_arrays_scalar_contig(
            dtype, input_data[first->left], input_data[first->right],
            scalar_values[scalar_slot], tail->right == t0,
            PyArray_BYTES(out), size);
    }
    else if (payload->region_compiled_kind == 6) {
        kernel_ok = 0;
    }
    else if (all_c_contiguous && PyArray_IS_C_CONTIGUOUS(out)) {
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
    if (measure_region && clock_gettime(CLOCK_MONOTONIC, &region_t1) == 0) {
        uint64_t elapsed = (uint64_t)(region_t1.tv_sec - region_t0.tv_sec) *
            UINT64_C(1000000000) + (uint64_t)(region_t1.tv_nsec - region_t0.tv_nsec);
        const char *dir = getenv("PYTHON_JIT_EXT_STATS_DIR");
        if (dir == NULL || dir[0] == '\0') dir = "/tmp";
        char path[4096];
        snprintf(path, sizeof(path), "%s/crossjit_numpy_schedule_timing.ndjson", dir);
        FILE *fp = fopen(path, "a");
        if (fp != NULL) {
            fprintf(fp, "{\"schedule_id\":%u,\"nodes\":%u,\"elements\":%" PRIdPTR ",\"elapsed_ns\":%" PRIu64 ",\"kernel_ok\":%d}\n",
                    (unsigned)payload->region_compiled_kind,
                    (unsigned)payload->region_node_count, size, elapsed, kernel_ok);
            fclose(fp);
        }
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
    Py_ssize_t specialized_nargs =
        payload->region_matmul_epilogue == 2 ? 4 : 3;
    if ((!payload->region_generic && nargs != specialized_nargs) ||
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
       ignore and warn are safe for direct fused execution.  Raise/call/log
       stay on the exact local recipe; warn is reported through NumPy's
       standard floating-point error handler after the fused kernel. */
    for (int shift = 0; shift <= 9; shift += 3) {
        int mode = (errmask >> shift) & 7;
        if (mode > 1) {
            return PYJIT_EXT_MISS;
        }
    }
    char fp_barrier;
    if (errmask != 0) {
        npy_clear_floatstatus_barrier(&fp_barrier);
    }
    PyJitExtResult status;
    if (payload->region_matmul_epilogue == 2) {
        status = numpy_region_matmul_scaled_add_eval(
            payload, args, nargs, result);
    }
    else if (payload->region_matmul_epilogue == 3) {
        status = numpy_region_matmul_prologue_eval(
            payload, args, nargs, result);
    }
    else if (payload->region_matmul_epilogue) {
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
    if ((status == PYJIT_EXT_HIT || status == PYJIT_EXT_COLD) &&
        errmask != 0) {
        int fpe_errors = npy_get_floatstatus_barrier(&fp_barrier);
        if (fpe_errors != 0 &&
            PyUFunc_GiveFloatingpointErrors(
                "CrossJIT region", fpe_errors) < 0) {
            Py_CLEAR(*result);
            return PYJIT_EXT_ERROR;
        }
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
        if (ndim == 2) {
            return 4;
        }
    }
    return 0;
}

static int
numpy_jit_matmul_prologue_inspect(
        NumpyJitPlanPayload *payload, PyObject *const *args,
        Py_ssize_t nargs, int require_lowerable, uint64_t *signature)
{
    if (nargs != 3) {
        return 0;
    }
    uint64_t hash = 1469598103934665603ULL;
    hash = numpy_jit_region_hash_u64(hash, 0x4d415450524fULL);
    for (Py_ssize_t i = 0; i < nargs; i++) {
        hash = numpy_jit_region_hash_u64(
            hash, (uint64_t)(uintptr_t)Py_TYPE(args[i]));
        if (PyArray_CheckExact(args[i])) {
            PyArrayObject *current = (PyArrayObject *)args[i];
            hash = numpy_jit_region_hash_u64(
                hash, (uint64_t)PyArray_TYPE(current));
            hash = numpy_jit_region_hash_u64(
                hash, (uint64_t)PyArray_NDIM(current));
            hash = numpy_jit_region_hash_u64(
                hash, (uint64_t)numpy_jit_region_layout_class(current));
        }
    }
    int mul_left = payload->region_left[0];
    int mul_right = payload->region_right[0];
    int internal_slot = payload->region_input_count;
    int matmul_left = payload->region_left[1];
    int matmul_right = payload->region_right[1];
    int internal_is_left = matmul_left == internal_slot;
    int other_slot = internal_is_left ? matmul_right : matmul_left;
    int array_slot = -1;
    int scalar_slot = -1;
    if (mul_left >= 0 && mul_left < nargs &&
        mul_right >= 0 && mul_right < nargs) {
        if (PyArray_CheckExact(args[mul_left]) &&
            !PyArray_CheckExact(args[mul_right])) {
            array_slot = mul_left;
            scalar_slot = mul_right;
        }
        else if (!PyArray_CheckExact(args[mul_left]) &&
                 PyArray_CheckExact(args[mul_right])) {
            array_slot = mul_right;
            scalar_slot = mul_left;
        }
    }
    int supported = array_slot >= 0 && other_slot >= 0 &&
                    other_slot < nargs &&
                    (internal_is_left != (matmul_right == internal_slot)) &&
                    PyArray_CheckExact(args[other_slot]);
    npy_intp elements = 0;
    if (supported) {
        PyArrayObject *array = (PyArrayObject *)args[array_slot];
        PyArrayObject *other = (PyArrayObject *)args[other_slot];
        int dtype = PyArray_TYPE(array);
        int array_ndim = PyArray_NDIM(array);
        int other_ndim = PyArray_NDIM(other);
        supported = numpy_jit_is_supported_float_dtype(dtype) &&
                    PyArray_TYPE(other) == dtype &&
                    (array_ndim == 1 || array_ndim == 2) &&
                    (other_ndim == 1 || other_ndim == 2) &&
                    PyArray_IS_C_CONTIGUOUS(array) &&
                    PyArray_IS_C_CONTIGUOUS(other);
        if (supported) {
            npy_intp array_contract =
                internal_is_left ? PyArray_DIM(array, array_ndim - 1) :
                                   PyArray_DIM(array, 0);
            npy_intp other_contract =
                internal_is_left ? PyArray_DIM(other, 0) :
                                   PyArray_DIM(other, other_ndim - 1);
            supported = array_contract == other_contract;
            elements = PyArray_SIZE(array);
        }
        double ignored;
        if (supported && !exact_python_scalar_as_double(
                payload, args[scalar_slot], &ignored)) {
            PyErr_Clear();
            supported = 0;
        }
        uint64_t bytes = supported ?
            (uint64_t)elements * (uint64_t)PyArray_ITEMSIZE(array) : 0;
        if (bytes > 64ull * 1024ull * 1024ull) {
            supported = 0;
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
    return !require_lowerable || (supported && elements >= 64);
}

static int
numpy_jit_matmul_scaled_add_inspect(
        NumpyJitPlanPayload *payload, PyObject *const *args,
        Py_ssize_t nargs, int require_lowerable, uint64_t *signature)
{
    if (nargs != 4) {
        return 0;
    }
    uint64_t hash = 1469598103934665603ULL;
    hash = numpy_jit_region_hash_u64(hash, 0x4d4d53414444ULL);
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
    }
    int a_slot = payload->region_left[0];
    int b_slot = payload->region_right[0];
    int mul_left = payload->region_left[1];
    int mul_right = payload->region_right[1];
    if (a_slot < 0 || a_slot >= nargs || b_slot < 0 || b_slot >= nargs ||
        mul_left < 0 || mul_left >= nargs ||
        mul_right < 0 || mul_right >= nargs ||
        !PyArray_CheckExact(args[a_slot]) ||
        !PyArray_CheckExact(args[b_slot])) {
        supported = 0;
    }
    npy_intp output_elements = 0;
    if (supported) {
        PyArrayObject *a = (PyArrayObject *)args[a_slot];
        PyArrayObject *b = (PyArrayObject *)args[b_slot];
        int dtype = PyArray_TYPE(a);
        supported = PyArray_NDIM(a) == 2 && PyArray_NDIM(b) == 2 &&
                    dtype == PyArray_TYPE(b) &&
                    numpy_jit_is_supported_float_dtype(dtype) &&
                    PyArray_IS_C_CONTIGUOUS(a) &&
                    PyArray_IS_C_CONTIGUOUS(b) &&
                    PyArray_DIM(a, 1) == PyArray_DIM(b, 0);
        if (supported) {
            int array_slot;
            int scalar_slot;
            if (PyArray_CheckExact(args[mul_left]) &&
                !PyArray_CheckExact(args[mul_right])) {
                array_slot = mul_left;
                scalar_slot = mul_right;
            }
            else if (!PyArray_CheckExact(args[mul_left]) &&
                     PyArray_CheckExact(args[mul_right])) {
                array_slot = mul_right;
                scalar_slot = mul_left;
            }
            else {
                supported = 0;
                array_slot = 0;
                scalar_slot = 0;
            }
            if (supported) {
                PyArrayObject *array = (PyArrayObject *)args[array_slot];
                npy_intp m = PyArray_DIM(a, 0);
                npy_intp n = PyArray_DIM(b, 1);
                supported = PyArray_TYPE(array) == dtype &&
                            PyArray_NDIM(array) == 2 &&
                            PyArray_DIM(array, 0) == m &&
                            PyArray_DIM(array, 1) == n &&
                            PyArray_IS_C_CONTIGUOUS(array);
                double ignored;
                if (!exact_python_scalar_as_double(
                        payload, args[scalar_slot], &ignored)) {
                    PyErr_Clear();
                    supported = 0;
                }
                output_elements = m * n;
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
    return supported && output_elements >= 64;
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
    if (payload->region_matmul_epilogue == 2) {
        return numpy_jit_matmul_scaled_add_inspect(
            payload, args, nargs, require_lowerable, signature);
    }
    if (payload->region_matmul_epilogue == 3) {
        return numpy_jit_matmul_prologue_inspect(
            payload, args, nargs, require_lowerable, signature);
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
    npy_intp schedule_element_count = -1;
    int dtype = -1;
    int itemsize = 0;
    int supported = 1;
    for (Py_ssize_t i = 0; i < nargs; i++) {
        if (PyArray_CheckExact(args[i])) {
            PyArrayObject *array = (PyArrayObject *)args[i];
            if (schedule_element_count < 0) {
                schedule_element_count = PyArray_SIZE(array);
            }
            int current_dtype = PyArray_TYPE(array);
            int layout = numpy_jit_region_layout_class(array);
            hash = numpy_jit_region_hash_u64(hash, 0x100u + (uint64_t)i);
            hash = numpy_jit_region_hash_u64(hash, (uint64_t)current_dtype);
            hash = numpy_jit_region_hash_u64(
                hash, (uint64_t)PyArray_NDIM(array));
            hash = numpy_jit_region_hash_u64(hash, (uint64_t)layout);
            hash = numpy_jit_region_hash_u64(
                hash, PyArray_ISWRITEABLE(array) ? 1u : 0u);
            if (require_lowerable) {
                int layout4_fused_supported = !payload->region_generic &&
                    (payload->fused_op ==
                         NUMPY_JIT_FUSED_ADD_ARRAY_MUL_SCALAR ||
                     payload->fused_op ==
                         NUMPY_JIT_FUSED_SUB_ARRAY_MUL_SCALAR ||
                     payload->fused_op ==
                         NUMPY_JIT_FUSED_ADD_ARRAY_DIV_SCALAR ||
                     payload->fused_op ==
                         NUMPY_JIT_FUSED_SUB_ARRAY_DIV_SCALAR);
                layout4_fused_supported |=
                    !payload->region_generic &&
                    payload->fused_op == NUMPY_JIT_FUSED_ADD_THREE_ARRAYS;
                if (!numpy_jit_is_supported_region_dtype(current_dtype) ||
                    layout == 0 ||
                    (layout == 4 && payload->region_compiled_kind != 5 &&
                     !layout4_fused_supported)) {
                    supported = 0;
                }
                if (payload->region_generic &&
                    payload->region_compiled_kind == 0 && layout != 1) {
                    supported = 0;
                }
                if ((payload->region_compiled_kind == 3 ||
                     payload->region_compiled_kind == 4 ||
                     payload->region_compiled_kind == 6) && layout != 1) {
                    supported = 0;
                }
                if (shape_source == NULL) {
                    shape_source = array;
                    element_count = PyArray_SIZE(array);
                    dtype = current_dtype;
                    itemsize = PyArray_ITEMSIZE(array);
                }
                else if (current_dtype != dtype ||
                         PyArray_NDIM(array) !=
                             PyArray_NDIM(shape_source)) {
                    supported = 0;
                }
                else {
                    for (int dim = 0; dim < PyArray_NDIM(array); dim++) {
                        if (PyArray_DIM(array, dim) !=
                            PyArray_DIM(shape_source, dim)) {
                            supported = 0;
                            break;
                        }
                    }
                }
            }
        }
        else {
            hash = numpy_jit_region_hash_u64(hash, 0x200u + (uint64_t)i);
            hash = numpy_jit_region_hash_u64(
                hash, (uint64_t)(uintptr_t)Py_TYPE(args[i]));
            if (require_lowerable) {
                double ignored;
                if (!exact_python_scalar_as_double(
                        payload, args[i], &ignored)) {
                    PyErr_Clear();
                    supported = 0;
                }
            }
        }
    }
    /* The property identity includes size because candidate profitability is
     * not monotonic across problem sizes.  Do this in the cheap signature
     * pass too, so each bucket receives an independent PIC/decision entry. */
    hash = numpy_jit_region_hash_u64(
        hash, (uint64_t)numpy_jit_region_size_bucket(schedule_element_count));
    if (require_lowerable &&
        (dtype == NPY_INT || dtype == NPY_LONG || dtype == NPY_LONGLONG)) {
        for (Py_ssize_t i = 0; i < nargs; i++) {
            if (!PyArray_CheckExact(args[i]) &&
                !(PyLong_CheckExact(args[i]) ||
                  PyArray_IsScalar(args[i], SignedInteger))) {
                supported = 0;
            }
        }
        for (int i = 0; i < payload->region_node_count; i++) {
            if (payload->region_nodes[i].op ==
                    NUMPY_JIT_KERNEL_REGION_DIV) {
                supported = 0;
            }
        }
    }
    if (payload->region_compiled_kind == 8 ||
        payload->region_compiled_kind == 9) {
        /* The schedule choice is part of the PIC property key.  This must be
         * included in both the cheap property-signature pass and the full
         * specialization inspection; otherwise their signatures can never
         * match and no positive lowering is installed. */
        hash = numpy_jit_region_hash_u64(
            hash, schedule_element_count >=
                      NUMPY_JIT_MUL_ADDSUB_REGION_MIN_ELEMENTS ? 1u : 0u);
    }
    for (int i = 0; i < payload->region_node_count; i++) {
        if (payload->region_ops[i] != NUMPY_JIT_OP_POWER) {
            continue;
        }
        int exponent_slot = payload->region_right[i];
        int is_square = exponent_slot >= 0 && exponent_slot < nargs &&
            numpy_jit_is_exact_power2_exponent(args[exponent_slot]);
        /* POWER value eligibility is a true property specialization: cube
         * and square at one bytecode site must not share a negative PIC key. */
        hash = numpy_jit_region_hash_u64(
            hash, UINT64_C(0x504f574552320000) | (uint64_t)is_square);
    }
    if (require_lowerable && payload->region_compiled_kind == 5 &&
        !numpy_jit_is_supported_float_dtype(dtype)) {
        supported = 0;
    }
    if (require_lowerable && payload->region_compiled_kind == 6) {
        NumpyJitKernelRegionNode *first = &payload->region_nodes[0];
        NumpyJitKernelRegionNode *tail = &payload->region_nodes[1];
        int t0 = payload->region_input_count;
        int scalar_slot = tail->left == t0 ? tail->right : tail->left;
        if (first->left >= nargs || first->right >= nargs ||
            scalar_slot < 0 || scalar_slot >= nargs ||
            !PyArray_CheckExact(args[first->left]) ||
            !PyArray_CheckExact(args[first->right]) ||
            PyArray_CheckExact(args[scalar_slot])) {
            supported = 0;
        }
    }
    if (require_lowerable) {
        for (int i = 0; i < payload->region_node_count; i++) {
            if (payload->region_ops[i] != NUMPY_JIT_OP_POWER) {
                continue;
            }
            int exponent_slot = payload->region_right[i];
            if (exponent_slot < 0 || exponent_slot >= nargs ||
                !numpy_jit_is_exact_power2_exponent(args[exponent_slot])) {
                supported = 0;
            }
        }
    }
    int errmask;
    if (_get_errmask_fast(&errmask) < 0) {
        return -1;
    }
    hash = numpy_jit_region_hash_u64(hash, (uint64_t)(uint32_t)errmask);
    if (require_lowerable) {
        for (int shift = 0; shift <= 9; shift += 3) {
            if (((errmask >> shift) & 7) > 1) {
                supported = 0;
            }
        }
    }

    /* A PIC property key must be cheap and stable.  Shape equality, scalar
       conversion, aliasing and the byte-based profitability decision are
       execution preconditions, not lowering identity.  They are checked
       while installing a lowering and again by its semantic body.  Omitting
       them here avoids a second full validation pass on every hot PIC hit;
       a later shape/value change safely becomes a body MISS. */
    *signature = hash;
    if (!require_lowerable) {
        return 1;
    }

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
                if (overlap != 0) {
                    supported = 0;
                }
            }
        }
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
numpy_region_atomic_prefix_eval(NumpyJitPlanPayload *payload,
                                PyObject *const *args, Py_ssize_t nargs,
                                PyObject **result)
{
    if (payload == NULL || nargs != 3 || payload->region_input_count != 3 ||
        payload->region_node_count != 2) {
        return PYJIT_EXT_MISS;
    }
    PyJitExtPlan *first = payload->region_atomic_plans[0];
    PyJitExtPlan *tail = payload->region_atomic_plans[1];
    int first_left = payload->region_left[0];
    int first_right = payload->region_right[0];
    int tail_consumes_left =
        payload->region_left[1] == payload->region_input_count;
    int tail_consumes_right =
        payload->region_right[1] == payload->region_input_count;
    int tail_other = tail_consumes_left ? payload->region_right[1] :
        payload->region_left[1];
    if (first == NULL || tail == NULL ||
        first->kind != PYJIT_EXT_PLAN_BINARY ||
        tail->kind != PYJIT_EXT_PLAN_BINARY ||
        first_left < 0 || first_left >= nargs ||
        first_right < 0 || first_right >= nargs ||
        (!tail_consumes_left && !tail_consumes_right) ||
        tail_other < 0 || tail_other >= nargs) {
        return PYJIT_EXT_MISS;
    }

    PyObject *temporary = NULL;
    PyJitExtResult status = first->eval.binary_eval(
        first->payload, args[first_left], args[first_right], &temporary);
    if ((status != PYJIT_EXT_HIT && status != PYJIT_EXT_COLD) ||
        temporary == NULL) {
        if (status == PYJIT_EXT_ERROR) {
            return PYJIT_EXT_ERROR;
        }
        Py_XDECREF(temporary);
        return numpy_region_fallback_eval(payload, args, nargs, result);
    }
    PyObject *out = NULL;
    status = tail->eval.binary_eval(
        tail->payload,
        tail_consumes_left ? temporary : args[tail_other],
        tail_consumes_left ? args[tail_other] : temporary, &out);
    Py_DECREF(temporary);
    if ((status != PYJIT_EXT_HIT && status != PYJIT_EXT_COLD) || out == NULL) {
        if (status == PYJIT_EXT_ERROR) {
            return PYJIT_EXT_ERROR;
        }
        Py_XDECREF(out);
        /* A runtime property outside the atomic plan's fast domain is still
         * semantically valid.  Replay the exact NumPy recipe for this
         * execution without poisoning the positive Region PIC. */
        return numpy_region_fallback_eval(payload, args, nargs, result);
    }
    *result = out;
    /* Each atomic plan is executed through its original evaluator and keeps
     * its own PRESERVES_FP_POLICY contract.  Re-wrapping the whole prefix in
     * a second get/clear/check sequence both changes that execution model and
     * is measurable for ADI's 98-element inner loops. */
    return PYJIT_EXT_HIT;
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
    /* This is the committed DispatchEntry guard.  Validate every property
       captured by this lowering before entering the selected target; only
       value-dependent shape/alias and current FP policy remain in the body. */
    for (Py_ssize_t i = 0; i < nargs; i++) {
        if (Py_TYPE(args[i]) != lowering->types[i]) {
            return PYJIT_EXT_MISS;
        }
        if ((lowering->array_mask & (1u << i)) != 0) {
            if (!PyArray_CheckExact(args[i])) {
                return PYJIT_EXT_MISS;
            }
            PyArrayObject *array = (PyArrayObject *)args[i];
            if (PyArray_TYPE(array) != lowering->dtypes[i] ||
                PyArray_NDIM(array) != lowering->ndims[i] ||
                numpy_jit_region_layout_class(array) != lowering->layouts[i] ||
                (PyArray_ISWRITEABLE(array) !=
                 ((lowering->writable_mask & (1u << i)) != 0))) {
                return PYJIT_EXT_MISS;
            }
        }
    }
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

/* A schedule/lowering is merely a candidate.  Measure it against the exact
 * atomic replay used on Region misses before allowing it onto the hot path.
 * The PIC signature already includes the Region size bucket, so decisions
 * here are per graph + runtime property + size class. */
static PyJitExtResult
numpy_jit_region_execute_atomic(NumpyJitPlanPayload *payload,
                                NumpyJitDispatchEntry *entry,
                                PyObject *const *args, Py_ssize_t nargs,
                                PyObject **result)
{
    (void)entry;
    return numpy_region_local_fallback(payload, args, nargs, result);
}

static PyJitExtResult
numpy_jit_region_execute_candidate(NumpyJitPlanPayload *payload,
                                   NumpyJitDispatchEntry *entry,
                                   PyObject *const *args, Py_ssize_t nargs,
                                   PyObject **result)
{
    (void)payload;
    return entry->lowered->guard_eval(
        entry->lowered->payload, args, nargs, result);
}

/* The only path that is allowed to sample or make a selection.  Once it
 * commits, `selected_execute` is a stable direct target for that property. */
static PyJitExtResult
numpy_jit_region_execute_probation(NumpyJitPlanPayload *payload,
                                   NumpyJitDispatchEntry *entry,
                                   PyObject *const *args, Py_ssize_t nargs,
                                   PyObject **result)
{
    const char *force = Py_GETENV("NUMPY_JIT_VARIANT_FORCE");
    int force_atomic = force != NULL && strcmp(force, "atomic") == 0;
    int force_candidate = force != NULL && strcmp(force, "candidate") == 0;
    int use_atomic = force_atomic || (!force_candidate &&
        (entry->decision == NUMPY_JIT_VARIANT_COMMITTED_ATOMIC ||
        (entry->decision == NUMPY_JIT_VARIANT_PROBATION &&
         entry->profile.atomic_samples < NUMPY_JIT_VARIANT_PROBATION_SAMPLES)));
    struct timespec begin, end;
    int timed = clock_gettime(CLOCK_MONOTONIC, &begin) == 0;
    PyJitExtResult status = use_atomic ?
        numpy_jit_region_execute_atomic(payload, entry, args, nargs, result) :
        numpy_jit_region_execute_candidate(payload, entry, args, nargs, result);
    if ((status == PYJIT_EXT_HIT || status == PYJIT_EXT_COLD) && timed &&
        clock_gettime(CLOCK_MONOTONIC, &end) == 0) {
        uint64_t elapsed = numpy_jit_elapsed_ns(&begin, &end);
        if (use_atomic) {
            entry->profile.atomic_samples++;
            entry->profile.atomic_elapsed_ns += elapsed;
        }
        else if (entry->decision == NUMPY_JIT_VARIANT_PROBATION) {
            entry->profile.candidate_samples++;
            entry->profile.candidate_elapsed_ns += elapsed;
        }
    }
    if (status != PYJIT_EXT_HIT && status != PYJIT_EXT_COLD) {
        return status;
    }
    if (entry->decision == NUMPY_JIT_VARIANT_PROBATION &&
        entry->profile.atomic_samples >= NUMPY_JIT_VARIANT_PROBATION_SAMPLES &&
        entry->profile.candidate_samples >= NUMPY_JIT_VARIANT_PROBATION_SAMPLES) {
        /* Candidate needs a centralized margin, rather than schedule-local
         * notions of profitability. */
        uint64_t atomic = entry->profile.atomic_elapsed_ns;
        uint64_t candidate = entry->profile.candidate_elapsed_ns;
        uint64_t min_gain = numpy_jit_variant_min_gain_percent();
        entry->decision = candidate * 100u <=
            atomic * (100u - min_gain) ?
            NUMPY_JIT_VARIANT_COMMITTED_CANDIDATE :
            NUMPY_JIT_VARIANT_COMMITTED_ATOMIC;
        entry->selected_variant =
            entry->decision == NUMPY_JIT_VARIANT_COMMITTED_CANDIDATE ?
                (payload->region_compiled_kind != 0 ?
                    NUMPY_JIT_VARIANT_DEDICATED :
                    NUMPY_JIT_VARIANT_GENERIC) :
                NUMPY_JIT_VARIANT_ATOMIC;
        entry->selected_execute =
            entry->decision == NUMPY_JIT_VARIANT_COMMITTED_CANDIDATE ?
                numpy_jit_region_execute_candidate :
                numpy_jit_region_execute_atomic;
        if (payload->site != NULL) {
            payload->site->region_lowering_name =
                entry->decision == NUMPY_JIT_VARIANT_COMMITTED_CANDIDATE ?
                entry->lowered->debug_name : "numpy.region.atomic.committed";
        }
        if (numpy_jit_stats_enabled()) {
            const char *dir = getenv("PYTHON_JIT_EXT_STATS_DIR");
            if (dir == NULL || dir[0] == '\0') dir = "/tmp";
            char path[4096];
            snprintf(path, sizeof(path), "%s/crossjit_numpy_variant_decisions.ndjson", dir);
            FILE *fp = fopen(path, "a");
            if (fp != NULL) {
                fprintf(fp,
                    "{\"graph_id\":%" PRIu64
                    ",\"property_key\":%" PRIu64
                    ",\"size_bucket\":%u,\"atomic_sample_count\":%u"
                    ",\"candidate_sample_count\":%u,\"atomic_total_ns\":%" PRIu64
                    ",\"candidate_total_ns\":%" PRIu64
                    ",\"decision\":\"%s\",\"selected_variant\":\"%s\"}\n",
                    payload->region_graph_hash, entry->signature,
                    (unsigned)entry->size_bucket,
                    (unsigned)entry->profile.atomic_samples,
                    (unsigned)entry->profile.candidate_samples, atomic, candidate,
                    entry->decision == NUMPY_JIT_VARIANT_COMMITTED_CANDIDATE ?
                        "CANDIDATE_FASTER" : "ATOMIC_FASTER",
                    entry->decision == NUMPY_JIT_VARIANT_COMMITTED_CANDIDATE ?
                        "CANDIDATE" : "ATOMIC");
                fclose(fp);
            }
        }
    }
    return status;
}

static PyJitExtResult
numpy_jit_region_dispatch_entry(NumpyJitPlanPayload *payload,
                                NumpyJitDispatchEntry *entry,
                                PyObject *const *args, Py_ssize_t nargs,
                                PyObject **result)
{
    /* A committed DispatchEntry is the steady-state path: one guard/target
       call.  Candidate enumeration, AOT/LLVM lookup, timing and force-mode
       parsing are deliberately confined to probation/slow dispatch. */
    if (entry->decision != NUMPY_JIT_VARIANT_PROBATION &&
        entry->selected_execute != NULL) {
        return entry->selected_execute(payload, entry, args, nargs, result);
    }
    return numpy_jit_region_execute_probation(
        payload, entry, args, nargs, result);
}

static PyJitExtResult
numpy_masked_mean_region_eval(NumpyJitPlanPayload *payload,
                              PyObject *const *args, Py_ssize_t nargs,
                              PyObject **result)
{
    if (payload == NULL || nargs != payload->region_input_count) {
        return PYJIT_EXT_MISS;
    }
    NumpyJitPlanPayload *logical =
        (NumpyJitPlanPayload *)payload->region_atomic_plans[2]->payload;
    NumpyJitPlanPayload *mean =
        (NumpyJitPlanPayload *)payload->region_atomic_plans[4]->payload;
    int logical_guarded = 0;
    int mean_guarded = 0;
    for (Py_ssize_t i = 0; i < nargs; i++) {
        logical_guarded |= args[i] == logical->expected_callable;
        mean_guarded |= args[i] == mean->expected_callable;
    }
    if (!logical_guarded || !mean_guarded) {
        return PYJIT_EXT_MISS;
    }

    PyArrayObject *radius[2] = {NULL, NULL};
    double scalar[2] = {0.0, 0.0};
    int array_on_left[2] = {0, 0};
    for (int node = 0; node < 2; node++) {
        int left_slot = payload->region_left[node];
        int right_slot = payload->region_right[node];
        if (left_slot >= nargs || right_slot >= nargs) {
            return PYJIT_EXT_MISS;
        }
        PyObject *left = args[left_slot];
        PyObject *right = args[right_slot];
        if (PyArray_CheckExact(left) && !PyArray_CheckExact(right)) {
            radius[node] = (PyArrayObject *)left;
            array_on_left[node] = 1;
            scalar[node] = PyFloat_AsDouble(right);
        }
        else if (!PyArray_CheckExact(left) && PyArray_CheckExact(right)) {
            radius[node] = (PyArrayObject *)right;
            scalar[node] = PyFloat_AsDouble(left);
        }
        else {
            return PYJIT_EXT_MISS;
        }
        if (scalar[node] == -1.0 && PyErr_Occurred()) {
            PyErr_Clear();
            return PYJIT_EXT_MISS;
        }
        if (PyArray_TYPE(radius[node]) != NPY_DOUBLE ||
            PyArray_NDIM(radius[node]) != 1 ||
            !PyArray_IS_C_CONTIGUOUS(radius[node])) {
            return PYJIT_EXT_MISS;
        }
    }
    if (PyArray_SIZE(radius[0]) != PyArray_SIZE(radius[1])) {
        return PYJIT_EXT_MISS;
    }
    int mask_slot = payload->region_input_count + 2;
    int data_slot = payload->region_left[3] == mask_slot ?
        payload->region_right[3] : payload->region_left[3];
    if (data_slot < 0 || data_slot >= nargs ||
        !PyArray_CheckExact(args[data_slot])) {
        return PYJIT_EXT_MISS;
    }
    PyArrayObject *data = (PyArrayObject *)args[data_slot];
    if (PyArray_TYPE(data) != NPY_DOUBLE || PyArray_NDIM(data) != 1 ||
        !PyArray_IS_C_CONTIGUOUS(data) ||
        PyArray_SIZE(data) != PyArray_SIZE(radius[0])) {
        return PYJIT_EXT_MISS;
    }

    const double *radius0 = (const double *)PyArray_DATA(radius[0]);
    const double *radius1 = (const double *)PyArray_DATA(radius[1]);
    const double *values = (const double *)PyArray_DATA(data);
    npy_intp size = PyArray_SIZE(data);
    npy_intp count = 0;
    double sum = 0.0;
    for (npy_intp i = 0; i < size; i++) {
        double left0 = array_on_left[0] ? radius0[i] : scalar[0];
        double right0 = array_on_left[0] ? scalar[0] : radius0[i];
        double left1 = array_on_left[1] ? radius1[i] : scalar[1];
        double right1 = array_on_left[1] ? scalar[1] : radius1[i];
        int cmp0 = payload->region_ops[0] == NUMPY_JIT_OP_LESS ?
            left0 < right0 : left0 <= right0;
        int cmp1 = payload->region_ops[1] == NUMPY_JIT_OP_LESS ?
            left1 < right1 : left1 <= right1;
        if (cmp0 && cmp1) {
            sum += values[i];
            count++;
        }
    }
    if (count == 0) {
        return PYJIT_EXT_MISS;
    }
    PyObject *out = PyArrayScalar_New(Double);
    if (out == NULL) {
        return PYJIT_EXT_ERROR;
    }
    PyArrayScalar_ASSIGN(out, Double, sum / (double)count);
    *result = out;
    return PYJIT_EXT_HIT;
}

static PyJitExtResult
numpy_compare_logical_region_eval(NumpyJitPlanPayload *payload,
                                  PyObject *const *args, Py_ssize_t nargs,
                                  PyObject **result)
{
    if (payload == NULL || nargs != payload->region_input_count) {
        return PYJIT_EXT_MISS;
    }
    NumpyJitPlanPayload *logical =
        (NumpyJitPlanPayload *)payload->region_atomic_plans[2]->payload;
    int callable_guarded = 0;
    for (Py_ssize_t i = 0; i + 1 < nargs; i++) {
        if (args[i] == logical->expected_callable && args[i + 1] == NULL) {
            callable_guarded = 1;
            break;
        }
    }
    if (!callable_guarded) {
        return PYJIT_EXT_MISS;
    }

    PyArrayObject *arrays[2] = {NULL, NULL};
    double scalars[2] = {0.0, 0.0};
    int array_on_left[2] = {0, 0};
    for (int node = 0; node < 2; node++) {
        int left_slot = payload->region_left[node];
        int right_slot = payload->region_right[node];
        if (left_slot >= nargs || right_slot >= nargs) {
            return PYJIT_EXT_MISS;
        }
        PyObject *left = args[left_slot];
        PyObject *right = args[right_slot];
        if (PyArray_CheckExact(left) && !PyArray_CheckExact(right)) {
            arrays[node] = (PyArrayObject *)left;
            array_on_left[node] = 1;
            scalars[node] = PyFloat_AsDouble(right);
        }
        else if (!PyArray_CheckExact(left) && PyArray_CheckExact(right)) {
            arrays[node] = (PyArrayObject *)right;
            array_on_left[node] = 0;
            scalars[node] = PyFloat_AsDouble(left);
        }
        else {
            return PYJIT_EXT_MISS;
        }
        if (scalars[node] == -1.0 && PyErr_Occurred()) {
            PyErr_Clear();
            return PYJIT_EXT_MISS;
        }
        if (PyArray_TYPE(arrays[node]) != NPY_DOUBLE ||
            !PyArray_IS_C_CONTIGUOUS(arrays[node])) {
            return PYJIT_EXT_MISS;
        }
    }
    if (PyArray_NDIM(arrays[0]) != PyArray_NDIM(arrays[1]) ||
        PyArray_SIZE(arrays[0]) != PyArray_SIZE(arrays[1])) {
        return PYJIT_EXT_MISS;
    }
    for (int axis = 0; axis < PyArray_NDIM(arrays[0]); axis++) {
        if (PyArray_DIM(arrays[0], axis) != PyArray_DIM(arrays[1], axis)) {
            return PYJIT_EXT_MISS;
        }
    }
    PyArray_Descr *bool_descr = PyArray_DescrFromType(NPY_BOOL);
    if (bool_descr == NULL) {
        return PYJIT_EXT_ERROR;
    }
    PyArrayObject *out = (PyArrayObject *)PyArray_NewFromDescr(
        &PyArray_Type, bool_descr, PyArray_NDIM(arrays[0]),
        PyArray_DIMS(arrays[0]), NULL, NULL, 0, NULL);
    if (out == NULL) {
        return PYJIT_EXT_ERROR;
    }
    const double *data0 = (const double *)PyArray_DATA(arrays[0]);
    const double *data1 = (const double *)PyArray_DATA(arrays[1]);
    npy_bool *out_data = (npy_bool *)PyArray_DATA(out);
    npy_intp size = PyArray_SIZE(arrays[0]);
    for (npy_intp i = 0; i < size; i++) {
        double left0 = array_on_left[0] ? data0[i] : scalars[0];
        double right0 = array_on_left[0] ? scalars[0] : data0[i];
        double left1 = array_on_left[1] ? data1[i] : scalars[1];
        double right1 = array_on_left[1] ? scalars[1] : data1[i];
        int cmp0 = payload->region_ops[0] == NUMPY_JIT_OP_LESS ?
            left0 < right0 : left0 <= right0;
        int cmp1 = payload->region_ops[1] == NUMPY_JIT_OP_LESS ?
            left1 < right1 : left1 <= right1;
        out_data[i] = (npy_bool)(cmp0 && cmp1);
    }
    *result = (PyObject *)out;
    return PYJIT_EXT_HIT;
}

static PyJitExtResult
numpy_outer_minimum_region_eval(NumpyJitPlanPayload *payload,
                                PyObject *const *args, Py_ssize_t nargs,
                                PyObject **result)
{
    if (payload == NULL || nargs != 7 ||
        payload->region_left[0] + 1 != payload->region_right[0]) {
        return PYJIT_EXT_MISS;
    }
    int col_slot = payload->region_left[0];
    int row_slot = payload->region_right[0];
    int outer_output = payload->region_input_count;
    int path_slot = payload->region_left[1] == outer_output ?
        payload->region_right[1] : payload->region_left[1];
    int outer_callable_slot = col_slot - 2;
    int outer_receiver_slot = col_slot - 1;
    int minimum_callable_slot = path_slot - 2;
    int minimum_receiver_slot = path_slot - 1;
    if (path_slot < 2 || col_slot < 2 || row_slot >= nargs ||
        outer_callable_slot < 0 || minimum_callable_slot < 0 ||
        outer_receiver_slot >= nargs || minimum_receiver_slot >= nargs) {
        return PYJIT_EXT_MISS;
    }
    NumpyJitPlanPayload *outer =
        (NumpyJitPlanPayload *)payload->region_atomic_plans[0]->payload;
    NumpyJitPlanPayload *minimum =
        (NumpyJitPlanPayload *)payload->region_atomic_plans[1]->payload;
    if (args[outer_callable_slot] != outer->expected_callable ||
        args[outer_receiver_slot] != outer->expected_receiver ||
        args[minimum_callable_slot] != minimum->expected_callable ||
        args[minimum_receiver_slot] != NULL ||
        !PyArray_CheckExact(args[path_slot]) ||
        !PyArray_CheckExact(args[col_slot]) ||
        !PyArray_CheckExact(args[row_slot])) {
        return PYJIT_EXT_MISS;
    }
    PyArrayObject *path = (PyArrayObject *)args[path_slot];
    PyArrayObject *col = (PyArrayObject *)args[col_slot];
    PyArrayObject *row = (PyArrayObject *)args[row_slot];
    if (PyArray_TYPE(path) != NPY_INT || PyArray_TYPE(col) != NPY_INT ||
        PyArray_TYPE(row) != NPY_INT || PyArray_NDIM(path) != 2 ||
        PyArray_NDIM(col) != 1 || PyArray_NDIM(row) != 1 ||
        !PyArray_IS_C_CONTIGUOUS(path) ||
        PyArray_DIM(path, 0) != PyArray_DIM(path, 1) ||
        PyArray_DIM(col, 0) != PyArray_DIM(path, 0) ||
        PyArray_DIM(row, 0) != PyArray_DIM(path, 1) ||
        PyArray_STRIDE(col, 0) <= 0 || PyArray_STRIDE(row, 0) <= 0) {
        return PYJIT_EXT_MISS;
    }
    npy_intp dims[2] = {PyArray_DIM(path, 0), PyArray_DIM(path, 1)};
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_INT);
    if (out == NULL) {
        return PYJIT_EXT_ERROR;
    }
    const npy_int32 *path_data = (const npy_int32 *)PyArray_DATA(path);
    const char *col_data = PyArray_BYTES(col);
    const char *row_data = PyArray_BYTES(row);
    npy_int32 *out_data = (npy_int32 *)PyArray_DATA(out);
    npy_intp n = dims[0];
    npy_intp col_stride = PyArray_STRIDE(col, 0);
    npy_intp row_stride = PyArray_STRIDE(row, 0);
    for (npy_intp i = 0; i < n; i++) {
        npy_int32 col_value = *(const npy_int32 *)(col_data + i * col_stride);
        for (npy_intp j = 0; j < n; j++) {
            npy_int32 row_value =
                *(const npy_int32 *)(row_data + j * row_stride);
            npy_int32 sum = (npy_int32)(
                (npy_uint32)col_value + (npy_uint32)row_value);
            npy_int32 old = path_data[i * n + j];
            out_data[i * n + j] = old < sum ? old : sum;
        }
    }
    *result = (PyObject *)out;
    return PYJIT_EXT_HIT;
}

static PyJitExtResult
numpy_region_non_native_eval(void *payload_obj, PyObject *const *args,
                             Py_ssize_t nargs, PyObject **result)
{
    if (payload_obj == NULL || args == NULL) {
        return PYJIT_EXT_MISS;
    }
    NumpyJitPlanPayload *payload = (NumpyJitPlanPayload *)payload_obj;
    if (payload->region_masked_mean) {
        return numpy_masked_mean_region_eval(
            payload, args, nargs, result);
    }
    if (payload->region_compare_logical) {
        return numpy_compare_logical_region_eval(
            payload, args, nargs, result);
    }
    if (payload->region_outer_minimum) {
        return numpy_outer_minimum_region_eval(
            payload, args, nargs, result);
    }
    if (payload->region_compiled_kind == 8 ||
        payload->region_compiled_kind == 9) {
        /* Schedule 8/9 cross over at 256 elements.  Select the small atomic
         * recipe before entering the generic Region/PIC machinery: ADI
         * executes this branch hundreds of thousands of times, so even a
         * successful PIC envelope is material at its 98-element inner size.
         * The scan is bounded by the three-input schedule and also handles a
         * scalar occupying the first structural input. */
        npy_intp elements = -1;
        for (Py_ssize_t i = 0; i < nargs; i++) {
            if (PyArray_CheckExact(args[i])) {
                elements = PyArray_SIZE((PyArrayObject *)args[i]);
                break;
            }
        }
        if (elements >= 0 &&
            elements < NUMPY_JIT_MUL_ADDSUB_REGION_MIN_ELEMENTS) {
            if (payload->site != NULL) {
                payload->site->region_lowering_name =
                    "numpy.region.atomic_prefix.runtime";
            }
            return numpy_region_atomic_prefix_eval(
                payload, args, nargs, result);
        }
    }
    if (payload->region_compiled_kind == 5 &&
        payload->region_node_count == 2 && nargs == 3) {
        /* The structure ``(a +/- b) * scalar`` is also shared by hdiff's
           three-array ``a * (b - c)`` form.  Its last multiply consumes the
           temporary on the right, so it is not eligible for the scalar
           schedule.  Decline the Region here: returning MISS lets CPython
           replay the original atomic operations, retaining their independent
           specialization/PICs without paying Region entry bookkeeping. */
        NumpyJitKernelRegionNode *tail = &payload->region_nodes[1];
        int temporary = payload->region_input_count;
        int other = tail->left == temporary ? tail->right : tail->left;
        if ((tail->left == temporary || tail->right == temporary) &&
            other >= 0 && other < nargs && PyArray_CheckExact(args[other])) {
            if (payload->site != NULL) {
                payload->site->region_lowering_name =
                    "numpy.region.atomic_prefix.role_mismatch";
            }
            return PYJIT_EXT_MISS;
        }
    }
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
    int fast_pic = payload->region_fast_pic != 0 ?
        (int)payload->region_fast_pic - 1 : -1;
    if (fast_pic >= 0 && fast_pic < payload->region_pic_count) {
        NumpyJitDispatchEntry *entry = &payload->region_pic[fast_pic];
        /* This is the monomorphic fast entry.  The selector still owns the
           choice between the lowered candidate and the atomic baseline. */
        PyJitExtResult evaluated = numpy_jit_region_dispatch_entry(
            payload, entry, args, nargs, result);
        if (evaluated == PYJIT_EXT_HIT || evaluated == PYJIT_EXT_COLD) {
            numpy_jit_region_inc_u16(
                site != NULL ? &site->region_pic_hit_count : NULL);
            if (site != NULL &&
                entry->decision != NUMPY_JIT_VARIANT_COMMITTED_ATOMIC) {
                site->region_lowering_name = entry->lowered->debug_name;
            }
            return evaluated;
        }
        if (evaluated == PYJIT_EXT_ERROR) {
            return evaluated;
        }
    }
    for (int i = 0; i < payload->region_pic_count; i++) {
        if (i == fast_pic) {
            continue;
        }
        NumpyJitDispatchEntry *entry = &payload->region_pic[i];
        if (entry->negative) {
            continue;
        }
        PyJitExtResult evaluated = numpy_jit_region_dispatch_entry(
            payload, entry, args, nargs, result);
        if (evaluated == PYJIT_EXT_HIT || evaluated == PYJIT_EXT_COLD) {
            numpy_jit_region_inc_u16(
                site != NULL ? &site->region_pic_hit_count : NULL);
            if (site != NULL &&
                entry->decision != NUMPY_JIT_VARIANT_COMMITTED_ATOMIC) {
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
        NumpyJitDispatchEntry *entry = &payload->region_pic[i];
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
            if (site != NULL) {
                site->region_lowering_name = "numpy.region.atomic.committed";
            }
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
        NumpyJitDispatchEntry *entry = &payload->region_pic[i];
        if (entry->signature != signature) {
            continue;
        }
        if (entry->negative) {
            numpy_jit_region_inc_u16(
                site != NULL ? &site->region_pic_miss_count : NULL);
            if (site != NULL) {
                site->region_lowering_name = "numpy.region.atomic.committed";
            }
            if (payload->region_runtime_miss_on_decline &&
                payload->region_fast_pic == 0) {
                return PYJIT_EXT_MISS;
            }
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
    NumpyJitDispatchEntry *entry =
        &payload->region_pic[payload->region_pic_count++];
    entry->signature = signature;
    entry->lowered = lowered;
    entry->negative = lowered == NULL;
    npy_intp entry_size = 0;
    for (Py_ssize_t i = 0; i < nargs; i++) {
        if (PyArray_CheckExact(args[i])) {
            entry_size = PyArray_SIZE((PyArrayObject *)args[i]);
            break;
        }
    }
    entry->size_bucket = (uint8_t)numpy_jit_region_size_bucket(entry_size);
    entry->decision = lowered == NULL ? 0 : NUMPY_JIT_VARIANT_PROBATION;
    entry->selected_variant = NUMPY_JIT_VARIANT_ATOMIC;
    entry->selected_execute = lowered == NULL ?
        numpy_jit_region_execute_atomic : NULL;
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
        if (site != NULL) {
            site->region_lowering_name = "numpy.region.atomic.committed";
        }
        if (payload->region_runtime_miss_on_decline &&
            payload->region_fast_pic == 0) {
            return PYJIT_EXT_MISS;
        }
        return numpy_region_local_fallback(payload, args, nargs, result);
    }
    payload->region_fast_pic = payload->region_pic_count;
    PyJitExtResult evaluated = numpy_jit_region_dispatch_entry(
        payload, entry, args, nargs, result);
    if (evaluated == PYJIT_EXT_HIT || evaluated == PYJIT_EXT_COLD) {
        numpy_jit_region_inc_u16(
            site != NULL ? &site->region_pic_hit_count : NULL);
        if (site != NULL &&
            entry->decision != NUMPY_JIT_VARIANT_COMMITTED_ATOMIC) {
            site->region_lowering_name = lowered->debug_name;
        }
        return evaluated;
    }
    if (evaluated == PYJIT_EXT_ERROR) {
        return evaluated;
    }
    return numpy_region_local_fallback(payload, args, nargs, result);
}

static PyJitExtResult
numpy_region_eval(void *payload_obj, PyObject *const *args,
                  Py_ssize_t nargs, PyObject **result)
{
    if (payload_obj == NULL || args == NULL) {
        return PYJIT_EXT_MISS;
    }
    NumpyJitPlanPayload *payload = (NumpyJitPlanPayload *)payload_obj;
    if (!payload->region_generic || payload->inplace ||
        (numpy_jit_llvm_mode() == NUMPY_JIT_LLVM_MODE_OFF &&
         numpy_jit_aot_mode() == NUMPY_JIT_LLVM_MODE_OFF)) {
        return numpy_region_non_native_eval(
            payload_obj, args, nargs, result);
    }

    int fast_pic = payload->region_fast_pic != 0 ?
        (int)payload->region_fast_pic - 1 : -1;
    NumpyJitDispatchEntry *dispatch_entry =
        fast_pic >= 0 && fast_pic < payload->region_pic_count ?
            &payload->region_pic[fast_pic] : NULL;
    /* Establish and commit the atomic/Region incumbent before asking native
       producers for a challenger.  Thereafter this entry is the sole owner
       of the final target for its PropertyKey. */
    if (dispatch_entry == NULL ||
        dispatch_entry->decision == NUMPY_JIT_VARIANT_PROBATION) {
        return numpy_region_non_native_eval(
            payload_obj, args, nargs, result);
    }
    if (dispatch_entry->selected_execute ==
            numpy_jit_region_execute_native) {
        PyJitExtResult evaluated = numpy_jit_region_dispatch_entry(
            payload, dispatch_entry, args, nargs, result);
        if (evaluated == PYJIT_EXT_HIT || evaluated == PYJIT_EXT_COLD ||
            evaluated == PYJIT_EXT_ERROR) {
            return evaluated;
        }
        /* A specialized native guard miss is a slow-path event. */
    }

    struct timespec probation_begin, probation_end;
    int timing_started = 0;
    if (payload->region_llvm_entry != NULL &&
        numpy_jit_llvm_entry_state(payload->region_llvm_entry) ==
            NUMPY_JIT_LLVM_READY) {
        for (Py_ssize_t i = 0; i < nargs; i++) {
            if (PyArray_CheckExact(args[i])) {
                unsigned bucket = numpy_jit_llvm_size_bucket(
                    PyArray_SIZE((PyArrayObject *)args[i]));
                uint8_t decision =
                    payload->region_llvm_bucket_decision[bucket];
                if (decision == 3) {
                    return numpy_region_non_native_eval(
                        payload_obj, args, nargs, result);
                }
                if (decision == 0 || decision == 1) {
                    timing_started = clock_gettime(
                        CLOCK_MONOTONIC, &probation_begin) == 0;
                }
                break;
            }
        }
    }
    /* Once an AOT lookup is known not to serve this Region (or this size
     * bucket), stay on the existing Region path when that decision rules out
     * every native backend.  The old code rebuilt the complete direct-call
     * property vector on every execution merely to rediscover the same AOT
     * miss.  That fixed cost is visible for large square/subtract inputs,
     * whose manifest deliberately excludes bucket 2. */
    if (payload->region_aot_lookup_done) {
        if (payload->region_aot_entry == NULL) {
            if (numpy_jit_llvm_mode() == NUMPY_JIT_LLVM_MODE_OFF) {
                return numpy_region_non_native_eval(
                    payload_obj, args, nargs, result);
            }
        }
        else {
            npy_intp elements = -1;
            for (Py_ssize_t i = 0; i < nargs; i++) {
                if (PyArray_CheckExact(args[i])) {
                    elements = PyArray_SIZE((PyArrayObject *)args[i]);
                    break;
                }
            }
            NumpyJitAOTEntryInfo aot_info = {0};
            numpy_jit_aot_entry_info(payload->region_aot_entry, &aot_info);
            if (elements >= 0 &&
                (aot_info.accepted_bucket_mask &
                 (1u << numpy_jit_llvm_size_bucket(elements))) == 0) {
                return numpy_region_non_native_eval(
                    payload_obj, args, nargs, result);
            }
        }
    }
    NumpyJitLLVMDirectCall call;
    int prepared = numpy_jit_llvm_prepare_direct(
        payload, args, nargs, &call, 1);
    if (prepared < 0) {
        return PYJIT_EXT_ERROR;
    }
    if (!prepared) {
        return numpy_region_non_native_eval(
            payload_obj, args, nargs, result);
    }

    unsigned selected_bucket = numpy_jit_llvm_size_bucket(
        call.kernel_args.size);
    NumpyJitVariant selected = {0};
    int use_native = numpy_jit_variant_selector(
        payload, call.dtype, call.scalar_mask, call.kernel_args.size,
        &selected);
    int measure_probation = selected.measure_probation != 0;
    if (measure_probation && !timing_started) {
        timing_started =
            clock_gettime(CLOCK_MONOTONIC, &probation_begin) == 0;
    }
    PyJitExtResult status;
    if (use_native) {
        status = numpy_jit_compiled_execute_direct(
            payload, &call, selected.function, result,
            selected.kind == NUMPY_JIT_VARIANT_PRECOMPILED);
        if (selected.kind == NUMPY_JIT_VARIANT_PRECOMPILED ||
            (selected.kind == NUMPY_JIT_VARIANT_LLVM &&
             payload->region_llvm_bucket_decision[selected_bucket] == 2)) {
            dispatch_entry->selected_variant = (uint8_t)selected.kind;
            dispatch_entry->selected_function = selected.function;
            dispatch_entry->selected_execute =
                numpy_jit_region_execute_native;
        }
        if (status == PYJIT_EXT_HIT && payload->site != NULL) {
            payload->site->region_lowering_name =
                selected.kind == NUMPY_JIT_VARIANT_PRECOMPILED ?
                    "numpy.region.precompiled" : "numpy.region.llvm";
        }
    }
    else {
        payload->region_llvm_probation_old_active =
            measure_probation != 0;
        status = numpy_region_non_native_eval(
            payload_obj, args, nargs, result);
        payload->region_llvm_probation_old_active = 0;
    }
    if (measure_probation && timing_started &&
        (status == PYJIT_EXT_HIT || status == PYJIT_EXT_COLD) &&
        clock_gettime(CLOCK_MONOTONIC, &probation_end) == 0) {
        numpy_jit_llvm_record_probation(
            payload, use_native,
            numpy_jit_elapsed_ns(&probation_begin, &probation_end),
            call.kernel_args.size);
        if (selected.function != NULL &&
            payload->region_llvm_bucket_decision[selected_bucket] == 2) {
            dispatch_entry->selected_variant = (uint8_t)selected.kind;
            dispatch_entry->selected_function = selected.function;
            dispatch_entry->selected_execute =
                numpy_jit_region_execute_native;
        }
    }
    return status;
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
                    if (PyJit_RegisterBinaryProvider(
                            &PyArray_Type, &PyLongArrType_Type, oparg,
                            &numpy_jit_provider) < 0) {
                        PyErr_Clear();
                    }
                    if (PyJit_RegisterBinaryProvider(
                            &PyArray_Type, &PyLongLongArrType_Type, oparg,
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
                    if (PyJit_RegisterBinaryProvider(
                            &PyLongArrType_Type, &PyArray_Type, oparg,
                            &numpy_jit_provider) < 0) {
                        PyErr_Clear();
                    }
                    if (PyJit_RegisterBinaryProvider(
                            &PyLongLongArrType_Type, &PyArray_Type, oparg,
                            &numpy_jit_provider) < 0) {
                        PyErr_Clear();
                    }
                    break;
            }
        }
    }

    static const int richcompare_ops[] = {Py_LT, Py_LE};
    PyTypeObject *richcompare_scalar_types[] = {
        &PyFloat_Type,
        &PyLong_Type,
        &PyFloatArrType_Type,
        &PyDoubleArrType_Type,
        &PyLongArrType_Type,
        &PyLongLongArrType_Type,
    };
    for (Py_ssize_t i = 0; i < Py_ARRAY_LENGTH(richcompare_ops); i++) {
        int registry_oparg = PYJIT_EXT_BINARY_RICHCOMPARE_BASE +
                             richcompare_ops[i];
        for (Py_ssize_t j = 0;
             j < Py_ARRAY_LENGTH(richcompare_scalar_types); j++) {
            if (PyJit_RegisterBinaryProvider(
                    &PyArray_Type, richcompare_scalar_types[j],
                    registry_oparg, &numpy_jit_provider) < 0) {
                PyErr_Clear();
            }
            if (PyJit_RegisterBinaryProvider(
                    richcompare_scalar_types[j], &PyArray_Type,
                    registry_oparg, &numpy_jit_provider) < 0) {
                PyErr_Clear();
            }
        }
    }
    if (PyJit_RegisterBinaryProvider(
            &PyArray_Type, &PyArray_Type, NB_SUBSCR,
            &numpy_jit_provider) < 0) {
        PyErr_Clear();
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
            &PyLongArrType_Type,
            &PyLongLongArrType_Type,
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
        if (op != NUMPY_JIT_OP_MINIMUM &&
            op != NUMPY_JIT_OP_LOGICAL_AND &&
            find_candidate(op, NPY_DOUBLE, NPY_DOUBLE,
                           NUMPY_JIT_LAYOUT_C_CONTIG_SAME_SHAPE) == NULL) {
            continue;
        }
        if (PyJit_RegisterCallableProvider(callable, &numpy_jit_provider) < 0) {
            PyErr_Clear();
        }
    }
    PyObject *outer_descriptor = PyObject_GetAttrString(
        (PyObject *)&PyUFunc_Type, "outer");
    if (outer_descriptor != NULL) {
        if (PyJit_RegisterCallableProvider(
                outer_descriptor, &numpy_jit_provider) < 0) {
            PyErr_Clear();
        }
        Py_DECREF(outer_descriptor);
    }
    else {
        PyErr_Clear();
    }
    if (PyJit_RegisterCallableProvider(
            (PyObject *)&PyUFunc_Type, &numpy_jit_provider) < 0) {
        PyErr_Clear();
    }
    if (PyJit_RegisterUnaryProvider(
            &PyArray_Type, PYJIT_EXT_UNARY_NEGATIVE,
            &numpy_jit_provider) < 0) {
        PyErr_Clear();
    }
    return 0;
}
