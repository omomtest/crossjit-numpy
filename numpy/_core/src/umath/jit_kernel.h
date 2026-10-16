#ifndef NUMPY_CORE_SRC_UMATH_JIT_KERNEL_H
#define NUMPY_CORE_SRC_UMATH_JIT_KERNEL_H

#include "numpy/arrayobject.h"
#include "generated_crossjit_execution.h"

_Static_assert(NPY_MAXDIMS == CJ_MAX_RANK, "iteration ABI rank mismatch");

#define NUMPY_CROSSJIT_MAX_NODES CJ_MAX_NODES
#define NUMPY_CROSSJIT_MAX_INPUTS CJ_MAX_INPUTS
#define NUMPY_CROSSJIT_MAX_SLOTS \
    (NUMPY_CROSSJIT_MAX_INPUTS + NUMPY_CROSSJIT_MAX_NODES)

/* Provider-owned iteration projection shared with descriptor-driven generated
 * registry lookup.  These values are not part of the CPython Host ABI. */
#define NUMPY_CROSSJIT_ITERATION_DESCRIPTOR_VERSION 1

typedef enum {
    NUMPY_CROSSJIT_ITERATION_FLAT = 1,
    NUMPY_CROSSJIT_ITERATION_OUTER_INNER = 2,
} NumpyCrossJitIterationRankClass;

typedef enum {
    NUMPY_CROSSJIT_OUTER_NONE = 0,
    NUMPY_CROSSJIT_OUTER_AFFINE_ROWS = 1,
    NUMPY_CROSSJIT_OUTER_GENERAL = 2,
} NumpyCrossJitOuterShapeClass;

typedef enum {
    NUMPY_CROSSJIT_BROADCAST_NONE = 0,
    NUMPY_CROSSJIT_BROADCAST_SCALAR = 1,
    NUMPY_CROSSJIT_BROADCAST_INNER = 2,
    NUMPY_CROSSJIT_BROADCAST_OUTER = 3,
    NUMPY_CROSSJIT_BROADCAST_MULTIPLE = 4,
} NumpyCrossJitBroadcastRelation;

typedef enum {
    NUMPY_CROSSJIT_ACCESS_CONTIGUOUS = 1,
    NUMPY_CROSSJIT_ACCESS_SCALAR = 2,
    NUMPY_CROSSJIT_ACCESS_ZERO_STRIDE = 3,
    NUMPY_CROSSJIT_ACCESS_FIXED_STRIDE = 4,
    NUMPY_CROSSJIT_ACCESS_GENERAL_STRIDED = 5,
} NumpyCrossJitProfileAccessKind;

typedef struct {
    uint8_t logical_input_slot;
    uint8_t access_kind;
    uint8_t broadcast_relation;
    npy_intp inner_stride;
    npy_intp outer_stride;
} NumpyCrossJitOperandDescriptor;

typedef struct {
    uint8_t access_kind;
    npy_intp inner_stride;
    npy_intp outer_stride;
} NumpyCrossJitOutputDescriptor;

typedef struct {
    uint8_t version;
    uint8_t iteration_rank_class;
    uint8_t outer_shape_class;
    uint8_t operand_count;
    uint64_t outer_count;
    uint64_t inner_extent;
    NumpyCrossJitOperandDescriptor operands[NUMPY_CROSSJIT_MAX_INPUTS];
    NumpyCrossJitOutputDescriptor output;
} NumpyCrossJitIterationDescriptor;

/* Provider semantic projection binds to Generator primitive opcodes. */
typedef CjPrimitiveOpcode NumpyCrossJitKernelOp;
#define NUMPY_CROSSJIT_KERNEL_ADD CJ_OP_ADD
#define NUMPY_CROSSJIT_KERNEL_SUBTRACT CJ_OP_SUBTRACT
#define NUMPY_CROSSJIT_KERNEL_MULTIPLY CJ_OP_MULTIPLY
#define NUMPY_CROSSJIT_KERNEL_TRUE_DIVIDE CJ_OP_TRUE_DIVIDE
#define NUMPY_CROSSJIT_KERNEL_NEGATIVE CJ_OP_NEGATIVE
#define NUMPY_CROSSJIT_KERNEL_MINIMUM CJ_OP_MINIMUM
#define NUMPY_CROSSJIT_KERNEL_MAXIMUM CJ_OP_MAXIMUM
typedef CjKernelNode NumpyCrossJitKernelNode;

#define NUMPY_CROSSJIT_MAX_FAST_INPUTS NUMPY_CROSSJIT_MAX_INPUTS

typedef void (*NumpyCrossJitDoubleKernelEntry)(
    const void *const *inputs, double *restrict output, size_t size);
typedef void (*NumpyCrossJitDoubleOuterInnerKernelEntry)(
    const void *const *inputs, const ptrdiff_t *outer_strides,
    double *restrict output, size_t outer_count, size_t inner_extent);
typedef void (*NumpyCrossJitDoubleAffineKernelEntry)(
    const void *const *inputs, const ptrdiff_t *inner_strides,
    const ptrdiff_t *outer_strides, double *restrict output,
    size_t outer_count, size_t inner_extent);

/* A prepare-time-selected straight-line loop and its compact operand roles. */
typedef struct {
    NumpyCrossJitDoubleKernelEntry entry;
    NumpyCrossJitDoubleOuterInnerKernelEntry outer_inner_entry;
    NumpyCrossJitDoubleAffineKernelEntry affine_entry;
    uint8_t input_slots[NUMPY_CROSSJIT_MAX_FAST_INPUTS];
    uint8_t input_count;
    uint8_t scalar_classes_required;
} NumpyCrossJitFixedPlan;

typedef enum {
    NUMPY_CROSSJIT_POLICY_ORIGINAL = 0,
    NUMPY_CROSSJIT_POLICY_GENERATED = 1,
    NUMPY_CROSSJIT_POLICY_GENERIC = 2,
} NumpyCrossJitPolicyDecision;

NumpyCrossJitPolicyDecision numpy_crossjit_generated_registry_lookup(
    const NumpyCrossJitKernelNode *nodes, uint8_t node_count,
    uint8_t input_count, uint16_t scalar_mask, uint8_t output_slot,
    uint64_t element_count, NumpyCrossJitFixedPlan *plan);

NumpyCrossJitPolicyDecision numpy_crossjit_profiled_iteration_registry_lookup(
    const NumpyCrossJitKernelNode *nodes, uint8_t node_count,
    uint8_t input_count, const NumpyCrossJitIterationDescriptor *descriptor,
    uint8_t output_slot, uint64_t element_count, uint64_t scalar_class_mask,
    int force_generated,
    NumpyCrossJitFixedPlan *plan);

/* Executes a validated two-to-twelve-node DAG over one strided iteration domain.
 * Returns zero without publishing partial output when the program is invalid. */
int numpy_crossjit_kernel_strided(
    const NumpyCrossJitKernelNode *nodes, uint8_t node_count,
    uint8_t input_count, uint16_t scalar_mask, uint8_t output_slot, int typenum,
    const void *const *inputs,
    const npy_intp input_strides[][NPY_MAXDIMS], uint8_t ndim,
    const npy_intp *dimensions, void *output, npy_intp size);

#endif
