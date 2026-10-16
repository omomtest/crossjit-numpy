/* Strict precompiled loops for Host-selected CrossJIT elementwise regions. */
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include <string.h>

#include "generated_crossjit_kernels.h"
#include "jit_kernel.h"
#include "numpy/npy_math.h"

/* Preserve NumPy minimum/maximum operand selection, including asymmetric NaN
 * and signed-zero behavior, instead of relying on platform fmin/fmax. */
static inline double
crossjit_min_double(double left, double right)
{
    return (left <= right || npy_isnan(left)) ? left : right;
}

static inline double
crossjit_max_double(double left, double right)
{
    return (left >= right || npy_isnan(left)) ? left : right;
}

static inline float
crossjit_min_float(float left, float right)
{
    return (left <= right || npy_isnan(left)) ? left : right;
}

static inline float
crossjit_max_float(float left, float right)
{
    return (left >= right || npy_isnan(left)) ? left : right;
}

NumpyCrossJitPolicyDecision
numpy_crossjit_generated_registry_lookup(
    const NumpyCrossJitKernelNode *nodes, uint8_t node_count,
    uint8_t input_count, uint16_t scalar_mask, uint8_t output_slot,
    uint64_t element_count, NumpyCrossJitFixedPlan *plan)
{
    if (plan == NULL) {
        return NUMPY_CROSSJIT_POLICY_ORIGINAL;
    }
    memset(plan, 0, sizeof(*plan));
    for (size_t policy_index = 0;
         policy_index < NPY_CROSSJIT_PRODUCTION_POLICY_COUNT;
         policy_index++) {
        const NumpyCrossJitGeneratedPolicy *policy =
            &numpy_crossjit_production_policies[policy_index];
        if (policy->input_count != input_count ||
            policy->node_count != node_count ||
            policy->scalar_mask != scalar_mask ||
            policy->output_slot != output_slot) {
            continue;
        }
        int matches = 1;
        for (uint8_t node_index = 0; node_index < node_count; node_index++) {
            const NumpyCrossJitGeneratedPolicyNode *expected =
                &policy->nodes[node_index];
            if (nodes[node_index].op != expected->op ||
                nodes[node_index].left != expected->left ||
                nodes[node_index].right != expected->right ||
                nodes[node_index].output != expected->output) {
                matches = 0;
                break;
            }
        }
        if (!matches) {
            continue;
        }
        for (uint8_t range_index = 0;
             range_index < policy->range_count; range_index++) {
            const NumpyCrossJitGeneratedPolicyRange *range =
                &policy->ranges[range_index];
            if (element_count < range->min_elements ||
                element_count > range->max_elements) {
                continue;
            }
            if (range->decision == NUMPY_CROSSJIT_POLICY_GENERATED &&
                    policy->entry != NULL) {
                plan->entry = policy->entry;
                plan->input_count = input_count;
                for (uint8_t i = 0; i < input_count; i++) {
                    plan->input_slots[i] = i;
                }
                return NUMPY_CROSSJIT_POLICY_GENERATED;
            }
            if (range->decision == NUMPY_CROSSJIT_POLICY_GENERIC) {
                return NUMPY_CROSSJIT_POLICY_GENERIC;
            }
            return NUMPY_CROSSJIT_POLICY_ORIGINAL;
        }
        return NUMPY_CROSSJIT_POLICY_ORIGINAL;
    }
    return NUMPY_CROSSJIT_POLICY_ORIGINAL;
}

NumpyCrossJitPolicyDecision
numpy_crossjit_profiled_iteration_registry_lookup(
    const NumpyCrossJitKernelNode *nodes, uint8_t node_count,
    uint8_t input_count, const NumpyCrossJitIterationDescriptor *descriptor,
    uint8_t output_slot, uint64_t element_count, uint64_t scalar_class_mask,
    int force_generated,
    NumpyCrossJitFixedPlan *plan)
{
    if (plan == NULL || descriptor == NULL ||
            input_count > NUMPY_CROSSJIT_MAX_FAST_INPUTS ||
            node_count > NUMPY_CROSSJIT_MAX_NODES ||
            descriptor->operand_count != input_count) {
        return NUMPY_CROSSJIT_POLICY_ORIGINAL;
    }
    memset(plan, 0, sizeof(*plan));
    for (size_t index = 0;
         index < NPY_CROSSJIT_PROFILED_ITERATION_KERNEL_COUNT; index++) {
        const NumpyCrossJitGeneratedIterationKernel *candidate =
            &numpy_crossjit_profiled_iteration_registry[index];
        if (candidate->input_count != input_count ||
                candidate->node_count != node_count ||
                candidate->output_slot != output_slot ||
                candidate->descriptor_version != descriptor->version ||
                candidate->iteration_rank_class !=
                    descriptor->iteration_rank_class ||
                candidate->outer_shape_class !=
                    descriptor->outer_shape_class ||
                memcmp(candidate->nodes, nodes,
                       (size_t)node_count * sizeof(nodes[0])) != 0) {
            continue;
        }
        int descriptor_match =
            descriptor->output.access_kind ==
                NUMPY_CROSSJIT_ACCESS_CONTIGUOUS &&
            descriptor->output.inner_stride == (npy_intp)sizeof(double) &&
            descriptor->output.outer_stride ==
                (descriptor->iteration_rank_class == NUMPY_CROSSJIT_ITERATION_FLAT ?
                    0 : (npy_intp)(descriptor->inner_extent * sizeof(double)));
        for (uint8_t slot = 0; descriptor_match && slot < input_count; slot++) {
            const NumpyCrossJitOperandDescriptor *operand =
                &descriptor->operands[slot];
            int access_match =
                operand->logical_input_slot == slot &&
                candidate->access_kinds[slot] == operand->access_kind &&
                candidate->broadcast_relations[slot] ==
                    operand->broadcast_relation;
            if (operand->access_kind == NUMPY_CROSSJIT_ACCESS_CONTIGUOUS) {
                access_match &=
                    operand->broadcast_relation ==
                        NUMPY_CROSSJIT_BROADCAST_NONE &&
                    operand->inner_stride == (npy_intp)sizeof(double);
            }
            else if (operand->access_kind == NUMPY_CROSSJIT_ACCESS_SCALAR) {
                access_match &=
                    operand->broadcast_relation ==
                        NUMPY_CROSSJIT_BROADCAST_SCALAR &&
                    operand->inner_stride == 0 && operand->outer_stride == 0;
            }
            else if (operand->access_kind ==
                     NUMPY_CROSSJIT_ACCESS_ZERO_STRIDE) {
                access_match &= operand->inner_stride == 0;
            }
            else if (operand->access_kind == NUMPY_CROSSJIT_ACCESS_FIXED_STRIDE) {
                access_match &= descriptor->version == 2 &&
                    candidate->affine_entry != NULL &&
                    operand->broadcast_relation == NUMPY_CROSSJIT_BROADCAST_NONE;
            }
            else {
                access_match = 0;
            }
            descriptor_match = access_match;
        }
        if (!descriptor_match) {
            continue;
        }
        NumpyCrossJitPolicyDecision decision =
            NUMPY_CROSSJIT_POLICY_ORIGINAL;
        int scalar_classes_required = 0;
        if (force_generated) {
            decision = NUMPY_CROSSJIT_POLICY_GENERATED;
            /* Calibration of exact value-class cells must include the same
             * integer conversion-class guard as measured production. Force
             * bypasses admission, never the cost/safety checks being measured. */
            scalar_classes_required = scalar_class_mask != 0;
        }
        else {
            /* A nonempty measured-cell list constrains every backend, not
             * just affine-v2. Equal element counts can have very different
             * row-loop overhead at different outer/inner factorizations. */
            /* Flat profiled artifacts require an explicit measured cell too.
             * Historical flat policies keep their separate registry; the
             * existence of a generated artifact grants no new admission. */
            int measured_shape = candidate->affine_entry == NULL &&
                candidate->flat_entry == NULL &&
                candidate->measured_cell_count == 0;
            for (uint8_t cell = 0; cell < candidate->measured_cell_count; cell++) {
                int match =
                    candidate->measured_outer_counts[cell] == descriptor->outer_count &&
                    candidate->measured_inner_extents[cell] == descriptor->inner_extent &&
                    (candidate->measured_scalar_classes[cell] == 0 ||
                     candidate->measured_scalar_classes[cell] == scalar_class_mask);
                measured_shape |= match;
                scalar_classes_required |= match &&
                    candidate->measured_scalar_classes[cell] != 0;
            }
            if (!measured_shape) {
                continue;
            }
            for (uint8_t range = 0; range < candidate->range_count; range++) {
                const NumpyCrossJitGeneratedPolicyRange *point =
                    &candidate->ranges[range];
                if (element_count >= point->min_elements &&
                        element_count <= point->max_elements) {
                    decision = (NumpyCrossJitPolicyDecision)point->decision;
                    break;
                }
            }
        }
        if (decision == NUMPY_CROSSJIT_POLICY_GENERATED) {
            plan->entry = candidate->flat_entry;
            plan->outer_inner_entry = candidate->entry;
            plan->affine_entry = candidate->affine_entry;
            plan->input_count = input_count;
            plan->scalar_classes_required = (uint8_t)scalar_classes_required;
            for (uint8_t slot = 0; slot < input_count; slot++) {
                plan->input_slots[slot] = slot;
            }
        }
        return decision;
    }
    return NUMPY_CROSSJIT_POLICY_ORIGINAL;
}

/* Execution synthesis is owned by crossjit-generator. */
#include "generated_crossjit_loops.h"

/* Validates the compact program envelope and dispatches by guarded dtype. */
int
numpy_crossjit_kernel_strided(
    const NumpyCrossJitKernelNode *nodes, uint8_t node_count,
    uint8_t input_count, uint16_t scalar_mask, uint8_t output_slot, int typenum,
    const void *const *inputs,
    const npy_intp input_strides[][NPY_MAXDIMS], uint8_t ndim,
    const npy_intp *dimensions, void *output, npy_intp size)
{
    if (nodes == NULL || inputs == NULL || input_strides == NULL ||
        dimensions == NULL || ndim > NPY_MAXDIMS ||
        (output == NULL && size != 0) ||
        node_count == 0 ||
        node_count > NUMPY_CROSSJIT_MAX_NODES || input_count == 0 ||
        input_count > NUMPY_CROSSJIT_MAX_INPUTS ||
        output_slot >= NUMPY_CROSSJIT_MAX_SLOTS) {
        return 0;
    }
    if (typenum == NPY_DOUBLE) {
        return cj_kernel_double(nodes, node_count, input_count, scalar_mask,
                                      output_slot, inputs, input_strides,
                                      ndim, dimensions, output, size);
    }
    if (typenum == NPY_FLOAT) {
        return cj_kernel_float(nodes, node_count, input_count, scalar_mask,
                                     output_slot, inputs, input_strides,
                                     ndim, dimensions, output, size);
    }
    if (typenum == NPY_INT && sizeof(npy_int) == sizeof(npy_uint32)) {
        return cj_kernel_uint32(nodes, node_count, input_count, scalar_mask,
                                      output_slot, inputs, input_strides,
                                      ndim, dimensions, output, size);
    }
    if ((typenum == NPY_LONG && sizeof(npy_long) == sizeof(npy_uint64)) ||
        (typenum == NPY_LONGLONG && sizeof(npy_longlong) == sizeof(npy_uint64))) {
        return cj_kernel_uint64(nodes, node_count, input_count, scalar_mask,
                                      output_slot, inputs, input_strides,
                                      ndim, dimensions, output, size);
    }
    return 0;
}
