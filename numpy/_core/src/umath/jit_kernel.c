/* Strict precompiled loops for Host-selected CrossJIT elementwise regions. */
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
    uint8_t output_slot, uint64_t element_count, int force_generated,
    NumpyCrossJitFixedPlan *plan)
{
    if (plan == NULL || descriptor == NULL ||
            input_count > NUMPY_CROSSJIT_MAX_FAST_INPUTS || node_count > 6 ||
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
                (npy_intp)(descriptor->inner_extent * sizeof(double));
        for (uint8_t slot = 0; descriptor_match && slot < input_count; slot++) {
            const NumpyCrossJitOperandDescriptor *operand =
                &descriptor->operands[slot];
            descriptor_match =
                operand->logical_input_slot == slot &&
                candidate->access_kinds[slot] == operand->access_kind &&
                operand->access_kind == NUMPY_CROSSJIT_ACCESS_CONTIGUOUS &&
                operand->broadcast_relation == NUMPY_CROSSJIT_BROADCAST_NONE &&
                operand->inner_stride == (npy_intp)sizeof(double);
        }
        if (!descriptor_match) {
            continue;
        }
        NumpyCrossJitPolicyDecision decision =
            NUMPY_CROSSJIT_POLICY_ORIGINAL;
        if (force_generated) {
            decision = NUMPY_CROSSJIT_POLICY_GENERATED;
        }
        else {
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
            plan->outer_inner_entry = candidate->entry;
            plan->input_count = input_count;
            for (uint8_t slot = 0; slot < input_count; slot++) {
                plan->input_slots[slot] = slot;
            }
        }
        return decision;
    }
    return NUMPY_CROSSJIT_POLICY_ORIGINAL;
}

/*
 * Executes the two dominant NPBench shapes without interpreting a node
 * switch for every element.  The first node must add two live-ins and the
 * second must add or multiply that virtual result by a third live-in.
 */
#define CROSSJIT_LINEAR_PAIR_CASES(FIRST_OPERATOR, SECOND_OPERATOR)         \
    switch (linear_scalar_mask) {                                           \
        case 0:                                                             \
            for (npy_intp i = 0; i < pair_size; i++)                       \
                pair_out[i] = (first_data[i] FIRST_OPERATOR second_data[i]) SECOND_OPERATOR \
                         third_data[i];                                     \
            break;                                                          \
        case 1:                                                             \
            for (npy_intp i = 0; i < pair_size; i++)                       \
                pair_out[i] = (first_data[0] FIRST_OPERATOR second_data[i]) SECOND_OPERATOR \
                         third_data[i];                                     \
            break;                                                          \
        case 2:                                                             \
            for (npy_intp i = 0; i < pair_size; i++)                       \
                pair_out[i] = (first_data[i] FIRST_OPERATOR second_data[0]) SECOND_OPERATOR \
                         third_data[i];                                     \
            break;                                                          \
        case 3:                                                             \
            for (npy_intp i = 0; i < pair_size; i++)                       \
                pair_out[i] = (first_data[0] FIRST_OPERATOR second_data[0]) SECOND_OPERATOR \
                         third_data[i];                                     \
            break;                                                          \
        case 4:                                                             \
            for (npy_intp i = 0; i < pair_size; i++)                       \
                pair_out[i] = (first_data[i] FIRST_OPERATOR second_data[i]) SECOND_OPERATOR \
                         third_data[0];                                     \
            break;                                                          \
        case 5:                                                             \
            for (npy_intp i = 0; i < pair_size; i++)                       \
                pair_out[i] = (first_data[0] FIRST_OPERATOR second_data[i]) SECOND_OPERATOR \
                         third_data[0];                                     \
            break;                                                          \
        case 6:                                                             \
            for (npy_intp i = 0; i < pair_size; i++)                       \
                pair_out[i] = (first_data[i] FIRST_OPERATOR second_data[0]) SECOND_OPERATOR \
                         third_data[0];                                     \
            break;                                                          \
        default:                                                            \
            for (npy_intp i = 0; i < pair_size; i++)                       \
                pair_out[i] = (first_data[0] FIRST_OPERATOR second_data[0]) SECOND_OPERATOR \
                         third_data[0];                                     \
            break;                                                          \
    }

#define CROSSJIT_DISPATCH_LINEAR_PAIR()                                     \
    do {                                                                    \
        if (pair_kind == 1) {                                               \
            if (nodes[1].op == NUMPY_CROSSJIT_KERNEL_ADD) {                \
                CROSSJIT_LINEAR_PAIR_CASES(+, +);                           \
            }                                                               \
            else {                                                          \
                CROSSJIT_LINEAR_PAIR_CASES(+, *);                           \
            }                                                               \
        }                                                                   \
        else if (pair_kind == 2) {                                          \
            CROSSJIT_LINEAR_PAIR_CASES(*, -);                               \
        }                                                                   \
        else {                                                              \
            CROSSJIT_LINEAR_PAIR_CASES(-, *);                               \
        }                                                                   \
    } while (0)

/*
 * These fixed-index evaluators let the compiler produce separate unrolled
 * loop bodies for each supported two-to-six-node provider program.  Node
 * operation dispatch remains strict NumPy lowering metadata; there is no
 * Host-specific pattern knowledge in this execution layer.
 */
#define CROSSJIT_EVAL_FLOAT_NODE(TYPE, NAME, INDEX)                         \
    do {                                                                    \
        const NumpyCrossJitKernelNode *node = &nodes[INDEX];                \
        TYPE left = slots[node->left];                                      \
        TYPE right = slots[node->right];                                    \
        switch (node->op) {                                                 \
            case NUMPY_CROSSJIT_KERNEL_ADD:                                \
                slots[node->output] = left + right;                         \
                break;                                                      \
            case NUMPY_CROSSJIT_KERNEL_SUBTRACT:                           \
                slots[node->output] = left - right;                         \
                break;                                                      \
            case NUMPY_CROSSJIT_KERNEL_MULTIPLY:                           \
                slots[node->output] = left * right;                         \
                break;                                                      \
            case NUMPY_CROSSJIT_KERNEL_TRUE_DIVIDE:                        \
                slots[node->output] = left / right;                         \
                break;                                                      \
            case NUMPY_CROSSJIT_KERNEL_NEGATIVE:                           \
                slots[node->output] = -left;                               \
                break;                                                      \
            case NUMPY_CROSSJIT_KERNEL_MINIMUM:                            \
                slots[node->output] = crossjit_min_##NAME(left, right);    \
                break;                                                      \
            case NUMPY_CROSSJIT_KERNEL_MAXIMUM:                            \
                slots[node->output] = crossjit_max_##NAME(left, right);    \
                break;                                                      \
            default:                                                        \
                return 0;                                                   \
        }                                                                   \
    } while (0)

#define CROSSJIT_EVAL_INTEGER_NODE(TYPE, INDEX)                             \
    do {                                                                    \
        const NumpyCrossJitKernelNode *node = &nodes[INDEX];                \
        TYPE left = slots[node->left];                                      \
        TYPE right = slots[node->right];                                    \
        switch (node->op) {                                                 \
            case NUMPY_CROSSJIT_KERNEL_ADD:                                \
                slots[node->output] = left + right;                         \
                break;                                                      \
            case NUMPY_CROSSJIT_KERNEL_SUBTRACT:                           \
                slots[node->output] = left - right;                         \
                break;                                                      \
            case NUMPY_CROSSJIT_KERNEL_MULTIPLY:                           \
                slots[node->output] = left * right;                         \
                break;                                                      \
            case NUMPY_CROSSJIT_KERNEL_NEGATIVE:                           \
                slots[node->output] = (TYPE)(0 - left);                    \
                break;                                                      \
            default:                                                        \
                return 0;                                                   \
        }                                                                   \
    } while (0)

/*
 * Contiguous inner dimensions are evaluated eight elements at a time.  The
 * operation switch is therefore paid once per node and chunk, rather than
 * once per node and element, while every lane retains the original node
 * evaluation order.  The fixed node-count switch provides five precompiled
 * loop shapes and deliberately stops at the provider's six-node limit.
 */
#define CROSSJIT_EVAL_FLOAT_CHUNK_NODE(TYPE, NAME, INDEX)                   \
    do {                                                                    \
        const NumpyCrossJitKernelNode *node = &nodes[INDEX];                \
        switch (node->op) {                                                 \
            case NUMPY_CROSSJIT_KERNEL_ADD:                                \
                for (int lane = 0; lane < lanes; lane++)                   \
                    slots[node->output][lane] = slots[node->left][lane] +   \
                        slots[node->right][lane];                           \
                break;                                                      \
            case NUMPY_CROSSJIT_KERNEL_SUBTRACT:                           \
                for (int lane = 0; lane < lanes; lane++)                   \
                    slots[node->output][lane] = slots[node->left][lane] -   \
                        slots[node->right][lane];                           \
                break;                                                      \
            case NUMPY_CROSSJIT_KERNEL_MULTIPLY:                           \
                for (int lane = 0; lane < lanes; lane++)                   \
                    slots[node->output][lane] = slots[node->left][lane] *   \
                        slots[node->right][lane];                           \
                break;                                                      \
            case NUMPY_CROSSJIT_KERNEL_TRUE_DIVIDE:                        \
                for (int lane = 0; lane < lanes; lane++)                   \
                    slots[node->output][lane] = slots[node->left][lane] /   \
                        slots[node->right][lane];                           \
                break;                                                      \
            case NUMPY_CROSSJIT_KERNEL_NEGATIVE:                           \
                for (int lane = 0; lane < lanes; lane++)                   \
                    slots[node->output][lane] = -slots[node->left][lane];  \
                break;                                                      \
            case NUMPY_CROSSJIT_KERNEL_MINIMUM:                            \
                for (int lane = 0; lane < lanes; lane++)                   \
                    slots[node->output][lane] = crossjit_min_##NAME(       \
                        slots[node->left][lane], slots[node->right][lane]); \
                break;                                                      \
            case NUMPY_CROSSJIT_KERNEL_MAXIMUM:                            \
                for (int lane = 0; lane < lanes; lane++)                   \
                    slots[node->output][lane] = crossjit_max_##NAME(       \
                        slots[node->left][lane], slots[node->right][lane]); \
                break;                                                      \
            default:                                                        \
                return 0;                                                   \
        }                                                                   \
    } while (0)

#define CROSSJIT_EVAL_INTEGER_CHUNK_NODE(TYPE, INDEX)                       \
    do {                                                                    \
        const NumpyCrossJitKernelNode *node = &nodes[INDEX];                \
        switch (node->op) {                                                 \
            case NUMPY_CROSSJIT_KERNEL_ADD:                                \
                for (int lane = 0; lane < lanes; lane++)                   \
                    slots[node->output][lane] = slots[node->left][lane] +   \
                        slots[node->right][lane];                           \
                break;                                                      \
            case NUMPY_CROSSJIT_KERNEL_SUBTRACT:                           \
                for (int lane = 0; lane < lanes; lane++)                   \
                    slots[node->output][lane] = slots[node->left][lane] -   \
                        slots[node->right][lane];                           \
                break;                                                      \
            case NUMPY_CROSSJIT_KERNEL_MULTIPLY:                           \
                for (int lane = 0; lane < lanes; lane++)                   \
                    slots[node->output][lane] = slots[node->left][lane] *   \
                        slots[node->right][lane];                           \
                break;                                                      \
            case NUMPY_CROSSJIT_KERNEL_NEGATIVE:                           \
                for (int lane = 0; lane < lanes; lane++)                   \
                    slots[node->output][lane] =                            \
                        (TYPE)(0 - slots[node->left][lane]);                \
                break;                                                      \
            default:                                                        \
                return 0;                                                   \
        }                                                                   \
    } while (0)

#define CROSSJIT_FIXED_FLOAT_CHUNK_NODES(TYPE, NAME)                        \
    switch (node_count) {                                                   \
        case 6: CROSSJIT_EVAL_FLOAT_CHUNK_NODE(TYPE, NAME, 0);             \
                CROSSJIT_EVAL_FLOAT_CHUNK_NODE(TYPE, NAME, 1);             \
                CROSSJIT_EVAL_FLOAT_CHUNK_NODE(TYPE, NAME, 2);             \
                CROSSJIT_EVAL_FLOAT_CHUNK_NODE(TYPE, NAME, 3);             \
                CROSSJIT_EVAL_FLOAT_CHUNK_NODE(TYPE, NAME, 4);             \
                CROSSJIT_EVAL_FLOAT_CHUNK_NODE(TYPE, NAME, 5); break;      \
        case 5: CROSSJIT_EVAL_FLOAT_CHUNK_NODE(TYPE, NAME, 0);             \
                CROSSJIT_EVAL_FLOAT_CHUNK_NODE(TYPE, NAME, 1);             \
                CROSSJIT_EVAL_FLOAT_CHUNK_NODE(TYPE, NAME, 2);             \
                CROSSJIT_EVAL_FLOAT_CHUNK_NODE(TYPE, NAME, 3);             \
                CROSSJIT_EVAL_FLOAT_CHUNK_NODE(TYPE, NAME, 4); break;      \
        case 4: CROSSJIT_EVAL_FLOAT_CHUNK_NODE(TYPE, NAME, 0);             \
                CROSSJIT_EVAL_FLOAT_CHUNK_NODE(TYPE, NAME, 1);             \
                CROSSJIT_EVAL_FLOAT_CHUNK_NODE(TYPE, NAME, 2);             \
                CROSSJIT_EVAL_FLOAT_CHUNK_NODE(TYPE, NAME, 3); break;      \
        case 3: CROSSJIT_EVAL_FLOAT_CHUNK_NODE(TYPE, NAME, 0);             \
                CROSSJIT_EVAL_FLOAT_CHUNK_NODE(TYPE, NAME, 1);             \
                CROSSJIT_EVAL_FLOAT_CHUNK_NODE(TYPE, NAME, 2); break;      \
        case 2: CROSSJIT_EVAL_FLOAT_CHUNK_NODE(TYPE, NAME, 0);             \
                CROSSJIT_EVAL_FLOAT_CHUNK_NODE(TYPE, NAME, 1); break;      \
        default: return 0;                                                  \
    }

#define CROSSJIT_FIXED_INTEGER_CHUNK_NODES(TYPE)                            \
    switch (node_count) {                                                   \
        case 6: CROSSJIT_EVAL_INTEGER_CHUNK_NODE(TYPE, 0);                 \
                CROSSJIT_EVAL_INTEGER_CHUNK_NODE(TYPE, 1);                 \
                CROSSJIT_EVAL_INTEGER_CHUNK_NODE(TYPE, 2);                 \
                CROSSJIT_EVAL_INTEGER_CHUNK_NODE(TYPE, 3);                 \
                CROSSJIT_EVAL_INTEGER_CHUNK_NODE(TYPE, 4);                 \
                CROSSJIT_EVAL_INTEGER_CHUNK_NODE(TYPE, 5); break;          \
        case 5: CROSSJIT_EVAL_INTEGER_CHUNK_NODE(TYPE, 0);                 \
                CROSSJIT_EVAL_INTEGER_CHUNK_NODE(TYPE, 1);                 \
                CROSSJIT_EVAL_INTEGER_CHUNK_NODE(TYPE, 2);                 \
                CROSSJIT_EVAL_INTEGER_CHUNK_NODE(TYPE, 3);                 \
                CROSSJIT_EVAL_INTEGER_CHUNK_NODE(TYPE, 4); break;          \
        case 4: CROSSJIT_EVAL_INTEGER_CHUNK_NODE(TYPE, 0);                 \
                CROSSJIT_EVAL_INTEGER_CHUNK_NODE(TYPE, 1);                 \
                CROSSJIT_EVAL_INTEGER_CHUNK_NODE(TYPE, 2);                 \
                CROSSJIT_EVAL_INTEGER_CHUNK_NODE(TYPE, 3); break;          \
        case 3: CROSSJIT_EVAL_INTEGER_CHUNK_NODE(TYPE, 0);                 \
                CROSSJIT_EVAL_INTEGER_CHUNK_NODE(TYPE, 1);                 \
                CROSSJIT_EVAL_INTEGER_CHUNK_NODE(TYPE, 2); break;          \
        case 2: CROSSJIT_EVAL_INTEGER_CHUNK_NODE(TYPE, 0);                 \
                CROSSJIT_EVAL_INTEGER_CHUNK_NODE(TYPE, 1); break;          \
        default: return 0;                                                  \
    }

/* Instantiate fixed-width chunk interpreters for each supported storage type. */
#define DEFINE_FLOAT_CHUNK_KERNEL(TYPE, NAME)                               \
static int                                                                 \
crossjit_chunk_##NAME(                                                      \
    const NumpyCrossJitKernelNode *nodes, uint8_t node_count,               \
    uint8_t input_count, uint8_t output_slot, const void *const *inputs,    \
    const npy_intp *strides, TYPE *out, npy_intp size)                     \
{                                                                           \
    TYPE slots[NUMPY_CROSSJIT_MAX_SLOTS][8];                               \
    for (npy_intp base = 0; base < size; base += 8) {                      \
        int lanes = (int)((size - base) < 8 ? size - base : 8);           \
        for (uint8_t input = 0; input < input_count; input++) {            \
            const char *source = (const char *)inputs[input];              \
            for (int lane = 0; lane < lanes; lane++) {                     \
                slots[input][lane] = *(const TYPE *)(source +              \
                    (base + lane) * strides[input]);                       \
            }                                                               \
        }                                                                   \
        CROSSJIT_FIXED_FLOAT_CHUNK_NODES(TYPE, NAME);                       \
        for (int lane = 0; lane < lanes; lane++) {                         \
            out[base + lane] = slots[output_slot][lane];                   \
        }                                                                   \
    }                                                                       \
    return 1;                                                               \
}

#define DEFINE_INTEGER_CHUNK_KERNEL(TYPE, NAME)                             \
static int                                                                 \
crossjit_chunk_##NAME(                                                      \
    const NumpyCrossJitKernelNode *nodes, uint8_t node_count,               \
    uint8_t input_count, uint8_t output_slot, const void *const *inputs,    \
    const npy_intp *strides, TYPE *out, npy_intp size)                     \
{                                                                           \
    TYPE slots[NUMPY_CROSSJIT_MAX_SLOTS][8];                               \
    for (npy_intp base = 0; base < size; base += 8) {                      \
        int lanes = (int)((size - base) < 8 ? size - base : 8);           \
        for (uint8_t input = 0; input < input_count; input++) {            \
            const char *source = (const char *)inputs[input];              \
            for (int lane = 0; lane < lanes; lane++) {                     \
                slots[input][lane] = *(const TYPE *)(source +              \
                    (base + lane) * strides[input]);                       \
            }                                                               \
        }                                                                   \
        CROSSJIT_FIXED_INTEGER_CHUNK_NODES(TYPE);                           \
        for (int lane = 0; lane < lanes; lane++) {                         \
            out[base + lane] = slots[output_slot][lane];                   \
        }                                                                   \
    }                                                                       \
    return 1;                                                               \
}

DEFINE_FLOAT_CHUNK_KERNEL(double, double)
DEFINE_FLOAT_CHUNK_KERNEL(float, float)
DEFINE_INTEGER_CHUNK_KERNEL(npy_uint32, uint32)
DEFINE_INTEGER_CHUNK_KERNEL(npy_uint64, uint64)

#define CROSSJIT_TRY_PAIR_FAST_PATH(TYPE)                                   \
    do {                                                                    \
        int pair_kind = 0;                                                  \
        if (node_count == 2 &&                                              \
            nodes[0].left < input_count && nodes[0].right < input_count && \
            nodes[0].output == input_count) {                              \
            uint8_t third;                                                  \
            int virtual_on_left = 0;                                       \
            if (nodes[1].left == nodes[0].output &&                        \
                nodes[1].right < input_count) {                            \
                third = nodes[1].right;                                    \
                virtual_on_left = 1;                                       \
            }                                                               \
            else if (nodes[1].right == nodes[0].output &&                  \
                     nodes[1].left < input_count) {                         \
                third = nodes[1].left;                                     \
            }                                                               \
            else {                                                          \
                break;                                                      \
            }                                                               \
            if (nodes[0].op == NUMPY_CROSSJIT_KERNEL_ADD &&                \
                (nodes[1].op == NUMPY_CROSSJIT_KERNEL_ADD ||               \
                 nodes[1].op == NUMPY_CROSSJIT_KERNEL_MULTIPLY)) {         \
                pair_kind = 1;                                             \
            }                                                               \
            else if (nodes[0].op == NUMPY_CROSSJIT_KERNEL_MULTIPLY &&      \
                     nodes[1].op == NUMPY_CROSSJIT_KERNEL_SUBTRACT &&      \
                     virtual_on_left) {                                    \
                pair_kind = 2;                                             \
            }                                                               \
            else if (nodes[0].op == NUMPY_CROSSJIT_KERNEL_SUBTRACT &&      \
                     nodes[1].op == NUMPY_CROSSJIT_KERNEL_MULTIPLY) {       \
                pair_kind = 3;                                             \
            }                                                               \
            if (pair_kind == 0) {                                          \
                break;                                                      \
            }                                                               \
            npy_intp first_stride = input_strides[nodes[0].left][0];       \
            npy_intp second_stride = input_strides[nodes[0].right][0];     \
            npy_intp third_stride = input_strides[third][0];                \
            if (ndim == 1 &&                                                \
                (first_stride == 0 || first_stride == (npy_intp)sizeof(TYPE)) && \
                (second_stride == 0 || second_stride == (npy_intp)sizeof(TYPE)) && \
                (third_stride == 0 || third_stride == (npy_intp)sizeof(TYPE))) { \
                const TYPE *first_data = inputs[nodes[0].left];             \
                const TYPE *second_data = inputs[nodes[0].right];           \
                const TYPE *third_data = inputs[third];                     \
                uint8_t linear_scalar_mask =                                \
                    (first_stride == 0) | ((second_stride == 0) << 1) |     \
                    ((third_stride == 0) << 2);                             \
                TYPE *pair_out = out;                                      \
                npy_intp pair_size = size;                                 \
                CROSSJIT_DISPATCH_LINEAR_PAIR();                            \
                return 1;                                                   \
            }                                                               \
            int inner_axis = (int)ndim - 1;                                \
            if (ndim > 1 &&                                                 \
                (input_strides[nodes[0].left][inner_axis] == 0 ||          \
                 input_strides[nodes[0].left][inner_axis] ==               \
                    (npy_intp)sizeof(TYPE)) &&                              \
                (input_strides[nodes[0].right][inner_axis] == 0 ||         \
                 input_strides[nodes[0].right][inner_axis] ==              \
                    (npy_intp)sizeof(TYPE)) &&                              \
                (input_strides[third][inner_axis] == 0 ||                  \
                 input_strides[third][inner_axis] ==                       \
                    (npy_intp)sizeof(TYPE))) {                              \
                npy_intp pair_size = dimensions[inner_axis];               \
                npy_intp outer_size = pair_size == 0 ? 0 : size / pair_size; \
                uint8_t linear_scalar_mask =                               \
                    (input_strides[nodes[0].left][inner_axis] == 0) |      \
                    ((input_strides[nodes[0].right][inner_axis] == 0) << 1) | \
                    ((input_strides[third][inner_axis] == 0) << 2);        \
                for (npy_intp outer = 0; outer < outer_size; outer++) {    \
                    const TYPE *first_data = (const TYPE *)((const char *) \
                        inputs[nodes[0].left] + offsets[nodes[0].left]);   \
                    const TYPE *second_data = (const TYPE *)((const char *)\
                        inputs[nodes[0].right] + offsets[nodes[0].right]); \
                    const TYPE *third_data = (const TYPE *)((const char *) \
                        inputs[third] + offsets[third]);                    \
                    TYPE *pair_out = out + outer * pair_size;              \
                    CROSSJIT_DISPATCH_LINEAR_PAIR();                        \
                    for (int axis = inner_axis - 1; axis >= 0; axis--) {   \
                        coordinates[axis]++;                               \
                        for (uint8_t input = 0; input < input_count; input++) { \
                            offsets[input] += input_strides[input][axis];   \
                        }                                                   \
                        if (coordinates[axis] < dimensions[axis]) {        \
                            break;                                         \
                        }                                                   \
                        coordinates[axis] = 0;                             \
                        for (uint8_t input = 0; input < input_count; input++) { \
                            offsets[input] -= dimensions[axis] *           \
                                input_strides[input][axis];                \
                        }                                                   \
                    }                                                       \
                }                                                           \
                return 1;                                                   \
            }                                                               \
            for (npy_intp index = 0; index < size; index++) {              \
                TYPE left = *(const TYPE *)((const char *)                 \
                    inputs[nodes[0].left] + offsets[nodes[0].left]);       \
                TYPE right = *(const TYPE *)((const char *)                \
                    inputs[nodes[0].right] + offsets[nodes[0].right]);     \
                TYPE tail = *(const TYPE *)((const char *)                 \
                    inputs[third] + offsets[third]);                        \
                TYPE sum = left + right;                                   \
                out[index] = nodes[1].op == NUMPY_CROSSJIT_KERNEL_ADD ?   \
                    sum + tail : sum * tail;                               \
                if (ndim == 1) {                                           \
                    for (uint8_t input = 0; input < input_count; input++) {\
                        offsets[input] += input_strides[input][0];          \
                    }                                                       \
                    continue;                                              \
                }                                                           \
                for (int axis = (int)ndim - 1; axis >= 0; axis--) {        \
                    coordinates[axis]++;                                   \
                    for (uint8_t input = 0; input < input_count; input++) {\
                        offsets[input] += input_strides[input][axis];       \
                    }                                                       \
                    if (coordinates[axis] < dimensions[axis]) {            \
                        break;                                             \
                    }                                                       \
                    coordinates[axis] = 0;                                 \
                    for (uint8_t input = 0; input < input_count; input++) {\
                        offsets[input] -= dimensions[axis] *               \
                            input_strides[input][axis];                    \
                    }                                                       \
                }                                                           \
            }                                                               \
            return 1;                                                       \
        }                                                                   \
    } while (0)

/* Dispatches contiguous inner dimensions to the provider's chunked loops. */
#define CROSSJIT_TRY_CHUNK_FAST_PATH(TYPE, NAME)                            \
    do {                                                                    \
        int inner_axis = (int)ndim - 1;                                    \
        int chunkable = inner_axis >= 0;                                   \
        npy_intp inner_strides[NUMPY_CROSSJIT_MAX_INPUTS];                 \
        const void *row_inputs[NUMPY_CROSSJIT_MAX_INPUTS];                 \
        for (uint8_t input = 0; input < input_count && chunkable; input++) { \
            inner_strides[input] = input_strides[input][inner_axis];       \
            if (inner_strides[input] != 0 &&                               \
                inner_strides[input] != (npy_intp)sizeof(TYPE)) {          \
                chunkable = 0;                                             \
            }                                                               \
        }                                                                   \
        if (!chunkable) {                                                   \
            break;                                                          \
        }                                                                   \
        if (ndim == 1) {                                                    \
            return crossjit_chunk_##NAME(                                  \
                nodes, node_count, input_count, output_slot, inputs,       \
                inner_strides, out, size);                                 \
        }                                                                   \
        npy_intp inner_size = dimensions[inner_axis];                      \
        if (inner_size == 0) {                                              \
            return 1;                                                       \
        }                                                                   \
        npy_intp outer_size = size / inner_size;                           \
        for (npy_intp outer = 0; outer < outer_size; outer++) {            \
            for (uint8_t input = 0; input < input_count; input++) {        \
                row_inputs[input] = (const char *)inputs[input] +          \
                    offsets[input];                                        \
            }                                                               \
            if (!crossjit_chunk_##NAME(                                    \
                    nodes, node_count, input_count, output_slot,           \
                    row_inputs, inner_strides, out + outer * inner_size,   \
                    inner_size)) {                                         \
                return 0;                                                   \
            }                                                               \
            for (int axis = inner_axis - 1; axis >= 0; axis--) {           \
                coordinates[axis]++;                                       \
                for (uint8_t input = 0; input < input_count; input++) {    \
                    offsets[input] += input_strides[input][axis];          \
                }                                                           \
                if (coordinates[axis] < dimensions[axis]) {                \
                    break;                                                  \
                }                                                           \
                coordinates[axis] = 0;                                     \
                for (uint8_t input = 0; input < input_count; input++) {    \
                    offsets[input] -= dimensions[axis] *                   \
                        input_strides[input][axis];                         \
                }                                                           \
            }                                                               \
        }                                                                   \
        return 1;                                                           \
    } while (0)

/* Instantiate the complete strided drivers.  Integer drivers use unsigned
 * storage so add/subtract/multiply retain NumPy's modular wraparound. */
#define DEFINE_FLOAT_KERNEL(TYPE, NAME)                                      \
static int                                                                  \
crossjit_kernel_##NAME(                                                     \
    const NumpyCrossJitKernelNode *nodes, uint8_t node_count,               \
    uint8_t input_count, uint16_t scalar_mask, uint8_t output_slot,         \
    const void *const *inputs,                                               \
    const npy_intp input_strides[][NPY_MAXDIMS], uint8_t ndim,              \
    const npy_intp *dimensions, void *output, npy_intp size)                \
{                                                                           \
    TYPE slots[NUMPY_CROSSJIT_MAX_SLOTS];                                   \
    TYPE *restrict out = (TYPE *)output;                                     \
    npy_intp coordinates[NPY_MAXDIMS] = {0};                                \
    npy_intp offsets[NUMPY_CROSSJIT_MAX_INPUTS] = {0};                      \
    CROSSJIT_TRY_PAIR_FAST_PATH(TYPE);                                      \
    CROSSJIT_TRY_CHUNK_FAST_PATH(TYPE, NAME);                               \
    for (npy_intp index = 0; index < size; index++) {                       \
        for (uint8_t input = 0; input < input_count; input++) {             \
            const char *address = (const char *)inputs[input] +             \
                offsets[input];                                             \
            slots[input] = *(const TYPE *)address;                          \
        }                                                                   \
        switch (node_count) {                                               \
            case 6: CROSSJIT_EVAL_FLOAT_NODE(TYPE, NAME, 0);               \
                    CROSSJIT_EVAL_FLOAT_NODE(TYPE, NAME, 1);               \
                    CROSSJIT_EVAL_FLOAT_NODE(TYPE, NAME, 2);               \
                    CROSSJIT_EVAL_FLOAT_NODE(TYPE, NAME, 3);               \
                    CROSSJIT_EVAL_FLOAT_NODE(TYPE, NAME, 4);               \
                    CROSSJIT_EVAL_FLOAT_NODE(TYPE, NAME, 5); break;        \
            case 5: CROSSJIT_EVAL_FLOAT_NODE(TYPE, NAME, 0);               \
                    CROSSJIT_EVAL_FLOAT_NODE(TYPE, NAME, 1);               \
                    CROSSJIT_EVAL_FLOAT_NODE(TYPE, NAME, 2);               \
                    CROSSJIT_EVAL_FLOAT_NODE(TYPE, NAME, 3);               \
                    CROSSJIT_EVAL_FLOAT_NODE(TYPE, NAME, 4); break;        \
            case 4: CROSSJIT_EVAL_FLOAT_NODE(TYPE, NAME, 0);               \
                    CROSSJIT_EVAL_FLOAT_NODE(TYPE, NAME, 1);               \
                    CROSSJIT_EVAL_FLOAT_NODE(TYPE, NAME, 2);               \
                    CROSSJIT_EVAL_FLOAT_NODE(TYPE, NAME, 3); break;        \
            case 3: CROSSJIT_EVAL_FLOAT_NODE(TYPE, NAME, 0);               \
                    CROSSJIT_EVAL_FLOAT_NODE(TYPE, NAME, 1);               \
                    CROSSJIT_EVAL_FLOAT_NODE(TYPE, NAME, 2); break;        \
            case 2: CROSSJIT_EVAL_FLOAT_NODE(TYPE, NAME, 0);               \
                    CROSSJIT_EVAL_FLOAT_NODE(TYPE, NAME, 1); break;        \
            default: return 0;                                              \
        }                                                                   \
        out[index] = slots[output_slot];                                    \
        if (ndim == 1) {                                                    \
            for (uint8_t input = 0; input < input_count; input++) {         \
                offsets[input] += input_strides[input][0];                  \
            }                                                               \
            continue;                                                       \
        }                                                                   \
        for (int axis = (int)ndim - 1; axis >= 0; axis--) {                 \
            coordinates[axis]++;                                            \
            for (uint8_t input = 0; input < input_count; input++) {         \
                offsets[input] += input_strides[input][axis];               \
            }                                                               \
            if (coordinates[axis] < dimensions[axis]) {                     \
                break;                                                      \
            }                                                               \
            coordinates[axis] = 0;                                          \
            for (uint8_t input = 0; input < input_count; input++) {         \
                offsets[input] -= dimensions[axis] *                        \
                    input_strides[input][axis];                             \
            }                                                               \
        }                                                                   \
    }                                                                       \
    return 1;                                                               \
}

#define DEFINE_INTEGER_KERNEL(TYPE, NAME)                                    \
static int                                                                  \
crossjit_kernel_##NAME(                                                     \
    const NumpyCrossJitKernelNode *nodes, uint8_t node_count,               \
    uint8_t input_count, uint16_t scalar_mask, uint8_t output_slot,         \
    const void *const *inputs,                                               \
    const npy_intp input_strides[][NPY_MAXDIMS], uint8_t ndim,              \
    const npy_intp *dimensions, void *output, npy_intp size)                \
{                                                                           \
    TYPE slots[NUMPY_CROSSJIT_MAX_SLOTS];                                   \
    TYPE *restrict out = (TYPE *)output;                                     \
    npy_intp coordinates[NPY_MAXDIMS] = {0};                                \
    npy_intp offsets[NUMPY_CROSSJIT_MAX_INPUTS] = {0};                      \
    CROSSJIT_TRY_PAIR_FAST_PATH(TYPE);                                      \
    CROSSJIT_TRY_CHUNK_FAST_PATH(TYPE, NAME);                               \
    for (npy_intp index = 0; index < size; index++) {                       \
        for (uint8_t input = 0; input < input_count; input++) {             \
            const char *address = (const char *)inputs[input] +             \
                offsets[input];                                             \
            slots[input] = *(const TYPE *)address;                          \
        }                                                                   \
        switch (node_count) {                                               \
            case 6: CROSSJIT_EVAL_INTEGER_NODE(TYPE, 0);                   \
                    CROSSJIT_EVAL_INTEGER_NODE(TYPE, 1);                   \
                    CROSSJIT_EVAL_INTEGER_NODE(TYPE, 2);                   \
                    CROSSJIT_EVAL_INTEGER_NODE(TYPE, 3);                   \
                    CROSSJIT_EVAL_INTEGER_NODE(TYPE, 4);                   \
                    CROSSJIT_EVAL_INTEGER_NODE(TYPE, 5); break;            \
            case 5: CROSSJIT_EVAL_INTEGER_NODE(TYPE, 0);                   \
                    CROSSJIT_EVAL_INTEGER_NODE(TYPE, 1);                   \
                    CROSSJIT_EVAL_INTEGER_NODE(TYPE, 2);                   \
                    CROSSJIT_EVAL_INTEGER_NODE(TYPE, 3);                   \
                    CROSSJIT_EVAL_INTEGER_NODE(TYPE, 4); break;            \
            case 4: CROSSJIT_EVAL_INTEGER_NODE(TYPE, 0);                   \
                    CROSSJIT_EVAL_INTEGER_NODE(TYPE, 1);                   \
                    CROSSJIT_EVAL_INTEGER_NODE(TYPE, 2);                   \
                    CROSSJIT_EVAL_INTEGER_NODE(TYPE, 3); break;            \
            case 3: CROSSJIT_EVAL_INTEGER_NODE(TYPE, 0);                   \
                    CROSSJIT_EVAL_INTEGER_NODE(TYPE, 1);                   \
                    CROSSJIT_EVAL_INTEGER_NODE(TYPE, 2); break;            \
            case 2: CROSSJIT_EVAL_INTEGER_NODE(TYPE, 0);                   \
                    CROSSJIT_EVAL_INTEGER_NODE(TYPE, 1); break;            \
            default: return 0;                                              \
        }                                                                   \
        out[index] = slots[output_slot];                                    \
        if (ndim == 1) {                                                    \
            for (uint8_t input = 0; input < input_count; input++) {         \
                offsets[input] += input_strides[input][0];                  \
            }                                                               \
            continue;                                                       \
        }                                                                   \
        for (int axis = (int)ndim - 1; axis >= 0; axis--) {                 \
            coordinates[axis]++;                                            \
            for (uint8_t input = 0; input < input_count; input++) {         \
                offsets[input] += input_strides[input][axis];               \
            }                                                               \
            if (coordinates[axis] < dimensions[axis]) {                     \
                break;                                                      \
            }                                                               \
            coordinates[axis] = 0;                                          \
            for (uint8_t input = 0; input < input_count; input++) {         \
                offsets[input] -= dimensions[axis] *                        \
                    input_strides[input][axis];                             \
            }                                                               \
        }                                                                   \
    }                                                                       \
    return 1;                                                               \
}

DEFINE_FLOAT_KERNEL(double, double)
DEFINE_FLOAT_KERNEL(float, float)
DEFINE_INTEGER_KERNEL(npy_uint32, uint32)
DEFINE_INTEGER_KERNEL(npy_uint64, uint64)

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
        return crossjit_kernel_double(nodes, node_count, input_count, scalar_mask,
                                      output_slot, inputs, input_strides,
                                      ndim, dimensions, output, size);
    }
    if (typenum == NPY_FLOAT) {
        return crossjit_kernel_float(nodes, node_count, input_count, scalar_mask,
                                     output_slot, inputs, input_strides,
                                     ndim, dimensions, output, size);
    }
    if (typenum == NPY_INT && sizeof(npy_int) == sizeof(npy_uint32)) {
        return crossjit_kernel_uint32(nodes, node_count, input_count, scalar_mask,
                                      output_slot, inputs, input_strides,
                                      ndim, dimensions, output, size);
    }
    if ((typenum == NPY_LONG && sizeof(npy_long) == sizeof(npy_uint64)) ||
        (typenum == NPY_LONGLONG && sizeof(npy_longlong) == sizeof(npy_uint64))) {
        return crossjit_kernel_uint64(nodes, node_count, input_count, scalar_mask,
                                      output_slot, inputs, input_strides,
                                      ndim, dimensions, output, size);
    }
    return 0;
}
