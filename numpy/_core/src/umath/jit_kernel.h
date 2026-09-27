#ifndef NUMPY_CORE_SRC_UMATH_JIT_KERNEL_H
#define NUMPY_CORE_SRC_UMATH_JIT_KERNEL_H

#include "numpy/arrayobject.h"

#define NUMPY_CROSSJIT_MAX_NODES 12
#define NUMPY_CROSSJIT_MAX_INPUTS 16
#define NUMPY_CROSSJIT_MAX_SLOTS \
    (NUMPY_CROSSJIT_MAX_INPUTS + NUMPY_CROSSJIT_MAX_NODES)

/* Compact opcodes interpreted by NumPy's fixed, precompiled loop bodies. */
typedef enum {
    NUMPY_CROSSJIT_KERNEL_ADD,
    NUMPY_CROSSJIT_KERNEL_SUBTRACT,
    NUMPY_CROSSJIT_KERNEL_MULTIPLY,
    NUMPY_CROSSJIT_KERNEL_TRUE_DIVIDE,
    NUMPY_CROSSJIT_KERNEL_NEGATIVE,
    NUMPY_CROSSJIT_KERNEL_MINIMUM,
    NUMPY_CROSSJIT_KERNEL_MAXIMUM,
} NumpyCrossJitKernelOp;

/* Describes one node in the provider-owned precompiled elementwise program. */
typedef struct {
    uint8_t op;                  /* NumpyCrossJitKernelOp */
    uint8_t left;                /* input or prior-node virtual slot */
    uint8_t right;               /* ignored by unary operations */
    uint8_t output;              /* virtual slot defined by this node */
} NumpyCrossJitKernelNode;

typedef enum {
    NUMPY_CROSSJIT_FIXED_GENERIC_PATH = 0,
    NUMPY_CROSSJIT_FIXED_FAST_PATH = 1,
} NumpyCrossJitFixedPath;

/* Executes a validated two-to-six-node DAG over one strided iteration domain.
 * Returns zero without publishing partial output when the program is invalid. */
int numpy_crossjit_kernel_strided(
    const NumpyCrossJitKernelNode *nodes, uint8_t node_count,
    uint8_t input_count, uint16_t scalar_mask, uint8_t output_slot, int typenum,
    const void *const *inputs,
    const npy_intp input_strides[][NPY_MAXDIMS], uint8_t ndim,
    const npy_intp *dimensions, void *output, npy_intp size,
    int allow_fast_path, NumpyCrossJitFixedPath *executed_path);

#endif
