#ifndef NUMPY_CORE_SRC_UMATH_JIT_KERNEL_H
#define NUMPY_CORE_SRC_UMATH_JIT_KERNEL_H

#include "numpy/arrayobject.h"

typedef enum {
    NUMPY_JIT_KERNEL_OP_ADD,
    NUMPY_JIT_KERNEL_OP_SUBTRACT,
    NUMPY_JIT_KERNEL_OP_MULTIPLY,
    NUMPY_JIT_KERNEL_OP_TRUE_DIVIDE,
    NUMPY_JIT_KERNEL_OP_SQUARE,
    NUMPY_JIT_KERNEL_OP_NEGATIVE,
} NumpyJitKernelOp;

#define NUMPY_JIT_KERNEL_REGION_MAX_NODES 4
#define NUMPY_JIT_KERNEL_REGION_MAX_INPUTS 6
#define NUMPY_JIT_KERNEL_REGION_MAX_SLOTS 10

typedef enum {
    NUMPY_JIT_KERNEL_REGION_ADD,
    NUMPY_JIT_KERNEL_REGION_SUB,
    NUMPY_JIT_KERNEL_REGION_MUL,
    NUMPY_JIT_KERNEL_REGION_DIV,
    NUMPY_JIT_KERNEL_REGION_SQUARE,
    NUMPY_JIT_KERNEL_REGION_NEGATIVE,
    NUMPY_JIT_KERNEL_REGION_FMA_ADD,
    NUMPY_JIT_KERNEL_REGION_FMA_SUB,
    NUMPY_JIT_KERNEL_REGION_FMA_RSUB,
    NUMPY_JIT_KERNEL_REGION_NOP,
} NumpyJitKernelRegionOp;

typedef struct {
    uint8_t op;
    uint8_t left;
    uint8_t right;
    uint8_t third;
    uint8_t out;
} NumpyJitKernelRegionNode;

int numpy_jit_kernel_region_contig(
    const NumpyJitKernelRegionNode *nodes, int node_count,
    int input_count, int output_slot, int dtype,
    const void *const *input_data, const double *scalar_values,
    const uint8_t *input_is_scalar, void *out, npy_intp size);

int numpy_jit_kernel_region_strided(
    const NumpyJitKernelRegionNode *nodes, int node_count,
    int input_count, int output_slot, int dtype, int ndim,
    const npy_intp *dims, const void *const *input_data,
    const npy_intp *const *input_strides, const double *scalar_values,
    const uint8_t *input_is_scalar, void *out,
    const npy_intp *out_strides, npy_intp size);

int numpy_jit_kernel_elemwise_contig(
    NumpyJitKernelOp op, int dtype, const void *left, const void *right,
    void *out, npy_intp size);

int numpy_jit_kernel_elemwise_strided_same_shape(
    NumpyJitKernelOp op, int dtype, int ndim, npy_intp size,
    const npy_intp *dims, const npy_intp *astrides,
    const npy_intp *bstrides, const char *abase, const char *bbase,
    void *out);

int numpy_jit_kernel_elemwise_strided_1d(
    NumpyJitKernelOp op, int dtype, npy_intp size,
    const char *left, npy_intp left_stride,
    const char *right, npy_intp right_stride, void *out);

int numpy_jit_kernel_array_scalar_contig(
    NumpyJitKernelOp op, int dtype, const void *array, double scalar,
    int scalar_first, void *out, npy_intp size);

int numpy_jit_kernel_row_bias_contig(
    NumpyJitKernelOp op, int dtype, void *matrix, const void *bias,
    npy_intp rows, npy_intp columns, int matrix_first);

int numpy_jit_kernel_array_scalar_strided_1d(
    NumpyJitKernelOp op, int dtype, const char *array, npy_intp stride,
    double scalar, int scalar_first, void *out, npy_intp size);

int numpy_jit_kernel_array_scalar_strided_inner(
    NumpyJitKernelOp op, int dtype, int ndim, npy_intp size,
    const npy_intp *dims, const npy_intp *strides, const char *array,
    double scalar, int scalar_first, void *out);

int numpy_jit_kernel_fused_contig_or_strided(
    int fused_op, int dtype, int layout_is_strided, int ndim, npy_intp size,
    const npy_intp *dims, const npy_intp *astrides,
    const npy_intp *bstrides, const char *abase, const char *bbase,
    double alpha, double beta, void *out);

int numpy_jit_kernel_add3_contig(
    int dtype, const void *a, const void *b, const void *c,
    void *out, npy_intp size, int order);

#endif
