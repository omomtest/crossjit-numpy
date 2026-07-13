#ifndef NUMPY_CORE_SRC_UMATH_JIT_KERNEL_H
#define NUMPY_CORE_SRC_UMATH_JIT_KERNEL_H

#include "numpy/arrayobject.h"

typedef enum {
    NUMPY_JIT_KERNEL_OP_ADD,
    NUMPY_JIT_KERNEL_OP_SUBTRACT,
    NUMPY_JIT_KERNEL_OP_MULTIPLY,
    NUMPY_JIT_KERNEL_OP_TRUE_DIVIDE,
} NumpyJitKernelOp;

int numpy_jit_kernel_elemwise_contig(
    NumpyJitKernelOp op, int dtype, const void *left, const void *right,
    void *out, npy_intp size);

int numpy_jit_kernel_elemwise_strided_same_shape(
    NumpyJitKernelOp op, int dtype, int ndim, npy_intp size,
    const npy_intp *dims, const npy_intp *astrides,
    const npy_intp *bstrides, const char *abase, const char *bbase,
    void *out);

int numpy_jit_kernel_array_scalar_contig(
    NumpyJitKernelOp op, int dtype, const void *array, double scalar,
    int scalar_first, void *out, npy_intp size);

int numpy_jit_kernel_fused_contig_or_strided(
    int fused_op, int dtype, int layout_is_strided, int ndim, npy_intp size,
    const npy_intp *dims, const npy_intp *astrides,
    const npy_intp *bstrides, const char *abase, const char *bbase,
    double alpha, double beta, void *out);

#endif
