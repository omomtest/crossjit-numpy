#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include "jit_kernel.h"

enum {
    NUMPY_JIT_KERNEL_FUSED_MUL_SCALAR_ADD_ARRAY,
    NUMPY_JIT_KERNEL_FUSED_ADD_ARRAY_MUL_SCALAR,
    NUMPY_JIT_KERNEL_FUSED_AXPBY,
};

static int
apply_double(NumpyJitKernelOp op, double left, double right, double *out)
{
    switch (op) {
        case NUMPY_JIT_KERNEL_OP_ADD:
            *out = left + right;
            return 1;
        case NUMPY_JIT_KERNEL_OP_SUBTRACT:
            *out = left - right;
            return 1;
        case NUMPY_JIT_KERNEL_OP_MULTIPLY:
            *out = left * right;
            return 1;
        case NUMPY_JIT_KERNEL_OP_TRUE_DIVIDE:
            *out = left / right;
            return 1;
        default:
            return 0;
    }
}

static int
apply_float(NumpyJitKernelOp op, float left, float right, float *out)
{
    switch (op) {
        case NUMPY_JIT_KERNEL_OP_ADD:
            *out = left + right;
            return 1;
        case NUMPY_JIT_KERNEL_OP_SUBTRACT:
            *out = left - right;
            return 1;
        case NUMPY_JIT_KERNEL_OP_MULTIPLY:
            *out = left * right;
            return 1;
        case NUMPY_JIT_KERNEL_OP_TRUE_DIVIDE:
            *out = left / right;
            return 1;
        default:
            return 0;
    }
}

int
numpy_jit_kernel_elemwise_contig(NumpyJitKernelOp op, int dtype,
                                 const void *left, const void *right,
                                 void *out, npy_intp size)
{
    if (dtype == NPY_DOUBLE) {
        const double *left_data = (const double *)left;
        const double *right_data = (const double *)right;
        double *out_data = (double *)out;
        switch (op) {
            case NUMPY_JIT_KERNEL_OP_ADD:
                for (npy_intp i = 0; i < size; i++) {
                    out_data[i] = left_data[i] + right_data[i];
                }
                return 1;
            case NUMPY_JIT_KERNEL_OP_SUBTRACT:
                for (npy_intp i = 0; i < size; i++) {
                    out_data[i] = left_data[i] - right_data[i];
                }
                return 1;
            case NUMPY_JIT_KERNEL_OP_MULTIPLY:
                for (npy_intp i = 0; i < size; i++) {
                    out_data[i] = left_data[i] * right_data[i];
                }
                return 1;
            case NUMPY_JIT_KERNEL_OP_TRUE_DIVIDE:
                for (npy_intp i = 0; i < size; i++) {
                    out_data[i] = left_data[i] / right_data[i];
                }
                return 1;
            default:
                return 0;
        }
    }

    const float *left_data = (const float *)left;
    const float *right_data = (const float *)right;
    float *out_data = (float *)out;
    switch (op) {
        case NUMPY_JIT_KERNEL_OP_ADD:
            for (npy_intp i = 0; i < size; i++) {
                out_data[i] = left_data[i] + right_data[i];
            }
            return 1;
        case NUMPY_JIT_KERNEL_OP_SUBTRACT:
            for (npy_intp i = 0; i < size; i++) {
                out_data[i] = left_data[i] - right_data[i];
            }
            return 1;
        case NUMPY_JIT_KERNEL_OP_MULTIPLY:
            for (npy_intp i = 0; i < size; i++) {
                out_data[i] = left_data[i] * right_data[i];
            }
            return 1;
        case NUMPY_JIT_KERNEL_OP_TRUE_DIVIDE:
            for (npy_intp i = 0; i < size; i++) {
                out_data[i] = left_data[i] / right_data[i];
            }
            return 1;
        default:
            return 0;
    }
}

int
numpy_jit_kernel_elemwise_strided_same_shape(
    NumpyJitKernelOp op, int dtype, int ndim, npy_intp size,
    const npy_intp *dims, const npy_intp *astrides,
    const npy_intp *bstrides, const char *abase, const char *bbase,
    void *out)
{
    if (ndim <= 0 || ndim > NPY_MAXDIMS) {
        return 0;
    }

    npy_intp inner = dims[ndim - 1];
    if (inner <= 0) {
        return 0;
    }

    npy_intp coords[NPY_MAXDIMS] = {0};
    npy_intp outer = size / inner;

    if (dtype == NPY_DOUBLE) {
        double *out_data = (double *)out;
        for (npy_intp outer_i = 0; outer_i < outer; outer_i++) {
            npy_intp a_offset = 0;
            npy_intp b_offset = 0;
            for (int dim = 0; dim < ndim - 1; dim++) {
                a_offset += coords[dim] * astrides[dim];
                b_offset += coords[dim] * bstrides[dim];
            }
            const double *left_data = (const double *)(abase + a_offset);
            const double *right_data = (const double *)(bbase + b_offset);
            for (npy_intp i = 0; i < inner; i++) {
                if (!apply_double(
                        op, left_data[i], right_data[i],
                        &out_data[outer_i * inner + i])) {
                    return 0;
                }
            }
            for (int dim = ndim - 2; dim >= 0; dim--) {
                coords[dim]++;
                if (coords[dim] < dims[dim]) {
                    break;
                }
                coords[dim] = 0;
            }
        }
        return 1;
    }

    float *out_data = (float *)out;
    for (npy_intp outer_i = 0; outer_i < outer; outer_i++) {
        npy_intp a_offset = 0;
        npy_intp b_offset = 0;
        for (int dim = 0; dim < ndim - 1; dim++) {
            a_offset += coords[dim] * astrides[dim];
            b_offset += coords[dim] * bstrides[dim];
        }
        const float *left_data = (const float *)(abase + a_offset);
        const float *right_data = (const float *)(bbase + b_offset);
        for (npy_intp i = 0; i < inner; i++) {
            if (!apply_float(
                    op, left_data[i], right_data[i],
                    &out_data[outer_i * inner + i])) {
                return 0;
            }
        }
        for (int dim = ndim - 2; dim >= 0; dim--) {
            coords[dim]++;
            if (coords[dim] < dims[dim]) {
                break;
            }
            coords[dim] = 0;
        }
    }
    return 1;
}

int
numpy_jit_kernel_array_scalar_contig(NumpyJitKernelOp op, int dtype,
                                     const void *array, double scalar,
                                     int scalar_first, void *out,
                                     npy_intp size)
{
    if (dtype == NPY_DOUBLE) {
        const double *array_data = (const double *)array;
        double *out_data = (double *)out;
        for (npy_intp i = 0; i < size; i++) {
            double left = scalar_first ? scalar : array_data[i];
            double right = scalar_first ? array_data[i] : scalar;
            if (!apply_double(op, left, right, &out_data[i])) {
                return 0;
            }
        }
        return 1;
    }

    const float *array_data = (const float *)array;
    float *out_data = (float *)out;
    float scalar_f = (float)scalar;
    for (npy_intp i = 0; i < size; i++) {
        float left = scalar_first ? scalar_f : array_data[i];
        float right = scalar_first ? array_data[i] : scalar_f;
        if (!apply_float(op, left, right, &out_data[i])) {
            return 0;
        }
    }
    return 1;
}

int
numpy_jit_kernel_fused_contig_or_strided(
    int fused_op, int dtype, int layout_is_strided, int ndim, npy_intp size,
    const npy_intp *dims, const npy_intp *astrides,
    const npy_intp *bstrides, const char *abase, const char *bbase,
    double alpha, double beta, void *out)
{
    npy_intp inner = ndim > 0 ? dims[ndim - 1] : size;
    npy_intp outer = inner > 0 ? size / inner : 0;
    npy_intp coords[NPY_MAXDIMS] = {0};

    if (dtype == NPY_DOUBLE) {
        double *out_data = (double *)out;
        for (npy_intp outer_i = 0; outer_i < outer; outer_i++) {
            npy_intp a_offset = 0;
            npy_intp b_offset = 0;
            if (layout_is_strided) {
                for (int dim = 0; dim < ndim - 1; dim++) {
                    a_offset += coords[dim] * astrides[dim];
                    b_offset += coords[dim] * bstrides[dim];
                }
            }
            else {
                a_offset = outer_i * inner * (npy_intp)sizeof(double);
                b_offset = outer_i * inner * (npy_intp)sizeof(double);
            }
            const double *a_data = (const double *)(abase + a_offset);
            const double *b_data = (const double *)(bbase + b_offset);
            for (npy_intp i = 0; i < inner; i++) {
                npy_intp out_i = outer_i * inner + i;
                switch (fused_op) {
                    case NUMPY_JIT_KERNEL_FUSED_MUL_SCALAR_ADD_ARRAY:
                        out_data[out_i] = a_data[i] * alpha + b_data[i];
                        break;
                    case NUMPY_JIT_KERNEL_FUSED_ADD_ARRAY_MUL_SCALAR:
                        out_data[out_i] = (a_data[i] + b_data[i]) * alpha;
                        break;
                    case NUMPY_JIT_KERNEL_FUSED_AXPBY:
                        out_data[out_i] = alpha * a_data[i] + beta * b_data[i];
                        break;
                    default:
                        return 0;
                }
            }
            if (layout_is_strided) {
                for (int dim = ndim - 2; dim >= 0; dim--) {
                    coords[dim]++;
                    if (coords[dim] < dims[dim]) {
                        break;
                    }
                    coords[dim] = 0;
                }
            }
        }
        return 1;
    }

    float alpha_f = (float)alpha;
    float beta_f = (float)beta;
    float *out_data = (float *)out;
    for (npy_intp outer_i = 0; outer_i < outer; outer_i++) {
        npy_intp a_offset = 0;
        npy_intp b_offset = 0;
        if (layout_is_strided) {
            for (int dim = 0; dim < ndim - 1; dim++) {
                a_offset += coords[dim] * astrides[dim];
                b_offset += coords[dim] * bstrides[dim];
            }
        }
        else {
            a_offset = outer_i * inner * (npy_intp)sizeof(float);
            b_offset = outer_i * inner * (npy_intp)sizeof(float);
        }
        const float *a_data = (const float *)(abase + a_offset);
        const float *b_data = (const float *)(bbase + b_offset);
        for (npy_intp i = 0; i < inner; i++) {
            npy_intp out_i = outer_i * inner + i;
            switch (fused_op) {
                case NUMPY_JIT_KERNEL_FUSED_MUL_SCALAR_ADD_ARRAY:
                    out_data[out_i] = a_data[i] * alpha_f + b_data[i];
                    break;
                case NUMPY_JIT_KERNEL_FUSED_ADD_ARRAY_MUL_SCALAR:
                    out_data[out_i] = (a_data[i] + b_data[i]) * alpha_f;
                    break;
                case NUMPY_JIT_KERNEL_FUSED_AXPBY:
                    out_data[out_i] = alpha_f * a_data[i] + beta_f * b_data[i];
                    break;
                default:
                    return 0;
            }
        }
        if (layout_is_strided) {
            for (int dim = ndim - 2; dim >= 0; dim--) {
                coords[dim]++;
                if (coords[dim] < dims[dim]) {
                    break;
                }
                coords[dim] = 0;
            }
        }
    }
    return 1;
}
