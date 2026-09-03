#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include "jit_kernel.h"

#if defined(__GNUC__) && !defined(__clang__)
#define NUMPY_JIT_REGION_OPTIMIZED __attribute__((optimize("O3")))
#define NUMPY_JIT_ALWAYS_INLINE __attribute__((always_inline))
#else
#define NUMPY_JIT_REGION_OPTIMIZED
#define NUMPY_JIT_ALWAYS_INLINE
#endif

#define DEFINE_REGION_CHUNK(TYPE, NAME)                                      \
NUMPY_JIT_REGION_OPTIMIZED                                                    \
static inline NUMPY_JIT_ALWAYS_INLINE int                                    \
numpy_jit_region_chunk_##NAME(                                               \
    const NumpyJitKernelRegionNode *nodes, int node_count,                   \
    int input_count, int output_slot, const void *const *input_data,         \
    const double *scalar_values, const uint8_t *input_is_scalar,             \
    TYPE *out, npy_intp base, int lanes)                                     \
{                                                                            \
    TYPE slots[NUMPY_JIT_KERNEL_REGION_MAX_SLOTS][8];                        \
    for (int input = 0; input < input_count; input++) {                      \
        if (input_is_scalar[input]) {                                        \
            TYPE value = (TYPE)scalar_values[input];                         \
            for (int lane = 0; lane < lanes; lane++) {                       \
                slots[input][lane] = value;                                  \
            }                                                                \
        }                                                                    \
        else {                                                               \
            const TYPE *src = (const TYPE *)input_data[input] + base;        \
            for (int lane = 0; lane < lanes; lane++) {                       \
                slots[input][lane] = src[lane];                              \
            }                                                                \
        }                                                                    \
    }                                                                        \
    for (int n = 0; n < node_count; n++) {                                  \
        const NumpyJitKernelRegionNode *node = &nodes[n];                    \
        if (node->op == NUMPY_JIT_KERNEL_REGION_NOP) {                       \
            continue;                                                        \
        }                                                                    \
        for (int lane = 0; lane < lanes; lane++) {                           \
            TYPE left = slots[node->left][lane];                             \
            TYPE right = slots[node->right][lane];                           \
            switch (node->op) {                                              \
                case NUMPY_JIT_KERNEL_REGION_ADD:                            \
                    slots[node->out][lane] = left + right;                   \
                    break;                                                   \
                case NUMPY_JIT_KERNEL_REGION_SUB:                            \
                    slots[node->out][lane] = left - right;                   \
                    break;                                                   \
                case NUMPY_JIT_KERNEL_REGION_MUL:                            \
                    slots[node->out][lane] = left * right;                   \
                    break;                                                   \
                case NUMPY_JIT_KERNEL_REGION_DIV:                            \
                    slots[node->out][lane] = left / right;                   \
                    break;                                                   \
                case NUMPY_JIT_KERNEL_REGION_SQUARE:                         \
                    slots[node->out][lane] = left * left;                    \
                    break;                                                   \
                case NUMPY_JIT_KERNEL_REGION_NEGATIVE:                       \
                    slots[node->out][lane] = -left;                          \
                    break;                                                   \
                case NUMPY_JIT_KERNEL_REGION_FMA_ADD:                        \
                    slots[node->out][lane] =                                 \
                        left * right + slots[node->third][lane];             \
                    break;                                                   \
                case NUMPY_JIT_KERNEL_REGION_FMA_SUB:                        \
                    slots[node->out][lane] =                                 \
                        left * right - slots[node->third][lane];             \
                    break;                                                   \
                case NUMPY_JIT_KERNEL_REGION_FMA_RSUB:                       \
                    slots[node->out][lane] =                                 \
                        slots[node->third][lane] - left * right;             \
                    break;                                                   \
                default:                                                     \
                    return 0;                                                \
            }                                                                \
        }                                                                    \
    }                                                                        \
    for (int lane = 0; lane < lanes; lane++) {                               \
        out[base + lane] = slots[output_slot][lane];                         \
    }                                                                        \
    return 1;                                                                \
}

DEFINE_REGION_CHUNK(double, double)
DEFINE_REGION_CHUNK(float, float)

NUMPY_JIT_REGION_OPTIMIZED
int
numpy_jit_kernel_region_contig(
    const NumpyJitKernelRegionNode *nodes, int node_count,
    int input_count, int output_slot, int dtype,
    const void *const *input_data, const double *scalar_values,
    const uint8_t *input_is_scalar, void *out, npy_intp size)
{
    if (nodes == NULL || input_data == NULL || scalar_values == NULL ||
        input_is_scalar == NULL || out == NULL || node_count < 1 ||
        node_count > NUMPY_JIT_KERNEL_REGION_MAX_NODES || input_count < 1 ||
        input_count > NUMPY_JIT_KERNEL_REGION_MAX_INPUTS ||
        output_slot < 0 || output_slot >= NUMPY_JIT_KERNEL_REGION_MAX_SLOTS) {
        return 0;
    }
    for (npy_intp base = 0; base < size; base += 8) {
        int lanes = (int)((size - base) < 8 ? (size - base) : 8);
        int ok = dtype == NPY_DOUBLE ?
            numpy_jit_region_chunk_double(
                nodes, node_count, input_count, output_slot, input_data,
                scalar_values, input_is_scalar, (double *)out, base, lanes) :
            dtype == NPY_FLOAT ?
            numpy_jit_region_chunk_float(
                nodes, node_count, input_count, output_slot, input_data,
                scalar_values, input_is_scalar, (float *)out, base, lanes) : 0;
        if (!ok) {
            return 0;
        }
    }
    return 1;
}

#define DEFINE_REGION_STRIDED(TYPE, NAME)                                    \
static int                                                                   \
numpy_jit_region_strided_##NAME(                                             \
    const NumpyJitKernelRegionNode *nodes, int node_count,                   \
    int input_count, int output_slot, int ndim, const npy_intp *dims,        \
    const void *const *input_data, const npy_intp *const *input_strides,     \
    const double *scalar_values, const uint8_t *input_is_scalar,             \
    void *out, const npy_intp *out_strides, npy_intp size)                  \
{                                                                            \
    for (npy_intp linear = 0; linear < size; linear++) {                     \
        npy_intp coordinates[NPY_MAXDIMS];                                   \
        npy_intp remaining = linear;                                         \
        for (int dim = ndim - 1; dim >= 0; dim--) {                         \
            coordinates[dim] = remaining % dims[dim];                       \
            remaining /= dims[dim];                                         \
        }                                                                    \
        TYPE slots[NUMPY_JIT_KERNEL_REGION_MAX_SLOTS];                       \
        for (int input = 0; input < input_count; input++) {                  \
            if (input_is_scalar[input]) {                                    \
                slots[input] = (TYPE)scalar_values[input];                   \
                continue;                                                    \
            }                                                                \
            npy_intp offset = 0;                                             \
            for (int dim = 0; dim < ndim; dim++) {                          \
                offset += coordinates[dim] * input_strides[input][dim];      \
            }                                                                \
            slots[input] = *(const TYPE *)                                   \
                ((const char *)input_data[input] + offset);                  \
        }                                                                    \
        for (int n = 0; n < node_count; n++) {                              \
            const NumpyJitKernelRegionNode *node = &nodes[n];                \
            TYPE left = slots[node->left];                                   \
            TYPE right = slots[node->right];                                 \
            switch (node->op) {                                              \
                case NUMPY_JIT_KERNEL_REGION_ADD:                            \
                    slots[node->out] = left + right;                         \
                    break;                                                   \
                case NUMPY_JIT_KERNEL_REGION_SUB:                            \
                    slots[node->out] = left - right;                         \
                    break;                                                   \
                case NUMPY_JIT_KERNEL_REGION_MUL:                            \
                    slots[node->out] = left * right;                         \
                    break;                                                   \
                case NUMPY_JIT_KERNEL_REGION_DIV:                            \
                    slots[node->out] = left / right;                         \
                    break;                                                   \
                case NUMPY_JIT_KERNEL_REGION_SQUARE:                         \
                    slots[node->out] = left * left;                          \
                    break;                                                   \
                case NUMPY_JIT_KERNEL_REGION_NEGATIVE:                       \
                    slots[node->out] = -left;                                \
                    break;                                                   \
                default:                                                     \
                    return 0;                                                \
            }                                                                \
        }                                                                    \
        npy_intp out_offset = 0;                                             \
        for (int dim = 0; dim < ndim; dim++) {                              \
            out_offset += coordinates[dim] * out_strides[dim];              \
        }                                                                    \
        *(TYPE *)((char *)out + out_offset) = slots[output_slot];            \
    }                                                                        \
    return 1;                                                                \
}

DEFINE_REGION_STRIDED(double, double)
DEFINE_REGION_STRIDED(float, float)

int
numpy_jit_kernel_region_strided(
    const NumpyJitKernelRegionNode *nodes, int node_count,
    int input_count, int output_slot, int dtype, int ndim,
    const npy_intp *dims, const void *const *input_data,
    const npy_intp *const *input_strides, const double *scalar_values,
    const uint8_t *input_is_scalar, void *out,
    const npy_intp *out_strides, npy_intp size)
{
    if (nodes == NULL || dims == NULL || input_data == NULL ||
        input_strides == NULL || scalar_values == NULL ||
        input_is_scalar == NULL || out == NULL || out_strides == NULL ||
        ndim <= 0 || ndim > NPY_MAXDIMS || node_count < 1 ||
        node_count > NUMPY_JIT_KERNEL_REGION_MAX_NODES || input_count < 1 ||
        input_count > NUMPY_JIT_KERNEL_REGION_MAX_INPUTS ||
        output_slot < 0 || output_slot >= NUMPY_JIT_KERNEL_REGION_MAX_SLOTS) {
        return 0;
    }
    if (dtype == NPY_DOUBLE) {
        return numpy_jit_region_strided_double(
            nodes, node_count, input_count, output_slot, ndim, dims,
            input_data, input_strides, scalar_values, input_is_scalar,
            out, out_strides, size);
    }
    if (dtype == NPY_FLOAT) {
        return numpy_jit_region_strided_float(
            nodes, node_count, input_count, output_slot, ndim, dims,
            input_data, input_strides, scalar_values, input_is_scalar,
            out, out_strides, size);
    }
    return 0;
}

enum {
    NUMPY_JIT_KERNEL_FUSED_MUL_SCALAR_ADD_ARRAY,
    NUMPY_JIT_KERNEL_FUSED_MUL_SCALAR_SUB_ARRAY,
    NUMPY_JIT_KERNEL_FUSED_ARRAY_SUB_MUL_SCALAR,
    NUMPY_JIT_KERNEL_FUSED_ADD_ARRAY_MUL_SCALAR,
    NUMPY_JIT_KERNEL_FUSED_SUB_ARRAY_MUL_SCALAR,
    NUMPY_JIT_KERNEL_FUSED_ADD_ARRAY_DIV_SCALAR,
    NUMPY_JIT_KERNEL_FUSED_SUB_ARRAY_DIV_SCALAR,
    NUMPY_JIT_KERNEL_FUSED_MUL_TWO_SCALARS,
    NUMPY_JIT_KERNEL_FUSED_AXPBY,
    NUMPY_JIT_KERNEL_FUSED_ADD_THREE_ARRAYS,
};

int
numpy_jit_kernel_add3_contig(int dtype, const void *a, const void *b,
                             const void *c, void *out, npy_intp size,
                             int order)
{
    if (size <= 0) {
        return 0;
    }
    if (dtype == NPY_DOUBLE) {
        const double *ad = (const double *)a;
        const double *bd = (const double *)b;
        const double *cd = (const double *)c;
        double *restrict od = (double *)out;
        if (order == 1) {
            for (npy_intp i = 0; i < size; i++) {
                od[i] = od[i] + (ad[i] + bd[i]);
            }
        }
        else if (order == 2) {
            for (npy_intp i = 0; i < size; i++) {
                od[i] = cd[i] + (ad[i] + bd[i]);
            }
        }
        else {
            for (npy_intp i = 0; i < size; i++) {
                od[i] = (ad[i] + bd[i]) + cd[i];
            }
        }
        return 1;
    }
    if (dtype == NPY_FLOAT) {
        const float *af = (const float *)a;
        const float *bf = (const float *)b;
        const float *cf = (const float *)c;
        float *restrict of = (float *)out;
        if (order == 1) {
            for (npy_intp i = 0; i < size; i++) {
                of[i] = of[i] + (af[i] + bf[i]);
            }
        }
        else if (order == 2) {
            for (npy_intp i = 0; i < size; i++) {
                of[i] = cf[i] + (af[i] + bf[i]);
            }
        }
        else {
            for (npy_intp i = 0; i < size; i++) {
                of[i] = (af[i] + bf[i]) + cf[i];
            }
        }
        return 1;
    }
    return 0;
}

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
        case NUMPY_JIT_KERNEL_OP_SQUARE:
            *out = left * left;
            return 1;
        case NUMPY_JIT_KERNEL_OP_NEGATIVE:
            *out = -left;
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
        case NUMPY_JIT_KERNEL_OP_SQUARE:
            *out = left * left;
            return 1;
        case NUMPY_JIT_KERNEL_OP_NEGATIVE:
            *out = -left;
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
            case NUMPY_JIT_KERNEL_OP_SQUARE:
                for (npy_intp i = 0; i < size; i++) {
                    out_data[i] = left_data[i] * left_data[i];
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
            case NUMPY_JIT_KERNEL_OP_SQUARE:
                for (npy_intp i = 0; i < size; i++) {
                    out_data[i] = left_data[i] * left_data[i];
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

    /*
     * Stencil slices are overwhelmingly 2-D with a contiguous inner
     * dimension.  Dispatch once per row through the contiguous kernel so
     * that its arithmetic loops remain vectorizable; applying the operation
     * through a helper for every element prevents that on some compilers.
     */
    if (ndim == 2) {
        npy_intp outer = dims[0];
        npy_intp itemsize = dtype == NPY_DOUBLE ?
                            (npy_intp)sizeof(double) :
                            (npy_intp)sizeof(float);
        char *out_bytes = (char *)out;
        for (npy_intp row = 0; row < outer; row++) {
            if (!numpy_jit_kernel_elemwise_contig(
                    op, dtype,
                    abase + row * astrides[0],
                    bbase + row * bstrides[0],
                    out_bytes + row * inner * itemsize,
                    inner)) {
                return 0;
            }
        }
        return 1;
    }

    npy_intp coords[NPY_MAXDIMS] = {0};
    npy_intp outer = size / inner;

    npy_intp itemsize = dtype == NPY_DOUBLE ?
                        (npy_intp)sizeof(double) :
                        (npy_intp)sizeof(float);
    char *out_bytes = (char *)out;
    for (npy_intp outer_i = 0; outer_i < outer; outer_i++) {
        npy_intp a_offset = 0;
        npy_intp b_offset = 0;
        for (int dim = 0; dim < ndim - 1; dim++) {
            a_offset += coords[dim] * astrides[dim];
            b_offset += coords[dim] * bstrides[dim];
        }
        if (!numpy_jit_kernel_elemwise_contig(
                op, dtype, abase + a_offset, bbase + b_offset,
                out_bytes + outer_i * inner * itemsize, inner)) {
            return 0;
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
numpy_jit_kernel_elemwise_strided_1d(
    NumpyJitKernelOp op, int dtype, npy_intp size,
    const char *left, npy_intp left_stride,
    const char *right, npy_intp right_stride, void *out)
{
    if (left_stride <= 0 || right_stride <= 0) {
        return 0;
    }

    if (dtype == NPY_DOUBLE) {
        double *out_data = (double *)out;
        for (npy_intp i = 0; i < size; i++) {
            double left_value =
                *(const double *)(left + i * left_stride);
            double right_value =
                *(const double *)(right + i * right_stride);
            if (!apply_double(op, left_value, right_value, &out_data[i])) {
                return 0;
            }
        }
        return 1;
    }

    if (dtype == NPY_FLOAT) {
        float *out_data = (float *)out;
        for (npy_intp i = 0; i < size; i++) {
            float left_value = *(const float *)(left + i * left_stride);
            float right_value = *(const float *)(right + i * right_stride);
            if (!apply_float(op, left_value, right_value, &out_data[i])) {
                return 0;
            }
        }
        return 1;
    }

    return 0;
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
        switch (op) {
            case NUMPY_JIT_KERNEL_OP_ADD:
                for (npy_intp i = 0; i < size; i++) {
                    out_data[i] = scalar_first ?
                        scalar + array_data[i] : array_data[i] + scalar;
                }
                return 1;
            case NUMPY_JIT_KERNEL_OP_SUBTRACT:
                for (npy_intp i = 0; i < size; i++) {
                    out_data[i] = scalar_first ?
                        scalar - array_data[i] : array_data[i] - scalar;
                }
                return 1;
            case NUMPY_JIT_KERNEL_OP_MULTIPLY:
                for (npy_intp i = 0; i < size; i++) {
                    out_data[i] = scalar_first ?
                        scalar * array_data[i] : array_data[i] * scalar;
                }
                return 1;
            case NUMPY_JIT_KERNEL_OP_TRUE_DIVIDE:
                for (npy_intp i = 0; i < size; i++) {
                    out_data[i] = scalar_first ?
                        scalar / array_data[i] : array_data[i] / scalar;
                }
                return 1;
            case NUMPY_JIT_KERNEL_OP_SQUARE:
                for (npy_intp i = 0; i < size; i++) {
                    out_data[i] = array_data[i] * array_data[i];
                }
                return 1;
            default:
                return 0;
        }
    }

    const float *array_data = (const float *)array;
    float *out_data = (float *)out;
    float scalar_f = (float)scalar;
    switch (op) {
        case NUMPY_JIT_KERNEL_OP_ADD:
            for (npy_intp i = 0; i < size; i++) {
                out_data[i] = scalar_first ?
                    scalar_f + array_data[i] : array_data[i] + scalar_f;
            }
            return 1;
        case NUMPY_JIT_KERNEL_OP_SUBTRACT:
            for (npy_intp i = 0; i < size; i++) {
                out_data[i] = scalar_first ?
                    scalar_f - array_data[i] : array_data[i] - scalar_f;
            }
            return 1;
        case NUMPY_JIT_KERNEL_OP_MULTIPLY:
            for (npy_intp i = 0; i < size; i++) {
                out_data[i] = scalar_first ?
                    scalar_f * array_data[i] : array_data[i] * scalar_f;
            }
            return 1;
        case NUMPY_JIT_KERNEL_OP_TRUE_DIVIDE:
            for (npy_intp i = 0; i < size; i++) {
                out_data[i] = scalar_first ?
                    scalar_f / array_data[i] : array_data[i] / scalar_f;
            }
            return 1;
        case NUMPY_JIT_KERNEL_OP_SQUARE:
            for (npy_intp i = 0; i < size; i++) {
                out_data[i] = array_data[i] * array_data[i];
            }
            return 1;
            default:
                return 0;
    }
}

int
numpy_jit_kernel_row_bias_contig(NumpyJitKernelOp op, int dtype,
                                 void *matrix, const void *bias,
                                 npy_intp rows, npy_intp columns,
                                 int matrix_first)
{
#define NUMPY_JIT_ROW_BIAS_BODY(TYPE)                                      \
    do {                                                                    \
        TYPE *m = (TYPE *)matrix;                                           \
        const TYPE *b = (const TYPE *)bias;                                 \
        for (npy_intp row = 0; row < rows; row++) {                         \
            TYPE *current = m + row * columns;                              \
            for (npy_intp col = 0; col < columns; col++) {                  \
                TYPE left = matrix_first ? current[col] : b[col];           \
                TYPE right = matrix_first ? b[col] : current[col];          \
                switch (op) {                                               \
                    case NUMPY_JIT_KERNEL_OP_ADD:                           \
                        current[col] = left + right;                         \
                        break;                                              \
                    case NUMPY_JIT_KERNEL_OP_SUBTRACT:                      \
                        current[col] = left - right;                         \
                        break;                                              \
                    case NUMPY_JIT_KERNEL_OP_MULTIPLY:                      \
                        current[col] = left * right;                         \
                        break;                                              \
                    case NUMPY_JIT_KERNEL_OP_TRUE_DIVIDE:                   \
                        current[col] = left / right;                         \
                        break;                                              \
                    default:                                                \
                        return 0;                                           \
                }                                                           \
            }                                                               \
        }                                                                   \
        return 1;                                                           \
    } while (0)
    if (matrix == NULL || bias == NULL || rows < 0 || columns < 0) {
        return 0;
    }
    if (dtype == NPY_DOUBLE) {
        NUMPY_JIT_ROW_BIAS_BODY(double);
    }
    if (dtype == NPY_FLOAT) {
        NUMPY_JIT_ROW_BIAS_BODY(float);
    }
    return 0;
#undef NUMPY_JIT_ROW_BIAS_BODY
}

int
numpy_jit_kernel_array_scalar_strided_1d(
    NumpyJitKernelOp op, int dtype, const char *array, npy_intp stride,
    double scalar, int scalar_first, void *out, npy_intp size)
{
    if (stride <= 0) {
        return 0;
    }

    if (dtype == NPY_DOUBLE) {
        double *out_data = (double *)out;
        for (npy_intp i = 0; i < size; i++) {
            double value = *(const double *)(array + i * stride);
            double left = scalar_first ? scalar : value;
            double right = scalar_first ? value : scalar;
            if (!apply_double(op, left, right, &out_data[i])) {
                return 0;
            }
        }
        return 1;
    }

    if (dtype == NPY_FLOAT) {
        float scalar_f = (float)scalar;
        float *out_data = (float *)out;
        for (npy_intp i = 0; i < size; i++) {
            float value = *(const float *)(array + i * stride);
            float left = scalar_first ? scalar_f : value;
            float right = scalar_first ? value : scalar_f;
            if (!apply_float(op, left, right, &out_data[i])) {
                return 0;
            }
        }
        return 1;
    }

    return 0;
}

int
numpy_jit_kernel_array_scalar_strided_inner(
    NumpyJitKernelOp op, int dtype, int ndim, npy_intp size,
    const npy_intp *dims, const npy_intp *strides, const char *array,
    double scalar, int scalar_first, void *out)
{
    if (ndim <= 1 || ndim > NPY_MAXDIMS) {
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
            npy_intp offset = 0;
            for (int dim = 0; dim < ndim - 1; dim++) {
                offset += coords[dim] * strides[dim];
            }
            const double *array_data = (const double *)(array + offset);
            for (npy_intp i = 0; i < inner; i++) {
                double value = array_data[i];
                double left = scalar_first ? scalar : value;
                double right = scalar_first ? value : scalar;
                if (!apply_double(
                        op, left, right,
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

    if (dtype == NPY_FLOAT) {
        float scalar_f = (float)scalar;
        float *out_data = (float *)out;
        for (npy_intp outer_i = 0; outer_i < outer; outer_i++) {
            npy_intp offset = 0;
            for (int dim = 0; dim < ndim - 1; dim++) {
                offset += coords[dim] * strides[dim];
            }
            const float *array_data = (const float *)(array + offset);
            for (npy_intp i = 0; i < inner; i++) {
                float value = array_data[i];
                float left = scalar_first ? scalar_f : value;
                float right = scalar_first ? value : scalar_f;
                if (!apply_float(
                        op, left, right,
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

    return 0;
}

static int
numpy_jit_kernel_fused_row_double(
    int fused_op, const double *a, const double *b,
    double alpha, double beta, double *out, npy_intp size)
{
    switch (fused_op) {
        case NUMPY_JIT_KERNEL_FUSED_MUL_SCALAR_ADD_ARRAY:
            for (npy_intp i = 0; i < size; i++) {
                out[i] = a[i] * alpha + b[i];
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_MUL_SCALAR_SUB_ARRAY:
            for (npy_intp i = 0; i < size; i++) {
                out[i] = a[i] * alpha - b[i];
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_ARRAY_SUB_MUL_SCALAR:
            for (npy_intp i = 0; i < size; i++) {
                out[i] = b[i] - a[i] * alpha;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_ADD_ARRAY_MUL_SCALAR:
            for (npy_intp i = 0; i < size; i++) {
                out[i] = (a[i] + b[i]) * alpha;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_SUB_ARRAY_MUL_SCALAR:
            for (npy_intp i = 0; i < size; i++) {
                out[i] = (a[i] - b[i]) * alpha;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_ADD_ARRAY_DIV_SCALAR:
            for (npy_intp i = 0; i < size; i++) {
                out[i] = (a[i] + b[i]) / alpha;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_SUB_ARRAY_DIV_SCALAR:
            for (npy_intp i = 0; i < size; i++) {
                out[i] = (a[i] - b[i]) / alpha;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_MUL_TWO_SCALARS:
            for (npy_intp i = 0; i < size; i++) {
                double first = a[i] * alpha;
                out[i] = first * beta;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_AXPBY:
            for (npy_intp i = 0; i < size; i++) {
                out[i] = alpha * a[i] + beta * b[i];
            }
            return 1;
        default:
            return 0;
    }
}

static int
numpy_jit_kernel_fused_row_float(
    int fused_op, const float *a, const float *b,
    float alpha, float beta, float *out, npy_intp size)
{
    switch (fused_op) {
        case NUMPY_JIT_KERNEL_FUSED_MUL_SCALAR_ADD_ARRAY:
            for (npy_intp i = 0; i < size; i++) {
                out[i] = a[i] * alpha + b[i];
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_MUL_SCALAR_SUB_ARRAY:
            for (npy_intp i = 0; i < size; i++) {
                out[i] = a[i] * alpha - b[i];
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_ARRAY_SUB_MUL_SCALAR:
            for (npy_intp i = 0; i < size; i++) {
                out[i] = b[i] - a[i] * alpha;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_ADD_ARRAY_MUL_SCALAR:
            for (npy_intp i = 0; i < size; i++) {
                out[i] = (a[i] + b[i]) * alpha;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_SUB_ARRAY_MUL_SCALAR:
            for (npy_intp i = 0; i < size; i++) {
                out[i] = (a[i] - b[i]) * alpha;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_ADD_ARRAY_DIV_SCALAR:
            for (npy_intp i = 0; i < size; i++) {
                out[i] = (a[i] + b[i]) / alpha;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_SUB_ARRAY_DIV_SCALAR:
            for (npy_intp i = 0; i < size; i++) {
                out[i] = (a[i] - b[i]) / alpha;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_MUL_TWO_SCALARS:
            for (npy_intp i = 0; i < size; i++) {
                float first = a[i] * alpha;
                out[i] = first * beta;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_AXPBY:
            for (npy_intp i = 0; i < size; i++) {
                out[i] = alpha * a[i] + beta * b[i];
            }
            return 1;
        default:
            return 0;
    }
}

static int
numpy_jit_kernel_fused_strided_1d_double(
    int fused_op, const char *abase, npy_intp astride,
    const char *bbase, npy_intp bstride,
    double alpha, double beta, double *out, npy_intp size)
{
    switch (fused_op) {
        case NUMPY_JIT_KERNEL_FUSED_MUL_SCALAR_ADD_ARRAY:
            for (npy_intp i = 0; i < size; i++) {
                double a = *(const double *)(abase + i * astride);
                double b = *(const double *)(bbase + i * bstride);
                out[i] = a * alpha + b;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_MUL_SCALAR_SUB_ARRAY:
            for (npy_intp i = 0; i < size; i++) {
                double a = *(const double *)(abase + i * astride);
                double b = *(const double *)(bbase + i * bstride);
                out[i] = a * alpha - b;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_ARRAY_SUB_MUL_SCALAR:
            for (npy_intp i = 0; i < size; i++) {
                double a = *(const double *)(abase + i * astride);
                double b = *(const double *)(bbase + i * bstride);
                out[i] = b - a * alpha;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_ADD_ARRAY_MUL_SCALAR:
            for (npy_intp i = 0; i < size; i++) {
                double a = *(const double *)(abase + i * astride);
                double b = *(const double *)(bbase + i * bstride);
                out[i] = (a + b) * alpha;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_SUB_ARRAY_MUL_SCALAR:
            for (npy_intp i = 0; i < size; i++) {
                double a = *(const double *)(abase + i * astride);
                double b = *(const double *)(bbase + i * bstride);
                out[i] = (a - b) * alpha;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_ADD_ARRAY_DIV_SCALAR:
            for (npy_intp i = 0; i < size; i++) {
                double a = *(const double *)(abase + i * astride);
                double b = *(const double *)(bbase + i * bstride);
                out[i] = (a + b) / alpha;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_SUB_ARRAY_DIV_SCALAR:
            for (npy_intp i = 0; i < size; i++) {
                double a = *(const double *)(abase + i * astride);
                double b = *(const double *)(bbase + i * bstride);
                out[i] = (a - b) / alpha;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_MUL_TWO_SCALARS:
            for (npy_intp i = 0; i < size; i++) {
                double a = *(const double *)(abase + i * astride);
                double first = a * alpha;
                out[i] = first * beta;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_AXPBY:
            for (npy_intp i = 0; i < size; i++) {
                double a = *(const double *)(abase + i * astride);
                double b = *(const double *)(bbase + i * bstride);
                out[i] = alpha * a + beta * b;
            }
            return 1;
        default:
            return 0;
    }
}

static int
numpy_jit_kernel_fused_strided_1d_float(
    int fused_op, const char *abase, npy_intp astride,
    const char *bbase, npy_intp bstride,
    float alpha, float beta, float *out, npy_intp size)
{
    switch (fused_op) {
        case NUMPY_JIT_KERNEL_FUSED_MUL_SCALAR_ADD_ARRAY:
            for (npy_intp i = 0; i < size; i++) {
                float a = *(const float *)(abase + i * astride);
                float b = *(const float *)(bbase + i * bstride);
                out[i] = a * alpha + b;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_MUL_SCALAR_SUB_ARRAY:
            for (npy_intp i = 0; i < size; i++) {
                float a = *(const float *)(abase + i * astride);
                float b = *(const float *)(bbase + i * bstride);
                out[i] = a * alpha - b;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_ARRAY_SUB_MUL_SCALAR:
            for (npy_intp i = 0; i < size; i++) {
                float a = *(const float *)(abase + i * astride);
                float b = *(const float *)(bbase + i * bstride);
                out[i] = b - a * alpha;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_ADD_ARRAY_MUL_SCALAR:
            for (npy_intp i = 0; i < size; i++) {
                float a = *(const float *)(abase + i * astride);
                float b = *(const float *)(bbase + i * bstride);
                out[i] = (a + b) * alpha;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_SUB_ARRAY_MUL_SCALAR:
            for (npy_intp i = 0; i < size; i++) {
                float a = *(const float *)(abase + i * astride);
                float b = *(const float *)(bbase + i * bstride);
                out[i] = (a - b) * alpha;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_ADD_ARRAY_DIV_SCALAR:
            for (npy_intp i = 0; i < size; i++) {
                float a = *(const float *)(abase + i * astride);
                float b = *(const float *)(bbase + i * bstride);
                out[i] = (a + b) / alpha;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_SUB_ARRAY_DIV_SCALAR:
            for (npy_intp i = 0; i < size; i++) {
                float a = *(const float *)(abase + i * astride);
                float b = *(const float *)(bbase + i * bstride);
                out[i] = (a - b) / alpha;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_MUL_TWO_SCALARS:
            for (npy_intp i = 0; i < size; i++) {
                float a = *(const float *)(abase + i * astride);
                float first = a * alpha;
                out[i] = first * beta;
            }
            return 1;
        case NUMPY_JIT_KERNEL_FUSED_AXPBY:
            for (npy_intp i = 0; i < size; i++) {
                float a = *(const float *)(abase + i * astride);
                float b = *(const float *)(bbase + i * bstride);
                out[i] = alpha * a + beta * b;
            }
            return 1;
        default:
            return 0;
    }
}

static int
numpy_jit_kernel_fused_row(
    int fused_op, int dtype, const char *a, const char *b,
    double alpha, double beta, void *out, npy_intp size)
{
    if (dtype == NPY_DOUBLE) {
        return numpy_jit_kernel_fused_row_double(
            fused_op, (const double *)a, (const double *)b,
            alpha, beta, (double *)out, size);
    }
    if (dtype == NPY_FLOAT) {
        return numpy_jit_kernel_fused_row_float(
            fused_op, (const float *)a, (const float *)b,
            (float)alpha, (float)beta, (float *)out, size);
    }
    return 0;
}

int
numpy_jit_kernel_fused_contig_or_strided(
    int fused_op, int dtype, int layout_is_strided, int ndim, npy_intp size,
    const npy_intp *dims, const npy_intp *astrides,
    const npy_intp *bstrides, const char *abase, const char *bbase,
    double alpha, double beta, void *out)
{
    if (ndim <= 0 || ndim > NPY_MAXDIMS || size <= 0 ||
        (dtype != NPY_DOUBLE && dtype != NPY_FLOAT)) {
        return 0;
    }

    if (!layout_is_strided) {
        return numpy_jit_kernel_fused_row(
            fused_op, dtype, abase, bbase, alpha, beta, out, size);
    }

    npy_intp inner = dims[ndim - 1];
    if (inner <= 0) {
        return 0;
    }
    if (ndim == 1) {
        if (astrides[0] <= 0 || bstrides[0] <= 0) {
            return 0;
        }
        if (dtype == NPY_DOUBLE) {
            return numpy_jit_kernel_fused_strided_1d_double(
                fused_op, abase, astrides[0], bbase, bstrides[0],
                alpha, beta, (double *)out, size);
        }
        return numpy_jit_kernel_fused_strided_1d_float(
            fused_op, abase, astrides[0], bbase, bstrides[0],
            (float)alpha, (float)beta, (float *)out, size);
    }
    npy_intp itemsize = dtype == NPY_DOUBLE ?
                        (npy_intp)sizeof(double) :
                        (npy_intp)sizeof(float);
    char *out_bytes = (char *)out;

    /* Common stencil case: positive-strided rows, contiguous inner loop. */
    if (ndim == 2) {
        for (npy_intp row = 0; row < dims[0]; row++) {
            if (!numpy_jit_kernel_fused_row(
                    fused_op, dtype,
                    abase + row * astrides[0],
                    bbase + row * bstrides[0],
                    alpha, beta, out_bytes + row * inner * itemsize,
                    inner)) {
                return 0;
            }
        }
        return 1;
    }

    npy_intp outer = size / inner;
    npy_intp coords[NPY_MAXDIMS] = {0};
    for (npy_intp outer_i = 0; outer_i < outer; outer_i++) {
        npy_intp a_offset = 0;
        npy_intp b_offset = 0;
        for (int dim = 0; dim < ndim - 1; dim++) {
            a_offset += coords[dim] * astrides[dim];
            b_offset += coords[dim] * bstrides[dim];
        }
        if (!numpy_jit_kernel_fused_row(
                fused_op, dtype, abase + a_offset, bbase + b_offset,
                alpha, beta, out_bytes + outer_i * inner * itemsize,
                inner)) {
            return 0;
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
