#include <stddef.h>

void
crossjit_pf4_oracle_add_multiply(
        const void *const *inputs, const ptrdiff_t *outer_strides,
        double *restrict output, size_t outer_count, size_t inner_extent)
{
    for (size_t outer = 0; outer < outer_count; outer++) {
        const double *const left = (const double *)(
            (const char *)inputs[0] + (ptrdiff_t)outer * outer_strides[0]);
        const double *const right = (const double *)(
            (const char *)inputs[1] + (ptrdiff_t)outer * outer_strides[1]);
        const double *const scale = (const double *)(
            (const char *)inputs[2] + (ptrdiff_t)outer * outer_strides[2]);
        double *restrict row = output + outer * inner_extent;
        for (size_t inner = 0; inner < inner_extent; inner++) {
            const double temporary = left[inner] + right[inner];
            row[inner] = temporary * scale[inner];
        }
    }
}

void
crossjit_pf4_oracle_subtract_multiply(
        const void *const *inputs, const ptrdiff_t *outer_strides,
        double *restrict output, size_t outer_count, size_t inner_extent)
{
    for (size_t outer = 0; outer < outer_count; outer++) {
        const double *const left = (const double *)(
            (const char *)inputs[0] + (ptrdiff_t)outer * outer_strides[0]);
        const double *const right = (const double *)(
            (const char *)inputs[1] + (ptrdiff_t)outer * outer_strides[1]);
        const double *const scale = (const double *)(
            (const char *)inputs[2] + (ptrdiff_t)outer * outer_strides[2]);
        double *restrict row = output + outer * inner_extent;
        for (size_t inner = 0; inner < inner_extent; inner++) {
            const double temporary = left[inner] - right[inner];
            row[inner] = scale[inner] * temporary;
        }
    }
}
