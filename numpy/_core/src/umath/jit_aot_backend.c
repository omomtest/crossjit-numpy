#include "jit_aot_backend.h"

#include <string.h>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define NUMPY_JIT_AOT_X86 1
#else
#define NUMPY_JIT_AOT_X86 0
#endif

typedef struct {
    uint64_t graph_hash;
    uint8_t scalar_mask;
    uint8_t accepted_bucket_mask;
    uint8_t node0_op, node0_left, node0_right, node0_out;
    uint8_t node1_op, node1_left, node1_right, node1_out;
    uint8_t node_count;
    const char *name;
} NumpyJitAOTManifestRecord;

#include "jit_aot_manifest_generated.h"

struct NumpyJitAOTEntry {
    NumpyJitKernelFunc function;
    uint8_t accepted_bucket_mask;
    uint16_t vector_bits;
    uint16_t code_size;
    const char *name;
};

/* The manifest's first reviewed shape is:
 *
 *     out = input0 - input1 ** input2, where input2 is exact integer 2.
 *
 * Keeping the exponent outside the native ABI is safe because the shared
 * Region power guard validates it before the AOT lookup.  This is a complete
 * two-node DAG loop, not a composition of unary/binary loop fragments. */
static void
numpy_jit_aot_square_sub_f64_scalar(const NumpyJitKernelArgs *args)
{
    const double *restrict a = (const double *)args->data[0];
    const double *restrict b = (const double *)args->data[1];
    double *restrict out = (double *)args->output;
    for (npy_intp i = 0; i < args->size; i++) {
        out[i] = a[i] - b[i] * b[i];
    }
}

#if NUMPY_JIT_AOT_X86
__attribute__((target("avx2")))
static void
numpy_jit_aot_square_sub_f64_avx2(const NumpyJitKernelArgs *args)
{
    const double *restrict a = (const double *)args->data[0];
    const double *restrict b = (const double *)args->data[1];
    double *restrict out = (double *)args->output;
    npy_intp i = 0;
    for (; i + 4 <= args->size; i += 4) {
        __m256d av = _mm256_loadu_pd(a + i);
        __m256d bv = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(out + i, _mm256_sub_pd(av, _mm256_mul_pd(bv, bv)));
    }
    for (; i < args->size; i++) {
        out[i] = a[i] - b[i] * b[i];
    }
}
#endif

static const NumpyJitAOTEntry numpy_jit_aot_square_sub_scalar = {
    numpy_jit_aot_square_sub_f64_scalar, 0x03, 0, 96,
    "square_sub_f64_scalar_exponent.scalar",
};

#if NUMPY_JIT_AOT_X86
static const NumpyJitAOTEntry numpy_jit_aot_square_sub_avx2 = {
    numpy_jit_aot_square_sub_f64_avx2, 0x03, 256, 128,
    "square_sub_f64_scalar_exponent.avx2",
};
#endif

static int
numpy_jit_aot_matches(const NumpyJitAOTManifestRecord *record,
                      const NumpyJitLLVMKernelSpec *spec)
{
    if (spec->graph_hash != record->graph_hash ||
        spec->dialect_version != 1 || spec->dtype != NPY_DOUBLE ||
        spec->scalar_mask != record->scalar_mask ||
        spec->input_count != 3 || spec->node_count != record->node_count ||
        spec->output_slot != 4) {
        return 0;
    }
    const NumpyJitKernelRegionNode *a = &spec->nodes[0];
    const NumpyJitKernelRegionNode *b = &spec->nodes[1];
    return a->op == record->node0_op && a->left == record->node0_left &&
           a->right == record->node0_right && a->out == record->node0_out &&
           b->op == record->node1_op && b->left == record->node1_left &&
           b->right == record->node1_right && b->out == record->node1_out;
}

const NumpyJitAOTEntry *
numpy_jit_aot_lookup(const NumpyJitLLVMKernelSpec *spec)
{
    if (spec == NULL) {
        return NULL;
    }
    for (int i = 0; i < NUMPY_JIT_AOT_MANIFEST_COUNT; i++) {
        const NumpyJitAOTManifestRecord *record = &numpy_jit_aot_manifest[i];
        if (!numpy_jit_aot_matches(record, spec)) {
            continue;
        }
#if NUMPY_JIT_AOT_X86
        __builtin_cpu_init();
        if (__builtin_cpu_supports("avx2")) {
            return &numpy_jit_aot_square_sub_avx2;
        }
#endif
        return &numpy_jit_aot_square_sub_scalar;
    }
    return NULL;
}

NumpyJitKernelFunc
numpy_jit_aot_entry_function(const NumpyJitAOTEntry *entry)
{
    return entry == NULL ? NULL : entry->function;
}

void
numpy_jit_aot_entry_info(
    const NumpyJitAOTEntry *entry, NumpyJitAOTEntryInfo *info)
{
    if (info == NULL) {
        return;
    }
    if (entry == NULL) {
        *info = (NumpyJitAOTEntryInfo){0, 0, 0, "none"};
        return;
    }
    *info = (NumpyJitAOTEntryInfo){
        entry->accepted_bucket_mask, entry->vector_bits, entry->code_size,
        entry->name,
    };
}

int
numpy_jit_aot_backend_available(void)
{
    return 1;
}
