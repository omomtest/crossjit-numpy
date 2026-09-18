#include "jit_aot_backend.h"

#include <string.h>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define NUMPY_JIT_AOT_X86 1
#else
#define NUMPY_JIT_AOT_X86 0
#endif

struct NumpyJitAOTEntry {
    NumpyJitKernelFunc function;
    uint8_t accepted_bucket_mask;
    uint16_t vector_bits;
    uint16_t code_size;
    const char *name;
};

typedef struct {
    uint64_t graph_hash;
    int16_t dtype;
    uint8_t scalar_mask;
    uint8_t accepted_bucket_mask;
    uint8_t input_count;
    uint8_t output_slot;
    uint8_t node_count;
    NumpyJitKernelRegionNode nodes[8];
    NumpyJitAOTEntry scalar_entry;
    NumpyJitAOTEntry avx2_entry;
} NumpyJitAOTManifestRecord;

#include "jit_aot_manifest_generated.h"

static int
numpy_jit_aot_matches(const NumpyJitAOTManifestRecord *record,
                      const NumpyJitLLVMKernelSpec *spec)
{
    if (spec->graph_hash != record->graph_hash ||
        spec->dialect_version != 1 || spec->dtype != record->dtype ||
        spec->scalar_mask != record->scalar_mask ||
        spec->input_count != record->input_count ||
        spec->node_count != record->node_count ||
        spec->output_slot != record->output_slot) {
        return 0;
    }
    for (uint8_t i = 0; i < record->node_count; i++) {
        const NumpyJitKernelRegionNode *actual = &spec->nodes[i];
        const NumpyJitKernelRegionNode *expected = &record->nodes[i];
        if (actual->op != expected->op || actual->left != expected->left ||
            actual->right != expected->right ||
            actual->third != expected->third || actual->out != expected->out) {
            return 0;
        }
    }
    return 1;
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
        if (__builtin_cpu_supports("avx2") &&
            record->avx2_entry.function != NULL) {
            return &record->avx2_entry;
        }
#endif
        return &record->scalar_entry;
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
