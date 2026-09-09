#ifndef NUMPY_CORE_SRC_UMATH_JIT_AOT_BACKEND_H
#define NUMPY_CORE_SRC_UMATH_JIT_AOT_BACKEND_H

/* AOT stencils deliberately use the same POD call ABI and canonical key as
 * ORC entries.  The only distinction is when machine code is produced. */
#include "jit_llvm_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NumpyJitAOTEntry NumpyJitAOTEntry;

typedef struct {
    uint8_t accepted_bucket_mask;
    uint16_t vector_bits;
    uint16_t code_size;
    const char *name;
} NumpyJitAOTEntryInfo;

const NumpyJitAOTEntry *numpy_jit_aot_lookup(
    const NumpyJitLLVMKernelSpec *spec);
NumpyJitKernelFunc numpy_jit_aot_entry_function(
    const NumpyJitAOTEntry *entry);
void numpy_jit_aot_entry_info(
    const NumpyJitAOTEntry *entry, NumpyJitAOTEntryInfo *info);
int numpy_jit_aot_backend_available(void);

#ifdef __cplusplus
}
#endif

#endif
