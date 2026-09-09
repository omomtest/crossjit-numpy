#include "jit_aot_backend.h"

const NumpyJitAOTEntry *
numpy_jit_aot_lookup(const NumpyJitLLVMKernelSpec *spec)
{
    (void)spec;
    return NULL;
}

NumpyJitKernelFunc
numpy_jit_aot_entry_function(const NumpyJitAOTEntry *entry)
{
    (void)entry;
    return NULL;
}

void
numpy_jit_aot_entry_info(
    const NumpyJitAOTEntry *entry, NumpyJitAOTEntryInfo *info)
{
    (void)entry;
    if (info != NULL) {
        *info = (NumpyJitAOTEntryInfo){0, 0, 0, "disabled"};
    }
}

int
numpy_jit_aot_backend_available(void)
{
    return 0;
}
