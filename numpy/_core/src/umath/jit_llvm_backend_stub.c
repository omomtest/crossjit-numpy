#include "jit_llvm_backend.h"

NumpyJitLLVMEntry *
numpy_jit_llvm_request(const NumpyJitLLVMKernelSpec *spec, int *cache_hit)
{
    (void)spec;
    if (cache_hit != NULL) {
        *cache_hit = 0;
    }
    return NULL;
}

int
numpy_jit_llvm_entry_state(const NumpyJitLLVMEntry *entry)
{
    (void)entry;
    return NUMPY_JIT_LLVM_FAILED;
}

NumpyJitKernelFunc
numpy_jit_llvm_entry_function(const NumpyJitLLVMEntry *entry)
{
    (void)entry;
    return NULL;
}

void
numpy_jit_llvm_entry_info(
    const NumpyJitLLVMEntry *entry, NumpyJitLLVMEntryInfo *info)
{
    (void)entry;
    if (info != NULL) {
        *info = (NumpyJitLLVMEntryInfo){
            .state = NUMPY_JIT_LLVM_FAILED,
            .error = "backend_disabled",
        };
    }
}

int
numpy_jit_llvm_backend_available(void)
{
    return 0;
}

const char *
numpy_jit_llvm_backend_name(void)
{
    return "disabled";
}
