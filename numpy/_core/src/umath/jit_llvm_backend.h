#ifndef NUMPY_CORE_SRC_UMATH_JIT_LLVM_BACKEND_H
#define NUMPY_CORE_SRC_UMATH_JIT_LLVM_BACKEND_H

#include <stdint.h>

#include "numpy/ndarraytypes.h"
#include "jit_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NUMPY_JIT_LLVM_MAX_INPUTS NUMPY_JIT_KERNEL_REGION_MAX_INPUTS
#define NUMPY_JIT_LLVM_MAX_NODES NUMPY_JIT_KERNEL_REGION_MAX_NODES

typedef struct {
    const void *data[NUMPY_JIT_LLVM_MAX_INPUTS];
    uint64_t scalars[NUMPY_JIT_LLVM_MAX_INPUTS];
    void *output;
    npy_intp size;
} NumpyJitKernelArgs;

typedef void (*NumpyJitKernelFunc)(const NumpyJitKernelArgs *args);

/* This structure is also the byte-for-byte global cache key.  Callers must
 * zero initialize it before assigning fields. */
typedef struct {
    uint64_t graph_hash;
    uint64_t dialect_id;
    uint16_t dialect_version;
    uint8_t node_count;
    uint8_t input_count;
    uint8_t output_slot;
    uint8_t scalar_mask;
    int16_t dtype;
    uint16_t reserved;
    NumpyJitKernelRegionNode nodes[NUMPY_JIT_LLVM_MAX_NODES];
} NumpyJitLLVMKernelSpec;

typedef struct NumpyJitLLVMEntry NumpyJitLLVMEntry;

enum {
    NUMPY_JIT_LLVM_PENDING = 0,
    NUMPY_JIT_LLVM_READY = 1,
    NUMPY_JIT_LLVM_FAILED = 2,
    NUMPY_JIT_LLVM_CAPACITY = 3,
    NUMPY_JIT_LLVM_ORPHANED = 4,
};

typedef struct {
    uint64_t queue_ns;
    uint64_t compile_ns;
    uint64_t optimization_ns;
    uint64_t codegen_ns;
    uint64_t object_bytes;
    uint64_t cache_uses;
    uint32_t vector_bits;
    int state;
    const char *error;
} NumpyJitLLVMEntryInfo;

/* Queue a cache lookup/compilation.  Compilation always happens on the
 * backend worker and never on the calling Python thread.  Entries are owned
 * by the process-wide cache and remain valid for the process lifetime. */
NumpyJitLLVMEntry *numpy_jit_llvm_request(
    const NumpyJitLLVMKernelSpec *spec, int *cache_hit);

int numpy_jit_llvm_entry_state(const NumpyJitLLVMEntry *entry);
NumpyJitKernelFunc numpy_jit_llvm_entry_function(
    const NumpyJitLLVMEntry *entry);
void numpy_jit_llvm_entry_info(
    const NumpyJitLLVMEntry *entry, NumpyJitLLVMEntryInfo *info);

int numpy_jit_llvm_backend_available(void);
const char *numpy_jit_llvm_backend_name(void);

#ifdef __cplusplus
}
#endif

#endif
