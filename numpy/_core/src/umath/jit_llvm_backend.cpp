#include "jit_llvm_backend.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <unistd.h>

#include "llvm/ADT/StringMap.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ObjectTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Config/llvm-config.h"

using namespace llvm;
using namespace llvm::orc;

struct NumpyJitLLVMEntry {
    NumpyJitLLVMKernelSpec spec{};
    pid_t owner_pid{getpid()};
    std::atomic<int> state{NUMPY_JIT_LLVM_PENDING};
    std::atomic<uintptr_t> function{0};
    std::atomic<uint64_t> cache_uses{1};
    uint64_t queued_at_ns{0};
    uint64_t queue_ns{0};
    uint64_t compile_ns{0};
    uint64_t optimization_ns{0};
    uint64_t codegen_ns{0};
    uint64_t optimization_finished_at_ns{0};
    uint64_t object_bytes{0};
    uint32_t vector_bits{0};
    char error[160]{};
};

namespace {

constexpr size_t kMaximumEntries = 128;
constexpr uint64_t kMaximumObjectBytes = 4u * 1024u * 1024u;

static std::string
errorToString(Error error)
{
    std::string message;
    raw_string_ostream stream(message);
    logAllUnhandledErrors(std::move(error), stream, "");
    return stream.str();
}

static uint64_t
monotonicNs()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

static bool
supportedDType(int dtype)
{
    return dtype == NPY_FLOAT || dtype == NPY_DOUBLE ||
           dtype == NPY_INT || dtype == NPY_LONG || dtype == NPY_LONGLONG;
}

static bool
isFloatingDType(int dtype)
{
    return dtype == NPY_FLOAT || dtype == NPY_DOUBLE;
}

class Backend {
public:
    Backend()
    {
        /* The cache is process-local, but retaining the target in the key
         * keeps its identity complete if it is later persisted or shared by
         * a higher-level cache.  Host detection is intentionally much
         * cheaper than constructing LLJIT and happens only after a graph has
         * crossed the adapter's compilation threshold. */
        target_key_ = sys::getHostCPUName().str();
        std::snprintf(capacity_entry_.error, sizeof(capacity_entry_.error),
                      "%s", "code_cache_capacity");
        capacity_entry_.state.store(
            NUMPY_JIT_LLVM_CAPACITY, std::memory_order_relaxed);
        StringMap<bool> features;
#if LLVM_VERSION_MAJOR >= 21
        features = sys::getHostCPUFeatures();
        {
#else
        if (sys::getHostCPUFeatures(features)) {
#endif
            for (const auto &feature : features) {
                target_key_ += feature.getValue() ? "+" : "-";
                target_key_ += feature.getKey().str();
            }
        }
    }

    NumpyJitLLVMEntry *request(
        const NumpyJitLLVMKernelSpec &spec, int *cache_hit)
    {
        if (cache_hit != nullptr) {
            *cache_hit = 0;
        }
        std::string key(reinterpret_cast<const char *>(&spec), sizeof(spec));
        key.append(target_key_);
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = cache_.find(key);
        if (found != cache_.end()) {
            found->second->cache_uses.fetch_add(1, std::memory_order_relaxed);
            if (cache_hit != nullptr) {
                *cache_hit = 1;
            }
            return found->second;
        }
        if (entries_.size() >= kMaximumEntries ||
            object_bytes_.load(std::memory_order_relaxed) >=
                kMaximumObjectBytes) {
            return &capacity_entry_;
        }
        auto entry = std::make_unique<NumpyJitLLVMEntry>();
        entry->spec = spec;
        entry->queued_at_ns = monotonicNs();
        NumpyJitLLVMEntry *raw = entry.get();
        entries_.push_back(std::move(entry));
        cache_.emplace(std::move(key), raw);
        queue_.push_back(raw);
        if (!worker_started_) {
            worker_started_ = true;
            /* Backend intentionally has process lifetime.  NumPy's core
             * extension is not unloaded during normal interpreter teardown,
             * and avoiding a static std::thread destructor also makes forked
             * children safe. */
            std::thread([this] { workerLoop(); }).detach();
        }
        condition_.notify_one();
        return raw;
    }

private:
    Error initialize()
    {
        InitializeNativeTarget();
        InitializeNativeTargetAsmPrinter();
        auto builder = LLJITBuilder();
        auto detected = JITTargetMachineBuilder::detectHost();
        if (!detected) {
            return detected.takeError();
        }
        detected->setCodeGenOptLevel(CodeGenOptLevel::Default);
        builder.setJITTargetMachineBuilder(std::move(*detected));
        auto created = builder.create();
        if (!created) {
            return created.takeError();
        }
        jit_ = std::move(*created);
        jit_->getIRTransformLayer().setTransform(
            [this](ThreadSafeModule module,
               const MaterializationResponsibility &) -> Expected<ThreadSafeModule> {
                uint64_t begin = monotonicNs();
                module.withModuleDo([](Module &m) {
                    LoopAnalysisManager lam;
                    FunctionAnalysisManager fam;
                    CGSCCAnalysisManager cgam;
                    ModuleAnalysisManager mam;
                    PassBuilder pb;
                    pb.registerModuleAnalyses(mam);
                    pb.registerCGSCCAnalyses(cgam);
                    pb.registerFunctionAnalyses(fam);
                    pb.registerLoopAnalyses(lam);
                    pb.crossRegisterProxies(lam, fam, cgam, mam);
                    ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(
                        OptimizationLevel::O2);
                    mpm.run(m, mam);
                });
                if (active_entry_ != nullptr) {
                    active_entry_->optimization_ns = monotonicNs() - begin;
                    active_entry_->optimization_finished_at_ns = monotonicNs();
                }
                return std::move(module);
            });
        jit_->getObjTransformLayer().setTransform(
            [this](std::unique_ptr<MemoryBuffer> object)
                -> Expected<std::unique_ptr<MemoryBuffer>> {
                if (active_entry_ != nullptr) {
                    active_entry_->object_bytes = object->getBufferSize();
                    if (active_entry_->optimization_finished_at_ns != 0) {
                        active_entry_->codegen_ns = monotonicNs() -
                            active_entry_->optimization_finished_at_ns;
                    }
                    object_bytes_.fetch_add(
                        active_entry_->object_bytes,
                        std::memory_order_relaxed);
                }
                return std::move(object);
            });
        return Error::success();
    }

    Type *elementType(LLVMContext &context, int dtype)
    {
        if (dtype == NPY_FLOAT) {
            return Type::getFloatTy(context);
        }
        if (dtype == NPY_DOUBLE) {
            return Type::getDoubleTy(context);
        }
        if (dtype == NPY_INT) {
            return Type::getInt32Ty(context);
        }
        return Type::getInt64Ty(context);
    }

    Value *loadScalar(IRBuilder<> &builder, Value *scalarArray,
                      unsigned slot, Type *element)
    {
        LLVMContext &context = builder.getContext();
        Value *index = ConstantInt::get(Type::getInt64Ty(context), slot);
        Value *address = builder.CreateInBoundsGEP(
            Type::getInt64Ty(context), scalarArray, index);
        Value *bits = builder.CreateLoad(Type::getInt64Ty(context), address);
        if (element->isFloatTy()) {
            return builder.CreateBitCast(
                builder.CreateTrunc(bits, Type::getInt32Ty(context)), element);
        }
        if (element->isDoubleTy()) {
            return builder.CreateBitCast(bits, element);
        }
        if (element->isIntegerTy(32)) {
            return builder.CreateTrunc(bits, element);
        }
        return bits;
    }

    Expected<std::pair<ThreadSafeModule, std::string>> buildModule(
        const NumpyJitLLVMKernelSpec &spec, uint64_t serial)
    {
        if (!supportedDType(spec.dtype) || spec.node_count < 2 ||
            spec.node_count > NUMPY_JIT_LLVM_MAX_NODES ||
            spec.input_count == 0 ||
            spec.input_count > NUMPY_JIT_LLVM_MAX_INPUTS ||
            spec.output_slot >= NUMPY_JIT_KERNEL_REGION_MAX_SLOTS) {
            return createStringError(inconvertibleErrorCode(),
                                     "unsupported kernel specification");
        }
        auto context = std::make_unique<LLVMContext>();
        LLVMContext &ctx = *context;
        std::string symbol = "numpy_region_" + std::to_string(serial);
        auto module = std::make_unique<Module>(symbol, ctx);
        module->setDataLayout(jit_->getDataLayout());
        module->setTargetTriple(jit_->getTargetTriple());

        Type *ptrTy = PointerType::getUnqual(ctx);
        Type *i64Ty = Type::getInt64Ty(ctx);
        ArrayType *dataTy = ArrayType::get(ptrTy, NUMPY_JIT_LLVM_MAX_INPUTS);
        ArrayType *scalarTy = ArrayType::get(i64Ty, NUMPY_JIT_LLVM_MAX_INPUTS);
        StructType *argsTy = StructType::create(
            ctx, {dataTy, scalarTy, ptrTy, i64Ty}, "NumpyJitKernelArgs");
        FunctionType *functionTy = FunctionType::get(
            Type::getVoidTy(ctx), {ptrTy}, false);
        Function *function = Function::Create(
            functionTy, Function::ExternalLinkage, symbol, *module);
        function->addParamAttr(0, Attribute::NonNull);

        StringMap<bool> hostFeatures;
        std::string featureString;
#if LLVM_VERSION_MAJOR >= 21
        hostFeatures = sys::getHostCPUFeatures();
        {
#else
        if (sys::getHostCPUFeatures(hostFeatures)) {
#endif
            for (const auto &feature : hostFeatures) {
                if (!featureString.empty()) {
                    featureString += ',';
                }
                featureString += feature.getValue() ? '+' : '-';
                featureString += feature.getKey().str();
            }
        }
        std::string cpu = sys::getHostCPUName().str();
        if (!cpu.empty()) {
            function->addFnAttr("target-cpu", cpu);
        }
        if (!featureString.empty()) {
            function->addFnAttr("target-features", featureString);
        }
        function->addFnAttr("no-infs-fp-math", "false");
        function->addFnAttr("no-nans-fp-math", "false");
        function->addFnAttr("unsafe-fp-math", "false");
        function->addFnAttr("fp-contract", "off");

        BasicBlock *entry = BasicBlock::Create(ctx, "entry", function);
        BasicBlock *loop = BasicBlock::Create(ctx, "loop", function);
        BasicBlock *exit = BasicBlock::Create(ctx, "exit", function);
        IRBuilder<> builder(entry);
        Value *args = function->getArg(0);
        Value *dataField = builder.CreateStructGEP(argsTy, args, 0);
        Value *scalarField = builder.CreateStructGEP(argsTy, args, 1);
        Value *outputField = builder.CreateStructGEP(argsTy, args, 2);
        Value *sizeField = builder.CreateStructGEP(argsTy, args, 3);
        Value *output = builder.CreateLoad(ptrTy, outputField, "output");
        Value *size = builder.CreateLoad(i64Ty, sizeField, "size");
        Value *empty = builder.CreateICmpEQ(size, ConstantInt::get(i64Ty, 0));
        builder.CreateCondBr(empty, exit, loop);

        builder.SetInsertPoint(loop);
        PHINode *index = builder.CreatePHI(i64Ty, 2, "index");
        index->addIncoming(ConstantInt::get(i64Ty, 0), entry);
        Type *element = elementType(ctx, spec.dtype);
        Value *values[NUMPY_JIT_KERNEL_REGION_MAX_SLOTS] = {};
        for (unsigned slot = 0; slot < spec.input_count; slot++) {
            if (spec.scalar_mask & (1u << slot)) {
                values[slot] = loadScalar(builder, scalarField, slot, element);
            }
            else {
                Value *slotIndex = ConstantInt::get(i64Ty, slot);
                Value *dataAddress = builder.CreateInBoundsGEP(
                    dataTy, dataField,
                    {ConstantInt::get(i64Ty, 0), slotIndex});
                Value *base = builder.CreateLoad(ptrTy, dataAddress);
                Value *address = builder.CreateInBoundsGEP(
                    element, base, index);
                values[slot] = builder.CreateLoad(element, address);
            }
        }
        bool floating = isFloatingDType(spec.dtype);
        for (unsigned nodeIndex = 0; nodeIndex < spec.node_count; nodeIndex++) {
            const NumpyJitKernelRegionNode &node = spec.nodes[nodeIndex];
            if (node.left >= NUMPY_JIT_KERNEL_REGION_MAX_SLOTS ||
                node.right >= NUMPY_JIT_KERNEL_REGION_MAX_SLOTS ||
                node.out >= NUMPY_JIT_KERNEL_REGION_MAX_SLOTS ||
                values[node.left] == nullptr ||
                (node.op != NUMPY_JIT_KERNEL_REGION_NEGATIVE &&
                 values[node.right] == nullptr)) {
                return createStringError(inconvertibleErrorCode(),
                                         "invalid DAG value slot");
            }
            Value *left = values[node.left];
            Value *right = values[node.right];
            Value *value = nullptr;
            switch (node.op) {
                case NUMPY_JIT_KERNEL_REGION_ADD:
                    value = floating ? builder.CreateFAdd(left, right) :
                                       builder.CreateAdd(left, right);
                    break;
                case NUMPY_JIT_KERNEL_REGION_SUB:
                    value = floating ? builder.CreateFSub(left, right) :
                                       builder.CreateSub(left, right);
                    break;
                case NUMPY_JIT_KERNEL_REGION_MUL:
                    value = floating ? builder.CreateFMul(left, right) :
                                       builder.CreateMul(left, right);
                    break;
                case NUMPY_JIT_KERNEL_REGION_DIV:
                    if (!floating) {
                        return createStringError(inconvertibleErrorCode(),
                                                 "integer divide unsupported");
                    }
                    value = builder.CreateFDiv(left, right);
                    break;
                case NUMPY_JIT_KERNEL_REGION_SQUARE:
                    value = floating ? builder.CreateFMul(left, left) :
                                       builder.CreateMul(left, left);
                    break;
                case NUMPY_JIT_KERNEL_REGION_NEGATIVE:
                    value = floating ? builder.CreateFNeg(left) :
                                       builder.CreateNeg(left);
                    break;
                default:
                    return createStringError(inconvertibleErrorCode(),
                                             "unsupported Region operation");
            }
            values[node.out] = value;
        }
        if (values[spec.output_slot] == nullptr) {
            return createStringError(inconvertibleErrorCode(),
                                     "missing output value");
        }
        Value *outputAddress = builder.CreateInBoundsGEP(
            element, output, index);
        builder.CreateStore(values[spec.output_slot], outputAddress);
        Value *next = builder.CreateAdd(index, ConstantInt::get(i64Ty, 1));
        Value *done = builder.CreateICmpEQ(next, size);
        builder.CreateCondBr(done, exit, loop);
        index->addIncoming(next, loop);

        builder.SetInsertPoint(exit);
        builder.CreateRetVoid();
        return std::make_pair(
            ThreadSafeModule(std::move(module), std::move(context)), symbol);
    }

    void failEntry(NumpyJitLLVMEntry *entry, const std::string &message)
    {
        std::snprintf(entry->error, sizeof(entry->error), "%s",
                      message.c_str());
        entry->state.store(NUMPY_JIT_LLVM_FAILED, std::memory_order_release);
    }

    void compile(NumpyJitLLVMEntry *entry)
    {
        auto begin = std::chrono::steady_clock::now();
        uint64_t begin_ns = monotonicNs();
        entry->queue_ns = begin_ns >= entry->queued_at_ns ?
            begin_ns - entry->queued_at_ns : 0;
        if (!jit_) {
            if (Error error = initialize()) {
                failEntry(entry, errorToString(std::move(error)));
                return;
            }
        }
        uint64_t serial = next_symbol_++;
        auto built = buildModule(entry->spec, serial);
        if (!built) {
            failEntry(entry, errorToString(built.takeError()));
            return;
        }
        ThreadSafeModule module = std::move(built->first);
        std::string symbol = std::move(built->second);
        active_entry_ = entry;
        if (Error error = jit_->addIRModule(std::move(module))) {
            active_entry_ = nullptr;
            failEntry(entry, errorToString(std::move(error)));
            return;
        }
        auto address = jit_->lookup(symbol);
        active_entry_ = nullptr;
        if (!address) {
            failEntry(entry, errorToString(address.takeError()));
            return;
        }
        auto end = std::chrono::steady_clock::now();
        entry->compile_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                .count());
        StringMap<bool> features;
#if LLVM_VERSION_MAJOR >= 21
        features = sys::getHostCPUFeatures();
        {
#else
        if (sys::getHostCPUFeatures(features)) {
#endif
            if (features.lookup("avx512f")) {
                entry->vector_bits = 512;
            }
            else if (features.lookup("avx2")) {
                entry->vector_bits = 256;
            }
            else {
                entry->vector_bits = 128;
            }
        }
        entry->function.store(
            static_cast<uintptr_t>(address->getValue()),
            std::memory_order_release);
        entry->state.store(NUMPY_JIT_LLVM_READY, std::memory_order_release);
    }

    void workerLoop()
    {
        for (;;) {
            NumpyJitLLVMEntry *entry = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this] { return !queue_.empty(); });
                entry = queue_.front();
                queue_.pop_front();
            }
            compile(entry);
        }
    }

    std::mutex mutex_;
    std::condition_variable condition_;
    std::unordered_map<std::string, NumpyJitLLVMEntry *> cache_;
    std::vector<std::unique_ptr<NumpyJitLLVMEntry>> entries_;
    std::deque<NumpyJitLLVMEntry *> queue_;
    std::unique_ptr<LLJIT> jit_;
    std::atomic<uint64_t> object_bytes_{0};
    uint64_t next_symbol_{1};
    bool worker_started_{false};
    NumpyJitLLVMEntry *active_entry_{nullptr};
    std::string target_key_;
    NumpyJitLLVMEntry capacity_entry_;
};

static Backend &backend()
{
    /* The parent backend is deliberately leaked across fork: its published
     * code remains executable.  A child gets a fresh mutex, LLJIT instance
     * and worker the first time it requests a new specialization. */
    static Backend *instance = new Backend();
    static pid_t instance_pid = getpid();
    pid_t current_pid = getpid();
    if (current_pid != instance_pid) {
        instance = new Backend();
        instance_pid = current_pid;
    }
    return *instance;
}

}  // namespace

extern "C" NumpyJitLLVMEntry *
numpy_jit_llvm_request(const NumpyJitLLVMKernelSpec *spec, int *cache_hit)
{
    if (spec == nullptr) {
        return nullptr;
    }
    return backend().request(*spec, cache_hit);
}

extern "C" int
numpy_jit_llvm_entry_state(const NumpyJitLLVMEntry *entry)
{
    if (entry == nullptr) {
        return NUMPY_JIT_LLVM_FAILED;
    }
    int state = entry->state.load(std::memory_order_acquire);
    if (state == NUMPY_JIT_LLVM_PENDING && entry->owner_pid != getpid()) {
        return NUMPY_JIT_LLVM_ORPHANED;
    }
    return state;
}

extern "C" NumpyJitKernelFunc
numpy_jit_llvm_entry_function(const NumpyJitLLVMEntry *entry)
{
    if (entry == nullptr ||
        entry->state.load(std::memory_order_acquire) != NUMPY_JIT_LLVM_READY) {
        return nullptr;
    }
    uintptr_t address = entry->function.load(std::memory_order_acquire);
    return reinterpret_cast<NumpyJitKernelFunc>(address);
}

extern "C" void
numpy_jit_llvm_entry_info(
    const NumpyJitLLVMEntry *entry, NumpyJitLLVMEntryInfo *info)
{
    if (info == nullptr) {
        return;
    }
    if (entry == nullptr) {
        *info = NumpyJitLLVMEntryInfo{
            0, 0, 0, 0, 0, 0, 0,
            NUMPY_JIT_LLVM_FAILED, "null_entry"};
        return;
    }
    /* The worker publishes every non-atomic telemetry field before the
     * release store to state.  Acquire state first, and never inspect those
     * fields while the worker can still be updating them. */
    int state = numpy_jit_llvm_entry_state(entry);
    if (state == NUMPY_JIT_LLVM_PENDING ||
        state == NUMPY_JIT_LLVM_ORPHANED) {
        *info = NumpyJitLLVMEntryInfo{
            0, 0, 0, 0, 0,
            entry->cache_uses.load(std::memory_order_relaxed),
            0, state,
            state == NUMPY_JIT_LLVM_ORPHANED ? "orphaned_after_fork" :
                                               "pending"};
        return;
    }
    *info = NumpyJitLLVMEntryInfo{
        entry->queue_ns,
        entry->compile_ns,
        entry->optimization_ns,
        entry->codegen_ns,
        entry->object_bytes,
        entry->cache_uses.load(std::memory_order_relaxed),
        entry->vector_bits,
        state,
        entry->error,
    };
}

extern "C" int
numpy_jit_llvm_backend_available(void)
{
    return 1;
}

extern "C" const char *
numpy_jit_llvm_backend_name(void)
{
    return "llvm-orc";
}
