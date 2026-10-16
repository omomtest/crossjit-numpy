/* Provider-private adapters for the initialized builtin NumPy loop table. */
typedef struct {
    PyUFuncObject *ufunc;
    int count;
    PyUFuncGenericFunction functions[64];
    void *data[64];
} NumpyCrossJitBuiltin;
static NumpyCrossJitBuiltin numpy_crossjit_builtins[11];

static void
numpy_crossjit_capture_builtins(void)
{
    PyObject *objects[]={crossjit_state.add,crossjit_state.subtract,crossjit_state.multiply,
        crossjit_state.true_divide,crossjit_state.negative,crossjit_state.minimum,crossjit_state.maximum,
        crossjit_state.sqrt,crossjit_state.exp,crossjit_state.tanh,crossjit_state.matmul};
    for (int i=0;i<11;i++) {
        PyUFuncObject *ufunc=(PyUFuncObject *)objects[i];
        NumpyCrossJitBuiltin *trusted=&numpy_crossjit_builtins[i];
        trusted->ufunc=ufunc;
        if (!ufunc || ufunc->ntypes>64) continue;
        trusted->count=ufunc->ntypes;
        memcpy(trusted->functions,ufunc->functions,ufunc->ntypes*sizeof(trusted->functions[0]));
        memcpy(trusted->data,ufunc->data,ufunc->ntypes*sizeof(trusted->data[0]));
    }
}

static int
numpy_crossjit_builtin_unchanged(int i)
{
    NumpyCrossJitBuiltin *trusted=&numpy_crossjit_builtins[i];
    PyUFuncObject *ufunc=trusted->ufunc;
    return ufunc && trusted->count>0 && ufunc->ntypes==trusted->count && ufunc->userloops==NULL &&
        memcmp(ufunc->functions,trusted->functions,trusted->count*sizeof(trusted->functions[0]))==0 &&
        memcmp(ufunc->data,trusted->data,trusted->count*sizeof(trusted->data[0]))==0;
}

static void
numpy_crossjit_select_native_loop(uint8_t op,int dtype,NumpyCrossJitNativeLoop *loop)
{
    PyObject *objects[]={crossjit_state.add,crossjit_state.subtract,
        crossjit_state.multiply,crossjit_state.true_divide,crossjit_state.negative,
        crossjit_state.minimum,crossjit_state.maximum};
    if (op>=sizeof(objects)/sizeof(objects[0]) || !numpy_crossjit_builtin_unchanged(op)) return;
    PyUFuncObject *ufunc=(PyUFuncObject *)objects[op];
    if (!ufunc || ufunc->nout!=1 || ufunc->nin!=(op==NUMPY_CROSSJIT_KERNEL_NEGATIVE?1:2)) return;
    for (int i=0;i<ufunc->ntypes;i++) {
        int same=1;
        for (int j=0;j<ufunc->nargs;j++) same&=ufunc->types[i*ufunc->nargs+j]==dtype;
        if (same && ufunc->functions[i] && ufunc->data[i]==NULL) {
            loop->ufunc=ufunc;
            loop->entry=ufunc->functions[i];
            loop->index=i;
            return;
        }
    }
}

static int
numpy_crossjit_native_loop(const NumpyCrossJitNativeLoop *loop,const NumpyCrossJitKernelNode *node,
    const void *const *inputs,
    const npy_intp strides[][NPY_MAXDIMS],int rank,const npy_intp *shape,
    char *output,npy_intp size,npy_intp width)
{
    PyUFuncObject *ufunc=loop->ufunc;
    if (ufunc->functions[loop->index]!=loop->entry ||
        ufunc->data[loop->index]!=NULL) return 0;
    int slots[2]={node->left,node->right};
    return cj_invoke_strided_backend(loop->entry,NULL,slots,ufunc->nin,
                                    inputs,strides,rank,shape,output,size,width);
}
