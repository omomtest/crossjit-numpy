/* NumPy domain validation, capability binding and runtime object bridge. */
static int numpy_crossjit_stage_contiguous(NumpyCrossJitArrayFact *);
static int
numpy_crossjit_stage_c_order(const NumpyCrossJitArrayFact *fact)
{
    npy_uintp previous=NPY_MAX_UINTP;
    for (uint8_t axis=0;axis<fact->ndim;axis++) {
        npy_intp stride=fact->strides[axis];
        if (fact->dimensions[axis]<=1 || stride==0) continue;
        npy_uintp magnitude=stride<0?(npy_uintp)0-(npy_uintp)stride:(npy_uintp)stride;
        if (magnitude>previous) return 0;
        previous=magnitude;
    }
    return 1;
}
static int
numpy_crossjit_constant_fact(PyObject *object, NumpyCrossJitArrayFact *fact)
{
    memset(fact,0,sizeof(*fact));
    int kind=0;
    if (object && object==npy_static_pydata._NoValue) kind=5;
    else if (object==Py_None) kind=2;
    else if (PyBool_Check(object)) {kind=3;fact->ndim=1;fact->dimensions[0]=object==Py_True;}
    else if (PyLong_CheckExact(object)) {
        kind=1;fact->ndim=1;fact->dimensions[0]=PyLong_AsLongLong(object);
        if (PyErr_Occurred()) {PyErr_Clear();return 0;}
    }
    else if (PyTuple_CheckExact(object) && PyTuple_GET_SIZE(object)<=NPY_MAXDIMS) {
        kind=4;fact->ndim=PyTuple_GET_SIZE(object);
        for (uint8_t i=0;i<fact->ndim;i++) {
            PyObject *item=PyTuple_GET_ITEM(object,i);
            if (!PyLong_CheckExact(item)) return 0;
            fact->dimensions[i]=PyLong_AsLongLong(item);
            if (PyErr_Occurred()) {PyErr_Clear();return 0;}
        }
    }
    fact->typenum=-kind;
    return kind;
}

static int
numpy_crossjit_stage_reduction(const PyJitNodeRef *node,
    const NumpyCrossJitArrayFact *input, NumpyCrossJitStage *stage)
{
    int positional=PyJit_NodeGetPositionalCount(node);
    int count=PyJit_NodeGetOperandCount(node);
    PyObject *names=PyJit_NodeGetKeywordNames(node);
    PyObject *axis=NULL,*keep=NULL;
    int public_reduce=stage->semantic_op>=NUMPY_CROSSJIT_SUM;
    int maximum=stage->semantic_op==NUMPY_CROSSJIT_MAX || stage->semantic_op==NUMPY_CROSSJIT_AMAX;
    if (positional>(maximum?6:7)) return 0;
    for (int arg=1;arg<count;arg++) {
        PyObject *value=PyJit_ValueGetObservedObject(PyJit_NodeGetOperand(node,arg));
        if (!value) return 0;
        const char *name=NULL;
        if (arg<positional) {
            static const char *sum_names[]={"axis","dtype","out","keepdims","initial","where"};
            static const char *max_names[]={"axis","out","keepdims","initial","where"};
            name=(maximum?max_names:sum_names)[arg-1];
        }
        else {
            PyObject *key=PyTuple_GET_ITEM(names,arg-positional);
            if (!PyUnicode_CheckExact(key)) return 0;
            name=PyUnicode_AsUTF8(key);
            if (!name) return 0;
        }
        if (strcmp(name,"axis")==0) {if (axis) return 0;axis=value;}
        else if (strcmp(name,"keepdims")==0) {if (keep) return 0;keep=value;}
        else if (strcmp(name,"dtype")==0 || strcmp(name,"out")==0) {if (value!=Py_None) return 0;}
        else if (strcmp(name,"initial")==0) {if (!value || value!=npy_static_pydata._NoValue) return 0;}
        else if (strcmp(name,"where")==0) {if (value!=Py_True && !(public_reduce && value==npy_static_pydata._NoValue)) return 0;}
        else return 0;
    }
    stage->keepdims=keep==Py_True;
    if (keep && keep!=Py_True && keep!=Py_False && !(public_reduce && keep==npy_static_pydata._NoValue)) return 0;
    npy_bool flags[NPY_MAXDIMS]={0};
    if (axis==Py_None || (!axis && public_reduce)) {
        for (uint8_t i=0;i<input->ndim;i++) stage->axes[stage->axis_count++]=i;
    }
    else if (!axis || PyLong_CheckExact(axis)) {
        long long value=axis?PyLong_AsLongLong(axis):0;
        if (PyErr_Occurred()) {PyErr_Clear();return 0;}
        if (value<0) value+=input->ndim;
        if (value<0 || value>=input->ndim) return 0;
        stage->axes[stage->axis_count++]=value;
    }
    else if (PyTuple_CheckExact(axis) && PyTuple_GET_SIZE(axis)<=input->ndim) {
        for (Py_ssize_t i=0;i<PyTuple_GET_SIZE(axis);i++) {
            PyObject *value=PyTuple_GET_ITEM(axis,i);
            if (!PyLong_CheckExact(value)) return 0;
            long long ax=PyLong_AsLongLong(value);
            if (PyErr_Occurred()) {PyErr_Clear();return 0;}
            if (ax<0) ax+=input->ndim;
            if (ax<0 || ax>=input->ndim) return 0;
            stage->axes[stage->axis_count++]=ax;
        }
    }
    else return 0;
    for (uint8_t i=0;i<stage->axis_count;i++) {
        int axis_value=stage->axes[i];
        if (flags[axis_value] || input->dimensions[axis_value]==0) return 0;
        flags[axis_value]=1;
    }
    /* Output KEEPORDER is currently proven only for C-compatible input axes. */
    npy_uintp previous=NPY_MAX_UINTP;
    for (uint8_t i=0;i<input->ndim;i++) {
        npy_intp stride=input->strides[i];
        npy_uintp magnitude=stride<0?(npy_uintp)0-(npy_uintp)stride:(npy_uintp)stride;
        if (input->dimensions[i]>1 && stride) {if (magnitude>previous) return 0;previous=magnitude;}
    }
    stage->domain.typenum=input->typenum;
    for (uint8_t i=0;i<input->ndim;i++) {
        if (!flags[i] || stage->keepdims) stage->domain.dimensions[stage->domain.ndim++]=flags[i]?1:input->dimensions[i];
    }
    stage->scalar_result=stage->domain.ndim==0;
    return numpy_crossjit_stage_contiguous(&stage->domain);
}

static int
numpy_crossjit_stage_contiguous(NumpyCrossJitArrayFact *fact)
{
    npy_intp stride = numpy_crossjit_dtype_width(fact->typenum);
    fact->c_contiguous = fact->native_byteorder = fact->aligned = 1;
    for (int axis = fact->ndim-1; axis >= 0; axis--) {
        fact->strides[axis] = stride;
        if (fact->dimensions[axis] < 0 ||
                (fact->dimensions[axis] && stride > NPY_MAX_INTP/fact->dimensions[axis])) return 0;
        stride *= fact->dimensions[axis];
    }
    return 1;
}

static int
numpy_crossjit_stage_shape(NumpyCrossJitOp op,
                          const NumpyCrossJitArrayFact *left,
                          const NumpyCrossJitArrayFact *right,
                          NumpyCrossJitStage *stage)
{
    NumpyCrossJitArrayFact *out = &stage->domain;
    memset(out, 0, sizeof(*out));
    if (left->typenum != right->typenum ||
            !numpy_crossjit_supported_dtype(left->typenum)) return 0;
    out->typenum = left->typenum;
    if (op == NUMPY_CROSSJIT_MATMUL || op==NUMPY_CROSSJIT_DOT || op==NUMPY_CROSSJIT_DOT_PUBLIC) {
        if (left->typenum!=NPY_FLOAT && left->typenum!=NPY_DOUBLE) return 0;
        if (!left->ndim || !right->ndim) return 0;
        int lv=left->ndim==1, rv=right->ndim==1;
        int lb=lv?0:left->ndim-2, rb=rv?0:right->ndim-2;
        int batch=lb>rb?lb:rb;
        npy_intp n=left->dimensions[left->ndim-1];
        if (n != right->dimensions[right->ndim-(rv?1:2)]) return 0;
        if (batch+!lv+!rv > NPY_MAXDIMS) return 0;
        out->ndim=batch+!lv+!rv;
        for (int axis=0; axis<batch; axis++) {
            npy_intp l=axis<batch-lb?1:left->dimensions[axis-(batch-lb)];
            npy_intp r=axis<batch-rb?1:right->dimensions[axis-(batch-rb)];
            if (l!=r && l!=1 && r!=1) return 0;
            out->dimensions[axis]=l==1?r:l;
        }
        if (!lv) out->dimensions[batch]=left->dimensions[left->ndim-2];
        if (!rv) out->dimensions[batch+!lv]=right->dimensions[right->ndim-1];
        stage->scalar_result=lv&&rv;
    }
    else if (op == NUMPY_CROSSJIT_ADD_OUTER) {
        if (!numpy_crossjit_stage_c_order(left) || !numpy_crossjit_stage_c_order(right)) return 0;
        if (left->ndim+right->ndim > NPY_MAXDIMS) return 0;
        out->ndim=left->ndim+right->ndim;
        if (!out->ndim) return 0;
        memcpy(out->dimensions,left->dimensions,left->ndim*sizeof(npy_intp));
        memcpy(out->dimensions+left->ndim,right->dimensions,right->ndim*sizeof(npy_intp));
    }
    else {
        if (!numpy_crossjit_stage_c_order(left) || !numpy_crossjit_stage_c_order(right)) return 0;
        out->ndim=left->ndim>right->ndim?left->ndim:right->ndim;
        for (int axis=0;axis<out->ndim;axis++) {
            int la=axis-(out->ndim-left->ndim), ra=axis-(out->ndim-right->ndim);
            npy_intp l=la<0?1:left->dimensions[la], r=ra<0?1:right->dimensions[ra];
            if (l!=r && l!=1 && r!=1) return 0;
            out->dimensions[axis]=l==1?r:l;
        }
        stage->scalar_result=out->ndim==0;
    }
    return numpy_crossjit_stage_contiguous(out);
}

static PyJitPrepareDecision
numpy_crossjit_build_stages(const PyJitRegionView *region, struct PyJitExecPlan *plan)
{
    struct PyJitGuardPlan *guards=&plan->guards;
    guards->input_count=PyJit_RegionGetInputCount(region);
    guards->staged=1;
    guards->iteration_class=NUMPY_CROSSJIT_GENERAL_STRIDED;
    plan->stage_count=plan->node_count=PyJit_RegionGetNodeCount(region);
    if (!guards->input_count || guards->input_count>NUMPY_CROSSJIT_MAX_INPUTS ||
            !plan->stage_count || plan->stage_count>NUMPY_CROSSJIT_MAX_NODES) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
    NumpyCrossJitArrayFact values[NUMPY_CROSSJIT_MAX_SLOTS]={{0}};
    for (uint8_t i=0;i<plan->stage_count;i++) {
        const PyJitNodeRef *node=PyJit_RegionGetNode(region,i);
        uint64_t semantic=PyJit_NodeGetSemanticId(node);
        if (!semantic || (semantic&NUMPY_CROSSJIT_INPLACE_FLAG)) return PYJIT_PREPARE_UNSUPPORTED_NODE;
        if (semantic>=NUMPY_CROSSJIT_SUM+1) {
            int alias=semantic-1-NUMPY_CROSSJIT_SUM;
            if (!numpy_crossjit_alias_valid(alias)) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
            guards->alias_mask|=1u<<alias;
        }
        if (numpy_crossjit_is_reduction(semantic-1)) {
            for (uint8_t arg=1;arg<PyJit_NodeGetOperandCount(node);arg++) {
                int slot=numpy_crossjit_value_slot(region,i,PyJit_NodeGetOperand(node,arg));
                if (slot<0 || slot>=guards->input_count) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
                guards->constant_mask|=1u<<slot;
            }
        }
    }
    int dtype=-1;
    for (uint8_t i=0;i<guards->input_count;i++) {
        PyObject *object=PyJit_ValueGetObservedObject(PyJit_RegionGetInput(region,i)->value);
        if (guards->constant_mask&(1u<<i)) {
            int kind=object?numpy_crossjit_constant_fact(object,&guards->inputs[i]):0;
            if (!kind) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
            guards->constant_kinds[i]=kind;
        }
        else if (object && PyArray_CheckExact(object)) {
            if (!numpy_crossjit_array_fact((PyArrayObject *)object,&values[i]) ||
                    !values[i].aligned || !values[i].native_byteorder ||
                    !numpy_crossjit_supported_dtype(values[i].typenum)) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
            if (dtype!=-1 && values[i].typenum!=dtype) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
            dtype=values[i].typenum;
            guards->inputs[i]=values[i];
        }
        else if (object && (PyFloat_CheckExact(object) || PyLong_CheckExact(object) ||
                           Py_TYPE(object)==&PyDoubleArrType_Type || Py_TYPE(object)==&PyFloatArrType_Type || numpy_crossjit_int64_class(object))) {
            guards->scalar_mask|=1u<<i;
            uint64_t cls=1;
            if (numpy_crossjit_int64_class(object)) {guards->numpy_int_scalar_mask|=1u<<i;cls=numpy_crossjit_int64_class(object);}
            if (PyLong_CheckExact(object)) {
                long long value=PyLong_AsLongLong(object);
                if (PyErr_Occurred()) {PyErr_Clear();return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;}
                guards->int_scalar_mask|=1u<<i;
                cls=value>=-INT64_C(1073741823) && value<=INT64_C(1073741823)?3:4;
            }
            if (Py_TYPE(object)==&PyDoubleArrType_Type) {guards->double_scalar_mask|=1u<<i;cls=2;}
            if (Py_TYPE(object)==&PyFloatArrType_Type) {guards->float_scalar_mask|=1u<<i;cls=6;}
            guards->scalar_class_mask|=cls<<(4*i);
        }
        else return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
    }
    /* A floating default cannot establish the promotion/publication contract
     * for integer-only NumPy scalars (minimum/maximum preserve integer dtype).
     * Keep scalar-only integer domains at a boundary until that contract is
     * explicitly supported; an ndarray still establishes the existing
     * homogeneous array/scalar contract. */
    if (dtype<0 && guards->scalar_mask && guards->scalar_mask==
            (guards->numpy_int_scalar_mask|guards->int_scalar_mask))
        return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
    if (dtype<0) dtype=guards->float_scalar_mask?NPY_FLOAT:NPY_DOUBLE;
    if (guards->double_scalar_mask && dtype!=NPY_DOUBLE) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
    int floating=dtype==NPY_FLOAT || dtype==NPY_DOUBLE;
    if (!floating && (guards->scalar_mask&~(guards->int_scalar_mask|guards->numpy_int_scalar_mask))) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
    for (uint8_t i=0;i<guards->input_count;i++) {
        if (guards->numpy_int_scalar_mask&(1u<<i)) {
            uint64_t cls=(guards->scalar_class_mask>>(4*i))&15;
            if (dtype!=NPY_DOUBLE && dtype!=(cls==7?NPY_LONG:NPY_LONGLONG)) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
        }
        if (guards->scalar_mask&(1u<<i)) {
            PyObject *object=PyJit_ValueGetObservedObject(PyJit_RegionGetInput(region,i)->value);
            if ((guards->int_scalar_mask&(1u<<i)) && !numpy_crossjit_safe_int_scalar(object,dtype)) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
            values[i].typenum=dtype;
            guards->inputs[i].typenum=dtype;
        }
    }
    for (uint8_t i=0;i<plan->stage_count;i++) {
        const PyJitNodeRef *node=PyJit_RegionGetNode(region,i);
        uint64_t semantic=PyJit_NodeGetSemanticId(node);
        if (!semantic || (semantic&NUMPY_CROSSJIT_INPLACE_FLAG)) return PYJIT_PREPARE_UNSUPPORTED_NODE;
        NumpyCrossJitOp op=semantic-1;
        int builtin=-1;
        if (op<=NUMPY_CROSSJIT_TRUE_DIVIDE) builtin=op;
        else if (op==NUMPY_CROSSJIT_NEGATIVE) builtin=4;
        else if (op==NUMPY_CROSSJIT_MINIMUM) builtin=5;
        else if (op==NUMPY_CROSSJIT_MAXIMUM) builtin=6;
        else if (op==NUMPY_CROSSJIT_SQRT) builtin=7;
        else if (op==NUMPY_CROSSJIT_EXP) builtin=8;
        else if (op==NUMPY_CROSSJIT_TANH) builtin=9;
        else if (op==NUMPY_CROSSJIT_MATMUL || op==NUMPY_CROSSJIT_DOT || op==NUMPY_CROSSJIT_DOT_PUBLIC) builtin=10;
        else if (op==NUMPY_CROSSJIT_ADD_OUTER || op==NUMPY_CROSSJIT_ADD_REDUCE || op==NUMPY_CROSSJIT_SUM) builtin=0;
        else if (numpy_crossjit_is_reduction(op)) builtin=6;
        if (builtin>=0) {
            if (!numpy_crossjit_builtin_unchanged(builtin)) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
            guards->builtin_ufunc_mask|=1u<<builtin;
        }
        uint8_t kernel=0;
        int integer_minmax=!floating && (op==NUMPY_CROSSJIT_MINIMUM || op==NUMPY_CROSSJIT_MAXIMUM);
        if (integer_minmax) kernel=op==NUMPY_CROSSJIT_MINIMUM?NUMPY_CROSSJIT_KERNEL_MINIMUM:NUMPY_CROSSJIT_KERNEL_MAXIMUM;
        if (op!=NUMPY_CROSSJIT_MATMUL && op!=NUMPY_CROSSJIT_DOT && op!=NUMPY_CROSSJIT_DOT_PUBLIC && op!=NUMPY_CROSSJIT_ADD_OUTER &&
                !numpy_crossjit_is_reduction(op) &&
                op!=NUMPY_CROSSJIT_SQRT && op!=NUMPY_CROSSJIT_EXP && op!=NUMPY_CROSSJIT_TANH &&
                !integer_minmax && !numpy_crossjit_kernel_op(semantic,dtype,&kernel)) return PYJIT_PREPARE_UNSUPPORTED_NODE;
        NumpyCrossJitStage *stage=&plan->stages[i];
        int left=numpy_crossjit_value_slot(region,i,PyJit_NodeGetOperand(node,0));
        const PyJitValueRef *rv=PyJit_NodeGetOperand(node,1);
        int right=rv?numpy_crossjit_value_slot(region,i,rv):left;
        if (left<0 || right<0 || left>=guards->input_count+i || right>=guards->input_count+i) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
        stage->semantic_op=op;stage->left=left;stage->right=right;stage->output=guards->input_count+i;
        if (op==NUMPY_CROSSJIT_SQRT || op==NUMPY_CROSSJIT_EXP || op==NUMPY_CROSSJIT_TANH) {
            if (!floating) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
            const NumpyCrossJitArrayFact *input=&values[left];
            if (input->ndim>1 && !input->c_contiguous) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
            if (!numpy_crossjit_stage_shape(NUMPY_CROSSJIT_ADD,input,input,stage)) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
            PyUFuncObject *ufunc=(PyUFuncObject *)(op==NUMPY_CROSSJIT_SQRT?crossjit_state.sqrt:
                op==NUMPY_CROSSJIT_EXP?crossjit_state.exp:crossjit_state.tanh);
            for (int j=0;j<ufunc->ntypes;j++) {
                if (ufunc->types[2*j]==dtype && ufunc->types[2*j+1]==dtype && ufunc->data[j]==NULL) {
                    stage->unary_entry=ufunc->functions[j];stage->unary_index=j;break;
                }
            }
            if (!stage->unary_entry) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
        }
        else if (numpy_crossjit_is_reduction(op)) {
            if (!floating) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
            if (!numpy_crossjit_stage_reduction(node,&values[left],stage)) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
        }
        else {
            if ((left<guards->input_count && (guards->constant_mask&(1u<<left))) ||
                (right<guards->input_count && (guards->constant_mask&(1u<<right)))) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
            if (!numpy_crossjit_stage_shape(op,&values[left],&values[right],stage)) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
        }
        values[stage->output]=stage->domain;
        plan->nodes[i]=(NumpyCrossJitKernelNode){(uint8_t)op,(uint8_t)left,(uint8_t)right,stage->output};
        if (op!=NUMPY_CROSSJIT_MATMUL && op!=NUMPY_CROSSJIT_DOT && op!=NUMPY_CROSSJIT_DOT_PUBLIC && op!=NUMPY_CROSSJIT_ADD_OUTER &&
            !numpy_crossjit_is_reduction(op) &&
            op!=NUMPY_CROSSJIT_SQRT && op!=NUMPY_CROSSJIT_EXP && op!=NUMPY_CROSSJIT_TANH)
            numpy_crossjit_select_native_loop(kernel,dtype,&stage->loop);
        if (op==NUMPY_CROSSJIT_ADD_OUTER) numpy_crossjit_select_native_loop(NUMPY_CROSSJIT_KERNEL_ADD,dtype,&stage->loop);
        if (integer_minmax && !stage->loop.entry) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
        /* Reuse an already admitted generated entry when this independent
         * elementwise stage has its exact contiguous/scalar domain.  Admission
         * of the enclosing mixed plan still requires a fresh measurement. */
        if (dtype==NPY_DOUBLE && op!=NUMPY_CROSSJIT_MATMUL && op!=NUMPY_CROSSJIT_DOT && op!=NUMPY_CROSSJIT_DOT_PUBLIC && !numpy_crossjit_is_reduction(op) &&
            op!=NUMPY_CROSSJIT_MINIMUM && op!=NUMPY_CROSSJIT_MAXIMUM &&
            op!=NUMPY_CROSSJIT_ADD_OUTER && op!=NUMPY_CROSSJIT_ADD_REDUCE &&
            op!=NUMPY_CROSSJIT_MAXIMUM_REDUCE && op!=NUMPY_CROSSJIT_SQRT &&
            op!=NUMPY_CROSSJIT_EXP && op!=NUMPY_CROSSJIT_TANH) {
            uint16_t scalars=0;
            int compatible=1;
            for (int role=0;role<2;role++) {
                const NumpyCrossJitArrayFact *f=&values[role?right:left];
                if (!f->ndim) scalars|=1u<<role;
                else if (!f->c_contiguous || f->ndim!=stage->domain.ndim ||
                    memcmp(f->dimensions,stage->domain.dimensions,f->ndim*sizeof(npy_intp))) compatible=0;
            }
            if (compatible) {
                NumpyCrossJitKernelNode single={kernel,0,1,2};
                if (numpy_crossjit_generated_registry_lookup(&single,1,2,scalars,2,
                        numpy_crossjit_domain_elements(&stage->domain),&stage->fixed)!=NUMPY_CROSSJIT_POLICY_GENERATED)
                    memset(&stage->fixed,0,sizeof(stage->fixed));
            }
        }
    }
    /* Project validated domain facts and opaque numerical capability bindings.
     * Generator owns dependency order, liveness, release and storage selection.
     * This bridge currently authorizes overlap only at the existing final
     * ndarray handoff, preserving the measured backend contract and key. */
    CjExecutionDescription description={0};
    description.input_count=guards->input_count;
    description.value_count=guards->input_count+plan->stage_count;
    description.stage_count=description.capability_count=plan->stage_count;
    description.output_value=description.value_count-1;
    for (uint8_t slot=0;slot<description.value_count;slot++) {
        description.values[slot].kind=slot<guards->input_count?CJ_INPUT_VALUE:
            slot==description.output_value?CJ_OUTPUT_VALUE:CJ_TEMPORARY_VALUE;
        /* No reuse is authorized until exact domain compatibility below. */
        description.values[slot].storage_class=slot;
    }
    for (uint8_t i=0;i<plan->stage_count;i++) {
        NumpyCrossJitStage *stage=&plan->stages[i];
        CjStageDescriptor *d=&description.stages[i];
        d->kind=(stage->fixed.entry || (!stage->loop.entry && !stage->unary_entry &&
            !numpy_crossjit_is_reduction(stage->semantic_op) &&
            stage->semantic_op!=NUMPY_CROSSJIT_MATMUL &&
            stage->semantic_op!=NUMPY_CROSSJIT_DOT &&
            stage->semantic_op!=NUMPY_CROSSJIT_DOT_PUBLIC))?CJ_GENERATED_STAGE:CJ_BACKEND_STAGE;
        d->capability=i;
        d->after=i?(UINT32_C(1)<<(i-1)):0; /* preserve original FPE barriers */
        d->input_count=(stage->unary_entry ||
            numpy_crossjit_is_reduction(stage->semantic_op) ||
            stage->semantic_op==NUMPY_CROSSJIT_NEGATIVE)?1:2;
        d->inputs[0]=stage->left;d->inputs[1]=stage->right;d->output=stage->output;
        description.capabilities[i]=(CjBackendCapability){i,d->input_count,0};
    }
    if (plan->stage_count>=2) {
        uint8_t i=plan->stage_count-1;
        NumpyCrossJitStage *last=&plan->stages[i];
        NumpyCrossJitStage *previous=&plan->stages[i-1];
        /* Typed loops permit private exact overlap. Domain semantics remain
         * here; last-use and operand eligibility are decided by Generator. */
        int compatible=(last->unary_entry || last->loop.entry) && !last->scalar_result &&
            previous->domain.typenum==last->domain.typenum &&
            previous->domain.ndim==last->domain.ndim &&
            memcmp(previous->domain.dimensions,last->domain.dimensions,
                last->domain.ndim*sizeof(npy_intp))==0 &&
            memcmp(previous->domain.strides,last->domain.strides,
                last->domain.ndim*sizeof(npy_intp))==0;
        if (compatible) {
            description.stages[i].reuse_candidates=UINT32_C(1)<<previous->output;
            description.capabilities[i].overlap_inputs=3;
            description.values[last->output].storage_class=previous->output;
        }
    }
    if (!cj_synthesize_plan(&description,&plan->execution))
        return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
    plan->output_slot=plan->execution.output_value;
    guards->domain=values[plan->output_slot];
    guards->guard_scalar_classes=1;
    plan->reuse_final_loop=plan->execution.reuse_source[plan->output_slot]!=CJ_NONE;
    if (plan->reuse_final_loop) {
        NumpyCrossJitStage *last=&plan->stages[plan->stage_count-1];
        memset(&last->fixed,0,sizeof(last->fixed));
        plan->execution.stages[plan->stage_count-1].kind=CJ_BACKEND_STAGE;
    }
    if (!numpy_crossjit_native_key(plan)) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
    plan->native_key[0]='S'; /* signature opcodes are semantic, not elementwise kernel opcodes */
    size_t key_used=strlen(plan->native_key);
    if (plan->reuse_final_loop && !numpy_crossjit_key_append(plan->native_key,
            sizeof(plan->native_key),&key_used,"|u:%u:%u:2",plan->stage_count-1,plan->stage_count-2))
        return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
    for (uint8_t i=0;i<plan->stage_count;i++) {
        NumpyCrossJitStage *stage=&plan->stages[i];
        if (!numpy_crossjit_key_append(plan->native_key,sizeof(plan->native_key),&key_used,
                "|v:%u:%d:%u:%u:",i,stage->domain.typenum,stage->scalar_result,stage->domain.ndim))
            return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
        for (uint8_t axis=0;axis<stage->domain.ndim;axis++) {
            if (!numpy_crossjit_key_append(plan->native_key,sizeof(plan->native_key),&key_used,
                    "%lld,%lld;",(long long)stage->domain.dimensions[axis],(long long)stage->domain.strides[axis]))
                return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
        }
        if (plan->stages[i].loop.entry && !plan->stages[i].fixed.entry && !numpy_crossjit_key_append(plan->native_key,
                sizeof(plan->native_key),&key_used,"|l:%u:1",i)) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
        if (plan->stages[i].fixed.entry && !numpy_crossjit_key_append(plan->native_key,
                sizeof(plan->native_key),&key_used,"|g:%u:1",i)) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
    }
    if (!numpy_crossjit_call_key(region,plan)) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
    if (!numpy_crossjit_native_observe(plan->native_key,"prepare")) return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
    if (!numpy_crossjit_audit_execution(plan->native_key,&description,&plan->execution))
        return PYJIT_PREPARE_UNSUPPORTED_CONTRACT;
    const char *calibration=Py_GETENV("NPY_CROSSJIT_NATIVE_CALIBRATION");
    if (!numpy_crossjit_native_measured(plan->native_key) &&
            !(calibration && strcmp(calibration,"1")==0)) return PYJIT_PREPARE_UNPROFITABLE;
    plan->native_backend=2;
    crossjit_stats.native_plans_built++;
    return PYJIT_PREPARE_ACCEPT;
}

static void
numpy_crossjit_stage_matmul(const NumpyCrossJitArrayFact *left,
                            const NumpyCrossJitArrayFact *right,
                            const NumpyCrossJitArrayFact *out,
                            char *lp, char *rp, char *op)
{
    int lv=left->ndim==1,rv=right->ndim==1;
    int lb=lv?0:left->ndim-2,rb=rv?0:right->ndim-2;
    int batch=lb>rb?lb:rb;
    npy_intp count=1;
    for (int axis=0;axis<batch;axis++) count*=out->dimensions[axis];
    npy_intp dims[4]={1,lv?1:left->dimensions[left->ndim-2],left->dimensions[left->ndim-1],rv?1:right->dimensions[right->ndim-1]};
    npy_intp width=out->typenum==NPY_FLOAT?sizeof(float):sizeof(double);
    npy_intp steps[9]={0,0,0,lv?0:left->strides[left->ndim-2],left->strides[left->ndim-1],
        right->strides[right->ndim-(rv?1:2)],rv?0:right->strides[right->ndim-1],rv?width:dims[3]*width,width};
    for (npy_intp index=0;index<count;index++) {
        npy_intp remainder=index,lo=0,ro=0,oo=0;
        for (int axis=batch-1;axis>=0;axis--) {
            npy_intp coordinate=remainder%out->dimensions[axis];remainder/=out->dimensions[axis];
            int la=axis-(batch-lb),ra=axis-(batch-rb);
            if (la>=0 && left->dimensions[la]!=1) lo+=coordinate*left->strides[la];
            if (ra>=0 && right->dimensions[ra]!=1) ro+=coordinate*right->strides[ra];
            oo+=coordinate*out->strides[axis];
        }
        char *args[3]={lp+lo,rp+ro,op+oo};
        native_packing_active=crossjit_stats.diagnostics_enabled;
        if (out->typenum==NPY_DOUBLE) DOUBLE_matmul(args,dims,steps,NULL);
        else FLOAT_matmul(args,dims,steps,NULL);
        native_packing_active=0;
        if (PyErr_Occurred()) return;
    }
}

typedef struct {
    struct PyJitExecPlan *plan;
    char *data[NUMPY_CROSSJIT_MAX_SLOTS];
    union {double d;float f;npy_int64 i64;npy_int32 i32;}
        scalar_results[NUMPY_CROSSJIT_MAX_SLOTS];
    uint8_t heap_owned[NUMPY_CROSSJIT_MAX_SLOTS];
    const NumpyCrossJitArrayFact *values[NUMPY_CROSSJIT_MAX_SLOTS];
    PyArrayObject *output;
    char barrier;
} NumpyCrossJitStageContext;

static CjExecStatus
numpy_crossjit_invoke_stage(void *opaque, uint8_t i, uint16_t binding)
{
    NumpyCrossJitStageContext *context=opaque;
    struct PyJitExecPlan *plan=context->plan;
    char **data=context->data;
    uint8_t *heap_owned=context->heap_owned;
    const NumpyCrossJitArrayFact **values=context->values;
    if (binding!=i) return CJ_EXEC_MISS;
    NumpyCrossJitStage *stage=&plan->stages[i];
    NumpyCrossJitArrayFact *fact=&stage->domain;
    uint64_t count=numpy_crossjit_domain_elements(fact);
    size_t width=numpy_crossjit_dtype_width(fact->typenum);
    if (count>NPY_MAX_INTP/width) goto done;
    if (plan->execution.reuse_source[stage->output]!=CJ_NONE) {
        data[stage->output]=data[plan->execution.reuse_source[stage->output]];
    }
    else if (plan->execution.output_storage[stage->output] && !stage->scalar_result) {
        context->output=(PyArrayObject *)PyArray_SimpleNew(fact->ndim,fact->dimensions,fact->typenum);
        if (!context->output) goto error;
        data[stage->output]=PyArray_BYTES(context->output);
        crossjit_stats.output_allocations++;
    }
    else if (stage->scalar_result) {
        if (fact->typenum==NPY_FLOAT) data[stage->output]=(char *)&context->scalar_results[stage->output].f;
        else if (fact->typenum==NPY_DOUBLE) data[stage->output]=(char *)&context->scalar_results[stage->output].d;
        else if (fact->typenum==NPY_INT) data[stage->output]=(char *)&context->scalar_results[stage->output].i32;
        else data[stage->output]=(char *)&context->scalar_results[stage->output].i64;
    }
    else {
        data[stage->output]=PyMem_Malloc(count?count*width:width);
        if (!data[stage->output]) {PyErr_NoMemory();goto error;}
        heap_owned[stage->output]=1;
        crossjit_stats.internal_allocations++;
        crossjit_stats.internal_buffer_bytes+=count?count*width:width;
    }
    values[stage->output]=fact;
    if (numpy_crossjit_is_reduction(stage->semantic_op)) {
        const NumpyCrossJitArrayFact *in=values[stage->left];
        PyArrayObject *iv=(PyArrayObject *)PyArray_New(&PyArray_Type,in->ndim,in->dimensions,
            in->typenum,in->strides,data[stage->left],0,NPY_ARRAY_ALIGNED,NULL);
        if (!iv) goto error;
        crossjit_stats.internal_object_allocations++;
        PyArrayObject *ov=(PyArrayObject *)PyArray_New(&PyArray_Type,fact->ndim,fact->dimensions,
            fact->typenum,fact->strides,data[stage->output],0,NPY_ARRAY_ALIGNED|NPY_ARRAY_WRITEABLE,NULL);
        if (!ov) {Py_DECREF(iv);goto error;}
        crossjit_stats.internal_object_allocations++;
        PyUFuncObject *ufunc=(PyUFuncObject *)((stage->semantic_op==NUMPY_CROSSJIT_ADD_REDUCE || stage->semantic_op==NUMPY_CROSSJIT_SUM)?crossjit_state.add:crossjit_state.maximum);
        PyArrayObject *reduced=numpy_crossjit_reduce_no_callbacks(ufunc,iv,ov,stage->axis_count,stage->axes,stage->keepdims);
        Py_DECREF(iv);Py_DECREF(ov);
        if (!reduced) goto error;
        Py_DECREF(reduced);
    }
    else if (stage->unary_entry) {
        PyUFuncObject *ufunc=(PyUFuncObject *)(stage->semantic_op==NUMPY_CROSSJIT_SQRT?crossjit_state.sqrt:
            stage->semantic_op==NUMPY_CROSSJIT_EXP?crossjit_state.exp:crossjit_state.tanh);
        if (ufunc->functions[stage->unary_index]!=stage->unary_entry || ufunc->data[stage->unary_index]!=NULL) goto done;
        const NumpyCrossJitArrayFact *in=values[stage->left];
        char *args[2]={data[stage->left],data[stage->output]};
        npy_intp dims[1]={count},steps[2]={in->ndim==0?0:in->ndim==1?in->strides[0]:(npy_intp)width,(npy_intp)width};
        stage->unary_entry(args,dims,steps,NULL);
    }
    else if (stage->semantic_op==NUMPY_CROSSJIT_MATMUL || stage->semantic_op==NUMPY_CROSSJIT_DOT || stage->semantic_op==NUMPY_CROSSJIT_DOT_PUBLIC) {
        numpy_crossjit_stage_matmul(values[stage->left],values[stage->right],fact,
            data[stage->left],data[stage->right],data[stage->output]);
        if (PyErr_Occurred()) goto error;
    }
    else {
        NumpyCrossJitKernelNode node={0,0,1,2};
        if (stage->semantic_op==NUMPY_CROSSJIT_ADD_OUTER) node.op=NUMPY_CROSSJIT_KERNEL_ADD;
        else if (stage->semantic_op==NUMPY_CROSSJIT_MINIMUM) node.op=NUMPY_CROSSJIT_KERNEL_MINIMUM;
        else if (stage->semantic_op==NUMPY_CROSSJIT_MAXIMUM) node.op=NUMPY_CROSSJIT_KERNEL_MAXIMUM;
        else if (!numpy_crossjit_kernel_op(stage->semantic_op+1,fact->typenum,&node.op)) goto done;
        const void *args[2]={data[stage->left],data[stage->right]};
        npy_intp strides[NUMPY_CROSSJIT_MAX_INPUTS][NPY_MAXDIMS]={{0}};
        for (int role=0;role<2;role++) {
            const NumpyCrossJitArrayFact *in=values[role?stage->right:stage->left];
            if (stage->semantic_op==NUMPY_CROSSJIT_ADD_OUTER) {
                int start=role?values[stage->left]->ndim:0;
                memcpy(strides[role]+start,in->strides,in->ndim*sizeof(npy_intp));
            }
            else {
                int lead=fact->ndim-in->ndim;
                for (int axis=0;axis<in->ndim;axis++) {
                    strides[role][lead+axis]=in->dimensions[axis]==1?0:in->strides[axis];
                }
            }
        }
        if (stage->fixed.entry) {
            const void *packed[NUMPY_CROSSJIT_MAX_INPUTS];
            for (uint8_t j=0;j<stage->fixed.input_count;j++) packed[j]=args[stage->fixed.input_slots[j]];
            stage->fixed.entry(packed,(double *)data[stage->output],count);
        }
        else if (stage->loop.entry) {
            if (!numpy_crossjit_native_loop(&stage->loop,&node,args,strides,fact->ndim,fact->dimensions,
                    data[stage->output],count,width)) goto done;
        }
        else if (!numpy_crossjit_kernel_strided(&node,1,2,0,2,fact->typenum,args,
                strides,fact->ndim,fact->dimensions,data[stage->output],count)) goto done;
    }
    /* FPE is checked per original operation; original execution reports once. */
    if (npy_get_floatstatus_barrier(&context->barrier)) goto done;
    return CJ_EXEC_OK;
error:
    return CJ_EXEC_ERROR;
done:
    return CJ_EXEC_MISS;
}

static void
numpy_crossjit_release_stage_values(void *opaque, uint32_t mask)
{
    NumpyCrossJitStageContext *context=opaque;
    while (mask) {
        uint8_t slot=(uint8_t)__builtin_ctz(mask);
        mask &= mask-1;
        if (context->heap_owned[slot]) PyMem_Free(context->data[slot]);
        context->heap_owned[slot]=0;context->data[slot]=NULL;
    }
}

static PyJitResult
numpy_crossjit_execute_stages(struct PyJitExecPlan *plan, PyObject *const *inputs,
                              PyObject **result, uint64_t started)
{
    NumpyCrossJitStageContext context={0};
    context.plan=plan;
    char **data=context.data;
    uint8_t *heap_owned=context.heap_owned;
    double scalars[NUMPY_CROSSJIT_MAX_INPUTS];
    float float_scalars[NUMPY_CROSSJIT_MAX_INPUTS];
    npy_int64 int_scalars[NUMPY_CROSSJIT_MAX_INPUTS];
    npy_int32 int32_scalars[NUMPY_CROSSJIT_MAX_INPUTS];
    const NumpyCrossJitArrayFact **values=context.values;
    PyJitResult status=PYJIT_RESULT_MISS;
    npy_clear_floatstatus_barrier(&context.barrier);
    for (uint8_t i=0;i<plan->guards.input_count;i++) {
        values[i]=&plan->guards.inputs[i];
        if (plan->guards.constant_mask&(1u<<i)) continue;
        if (plan->guards.scalar_mask&(1u<<i)) {
            if (plan->guards.domain.typenum!=NPY_FLOAT && plan->guards.domain.typenum!=NPY_DOUBLE) {
                npy_int64 v=plan->guards.numpy_int_scalar_mask&(1u<<i)?numpy_crossjit_int64_value(inputs[i]):PyLong_AsLongLong(inputs[i]);
                if (PyErr_Occurred()) goto error;
                if (values[i]->typenum==NPY_INT) {int32_scalars[i]=v;data[i]=(char *)&int32_scalars[i];}
                else {int_scalars[i]=v;data[i]=(char *)&int_scalars[i];}
                continue;
            }
            double v=plan->guards.numpy_int_scalar_mask&(1u<<i)?(double)numpy_crossjit_int64_value(inputs[i]):plan->guards.int_scalar_mask&(1u<<i)?PyLong_AsDouble(inputs[i]):
                (plan->guards.float_scalar_mask&(1u<<i)?PyArrayScalar_VAL(inputs[i],Float):PyFloat_AS_DOUBLE(inputs[i]));
            if (values[i]->typenum==NPY_FLOAT) {float_scalars[i]=(float)v;data[i]=(char *)&float_scalars[i];}
            else {scalars[i]=v;data[i]=(char *)&scalars[i];}
        }
        else data[i]=PyArray_BYTES((PyArrayObject *)inputs[i]);
    }
    const CjExecutionBridge bridge={numpy_crossjit_invoke_stage,
        numpy_crossjit_invoke_stage,numpy_crossjit_release_stage_values};
    CjExecStatus executed=cj_execute_plan(&plan->execution,&bridge,&context);
    if (executed==CJ_EXEC_ERROR) goto error;
    if (executed!=CJ_EXEC_OK) goto done;
    if (plan->stages[plan->stage_count-1].scalar_result) {
        PyArray_Descr *descr=PyArray_DescrFromType(plan->guards.domain.typenum);
        if (!descr) goto error;
        *result=PyArray_Scalar(data[plan->output_slot],descr,NULL);
        Py_DECREF(descr);
        if (!*result) goto error;
        crossjit_stats.output_allocations++;
    }
    else {*result=(PyObject *)context.output;context.output=NULL;}
    int mixed=0;
    for (uint8_t i=0;i<plan->stage_count;i++) mixed|=plan->stages[i].fixed.entry!=NULL;
    if (mixed) crossjit_stats.mixed_exec_hit++;
    else crossjit_stats.native_exec_hit++;
    if (!numpy_crossjit_native_observe(plan->native_key,"execute")) {
        Py_CLEAR(*result);goto error;
    }
    status=PYJIT_RESULT_HIT;
    goto done;
error:
    status=PYJIT_RESULT_ERROR;
done:
    for (uint8_t i=0;i<plan->stage_count;i++) {
        NumpyCrossJitStage *stage=&plan->stages[i];
        if (heap_owned[stage->output]) PyMem_Free(data[stage->output]);
    }
    Py_XDECREF(context.output);
    return numpy_crossjit_record_exec(status,started,0,0);
}
