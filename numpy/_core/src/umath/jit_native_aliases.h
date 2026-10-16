/* Installed only by the bundled NumPy modules during their initialization.
 * Execution reads exact objects and dictionaries; it never calls Python. */
typedef struct {
    PyObject *callable, *implementation, *relevant;
    PyObject *code, *defaults, *kwdefaults;
    PyObject *relevant_code, *relevant_defaults, *relevant_kwdefaults;
    PyObject *wrap, *wrap_code, *wrap_defaults, *wrap_kwdefaults;
    PyObject *numpy_module, *multiarray_module, *isinstance;
} NumpyCrossJitAlias;

static NumpyCrossJitAlias numpy_crossjit_aliases[4];

static int
numpy_crossjit_is_reduction(int op)
{
    return op==NUMPY_CROSSJIT_ADD_REDUCE || op==NUMPY_CROSSJIT_MAXIMUM_REDUCE ||
        op==NUMPY_CROSSJIT_SUM || op==NUMPY_CROSSJIT_MAX || op==NUMPY_CROSSJIT_AMAX;
}

static int
numpy_crossjit_alias_index(PyObject *callable)
{
    for (int i=0;i<4;i++) if (callable==numpy_crossjit_aliases[i].callable) return i;
    return -1;
}

static int
numpy_crossjit_function_unchanged(PyObject *object,PyObject *code,
    PyObject *defaults,PyObject *kwdefaults)
{
    if (!object || !PyFunction_Check(object)) return 0;
    PyFunctionObject *fn=(PyFunctionObject *)object;
    return fn->func_code==code && fn->func_defaults==defaults &&
        fn->func_kwdefaults==kwdefaults && kwdefaults==NULL && fn->func_closure==NULL;
}

static int
numpy_crossjit_alias_valid(int index)
{
    if (index<0 || index>=4) return 0;
    NumpyCrossJitAlias *a=&numpy_crossjit_aliases[index];
    PyObject *relevant,*impl;
    if (!a->callable || !numpy_crossjit_dispatcher_parts(a->callable,&relevant,&impl) ||
        relevant!=a->relevant || impl!=a->implementation ||
        !numpy_crossjit_function_unchanged(relevant,a->relevant_code,a->relevant_defaults,a->relevant_kwdefaults)) return 0;
    if (index==NUMPY_CROSSJIT_DOT_PUBLIC-NUMPY_CROSSJIT_SUM) return impl==crossjit_state.dot;
    if (!numpy_crossjit_function_unchanged(impl,a->code,a->defaults,a->kwdefaults) ||
        !numpy_crossjit_function_unchanged(a->wrap,a->wrap_code,a->wrap_defaults,a->wrap_kwdefaults)) return 0;
    PyFunctionObject *fn=(PyFunctionObject *)impl,*wrap=(PyFunctionObject *)a->wrap;
    PyObject *globals=fn->func_globals;
    if (!PyModule_CheckExact(a->numpy_module) || !PyModule_CheckExact(a->multiarray_module) ||
        PyDict_GetItemString(globals,"np")!=a->numpy_module ||
        PyDict_GetItemString(globals,"mu")!=a->multiarray_module ||
        PyDict_GetItemString(globals,"_wrapreduction")!=a->wrap ||
        wrap->func_globals!=globals || !PyDict_CheckExact(wrap->func_builtins)) return 0;
    PyObject *type=PyDict_GetItemString(globals,"type");
    if (type && type!=(PyObject *)&PyType_Type) return 0;
    if (!type && PyDict_GetItemString(wrap->func_builtins,"type")!=(PyObject *)&PyType_Type) return 0;
    if (index==0) {
        if (PyDict_GetItemString(globals,"_gentype")!=(PyObject *)&PyGen_Type ||
            fn->func_builtins!=wrap->func_builtins) return 0;
        PyObject *instance=PyDict_GetItemString(globals,"isinstance");
        if (!instance) instance=PyDict_GetItemString(fn->func_builtins,"isinstance");
        if (instance!=a->isinstance) return 0;
    }
    PyObject *npdict=PyModule_GetDict(a->numpy_module),*mudict=PyModule_GetDict(a->multiarray_module);
    const char *op=index==0?"add":"maximum";
    PyObject *ufunc=index==0?crossjit_state.add:crossjit_state.maximum;
    return PyDict_GetItemString(npdict,op)==ufunc &&
        PyDict_GetItemString(npdict,"_NoValue")==npy_static_pydata._NoValue &&
        PyDict_GetItemString(mudict,"ndarray")== (PyObject *)&PyArray_Type;
}

static void
numpy_crossjit_capture_function(PyObject *object,PyObject **code,
    PyObject **defaults,PyObject **kwdefaults)
{
    PyFunctionObject *fn=(PyFunctionObject *)object;
    *code=Py_NewRef(fn->func_code);
    *defaults=Py_XNewRef(fn->func_defaults);
    *kwdefaults=Py_XNewRef(fn->func_kwdefaults);
}

static PyObject *
numpy_crossjit_register_intrinsic(PyObject *module,PyObject *args)
{
    PyObject *alias;
    int op;
    if (!PyArg_ParseTuple(args,"Oi:_crossjit_register_intrinsic",&alias,&op)) return NULL;
    int index=op-NUMPY_CROSSJIT_SUM;
    if (index<0 || index>=4) {PyErr_SetString(PyExc_ValueError,"unsupported intrinsic");return NULL;}
    PyObject *globals=PyEval_GetGlobals();
    PyObject *name=globals?PyDict_GetItemString(globals,"__name__"):NULL;
    const char *expected=index==2?"numpy._core.multiarray":"numpy._core.fromnumeric";
    if (!name || !PyUnicode_CheckExact(name) || PyUnicode_CompareWithASCIIString(name,expected)!=0) {
        PyErr_SetString(PyExc_ValueError,"intrinsics are initialized by the bundled NumPy module");return NULL;
    }
    PyObject *relevant,*impl;
    if (!numpy_crossjit_dispatcher_parts(alias,&relevant,&impl) ||
        !relevant || !PyFunction_Check(relevant) ||
        ((PyFunctionObject *)relevant)->func_closure!=NULL ||
        PyTuple_GET_SIZE(((PyCodeObject *)((PyFunctionObject *)relevant)->func_code)->co_names)!=0) {
        PyErr_SetString(PyExc_ValueError,"intrinsic dispatcher is not a pure argument projection");return NULL;
    }
    if (numpy_crossjit_aliases[index].callable) Py_RETURN_NONE;
    PyObject *wrap=NULL,*np=NULL,*mu=NULL;
    if (index==2) {
        if (impl!=crossjit_state.dot) {PyErr_SetString(PyExc_ValueError,"unexpected dot implementation");return NULL;}
    }
    else {
        if (!PyFunction_Check(impl) || ((PyFunctionObject *)impl)->func_globals!=globals ||
            ((PyFunctionObject *)impl)->func_closure!=NULL) {
            PyErr_SetString(PyExc_ValueError,"unexpected reduction implementation");return NULL;
        }
        wrap=PyDict_GetItemString(globals,"_wrapreduction");
        np=PyDict_GetItemString(globals,"np");mu=PyDict_GetItemString(globals,"mu");
        if (!wrap || !PyFunction_Check(wrap) || !np || !mu) {
            PyErr_SetString(PyExc_ValueError,"reduction dependencies unavailable");return NULL;
        }
    }
    if (crossjit_provider.key_count>=sizeof(crossjit_keys)/sizeof(crossjit_keys[0])) {
        PyErr_SetString(PyExc_ValueError,"intrinsic registration capacity exceeded");return NULL;
    }
    NumpyCrossJitAlias *a=&numpy_crossjit_aliases[index];
    a->callable=Py_NewRef(alias);a->implementation=Py_NewRef(impl);a->relevant=Py_NewRef(relevant);
    numpy_crossjit_capture_function(relevant,&a->relevant_code,&a->relevant_defaults,&a->relevant_kwdefaults);
    if (wrap) {
        numpy_crossjit_capture_function(impl,&a->code,&a->defaults,&a->kwdefaults);
        a->wrap=Py_NewRef(wrap);a->numpy_module=Py_NewRef(np);a->multiarray_module=Py_NewRef(mu);
        numpy_crossjit_capture_function(wrap,&a->wrap_code,&a->wrap_defaults,&a->wrap_kwdefaults);
    }
    if (index==0) a->isinstance=Py_XNewRef(PyDict_GetItemString(((PyFunctionObject *)impl)->func_builtins,"isinstance"));
    crossjit_keys[crossjit_provider.key_count++]=(PyJitProviderKey){PYJIT_KEY_CALLABLE,0,NULL,NULL,alias};
    if (PyJit_RegisterProvider(&crossjit_provider)<0) {
        crossjit_provider.key_count--;
#define CLEAR_ALIAS(field) Py_CLEAR(a->field)
        CLEAR_ALIAS(callable);CLEAR_ALIAS(implementation);CLEAR_ALIAS(relevant);
        CLEAR_ALIAS(code);CLEAR_ALIAS(defaults);CLEAR_ALIAS(kwdefaults);
        CLEAR_ALIAS(relevant_code);CLEAR_ALIAS(relevant_defaults);CLEAR_ALIAS(relevant_kwdefaults);
        CLEAR_ALIAS(wrap);CLEAR_ALIAS(wrap_code);CLEAR_ALIAS(wrap_defaults);CLEAR_ALIAS(wrap_kwdefaults);
        CLEAR_ALIAS(numpy_module);CLEAR_ALIAS(multiarray_module);CLEAR_ALIAS(isinstance);
#undef CLEAR_ALIAS
        return NULL;
    }
    Py_RETURN_NONE;
}
