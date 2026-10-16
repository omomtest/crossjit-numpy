/* Opt-in evaluation artifacts. No domain facts are interpreted here. */
static PyObject *
numpy_crossjit_audit_u8(const uint8_t *values, size_t count)
{
    PyObject *items=PyTuple_New((Py_ssize_t)count);
    if (!items) return NULL;
    for (size_t i=0;i<count;i++) {
        PyObject *value=PyLong_FromUnsignedLong(values[i]);
        if (!value) {Py_DECREF(items);return NULL;}
        PyTuple_SET_ITEM(items,(Py_ssize_t)i,value);
    }
    return items;
}

static int
numpy_crossjit_audit_execution(const char *key,
    const CjExecutionDescription *d, const CjExecutionPlan *p)
{
    if (!crossjit_stats.diagnostics_enabled ||
        Py_GETENV("NPY_CROSSJIT_EVALUATION_AUDIT")==NULL) return 1;
    PyObject *row=PyDict_GetItemString(native_observations,key);
    if (!row) return 0;
    PyObject *values=PyTuple_New(d->value_count);
    PyObject *stages=PyTuple_New(d->stage_count);
    PyObject *caps=PyTuple_New(d->capability_count);
    PyObject *release=PyTuple_New(p->stage_count);
    PyObject *description=NULL,*plan=NULL;
    if (!values || !stages || !caps || !release) goto error;
    for (uint8_t i=0;i<d->value_count;i++) {
        PyObject *v=Py_BuildValue("{sI,sI}","kind",d->values[i].kind,
            "storage_class",d->values[i].storage_class);
        if (!v) goto error;
        PyTuple_SET_ITEM(values,i,v);
    }
    for (uint8_t i=0;i<d->stage_count;i++) {
        const CjStageDescriptor *s=&d->stages[i];
        PyObject *inputs=numpy_crossjit_audit_u8(s->inputs,s->input_count);
        if (!inputs) goto error;
        PyObject *v=Py_BuildValue("{sI,sI,sO,sI,sI,sI}",
            "kind",s->kind,"capability",s->capability,"inputs",inputs,
            "output",s->output,"after",s->after,"reuse_candidates",s->reuse_candidates);
        Py_DECREF(inputs);
        if (!v) goto error;
        PyTuple_SET_ITEM(stages,i,v);
        v=PyLong_FromUnsignedLong(p->release_mask[i]);
        if (!v) goto error;
        PyTuple_SET_ITEM(release,i,v);
    }
    for (uint8_t i=0;i<d->capability_count;i++) {
        PyObject *v=Py_BuildValue("{sI,sI,sI}","binding",d->capabilities[i].binding,
            "input_count",d->capabilities[i].input_count,
            "overlap_inputs",d->capabilities[i].overlap_inputs);
        if (!v) goto error;
        PyTuple_SET_ITEM(caps,i,v);
    }
    description=Py_BuildValue("{sI,sI,sI,sI,sO,sO,sO}",
        "input_count",d->input_count,"value_count",d->value_count,
        "stage_count",d->stage_count,"output_value",d->output_value,
        "values",values,"stages",stages,"capabilities",caps);
    plan=PyDict_New();
    if (!description || !plan) goto error;
#define CJ_AUDIT_ARRAY(field,count) do { \
    PyObject *v=numpy_crossjit_audit_u8(p->field,count); \
    if (!v) goto error; \
    int status=PyDict_SetItemString(plan,#field,v); Py_DECREF(v); \
    if (status<0) goto error; \
} while(0)
    CJ_AUDIT_ARRAY(order,p->stage_count);
    CJ_AUDIT_ARRAY(producer,d->value_count);
    CJ_AUDIT_ARRAY(last_use,d->value_count);
    CJ_AUDIT_ARRAY(storage_slot,d->value_count);
    CJ_AUDIT_ARRAY(reuse_source,d->value_count);
    CJ_AUDIT_ARRAY(output_storage,d->value_count);
#undef CJ_AUDIT_ARRAY
    if (PyDict_SetItemString(plan,"release_mask",release)<0 ||
        PyDict_SetItemString(row,"execution_description",description)<0 ||
        PyDict_SetItemString(row,"synthesized_plan",plan)<0) goto error;
    Py_DECREF(values);Py_DECREF(stages);Py_DECREF(caps);Py_DECREF(release);
    Py_DECREF(description);Py_DECREF(plan);
    return 1;
error:
    Py_XDECREF(values);Py_XDECREF(stages);Py_XDECREF(caps);Py_XDECREF(release);
    Py_XDECREF(description);Py_XDECREF(plan);
    return 0;
}
