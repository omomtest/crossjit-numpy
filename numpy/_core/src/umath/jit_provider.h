#ifndef NUMPY_CORE_SRC_UMATH_JIT_PROVIDER_H
#define NUMPY_CORE_SRC_UMATH_JIT_PROVIDER_H

#include <Python.h>

/* Installs the semantic provider after umath has created its ufunc objects. */
int numpy_crossjit_init(PyObject *module);
void numpy_crossjit_native_record_packing(size_t bytes);

#endif
