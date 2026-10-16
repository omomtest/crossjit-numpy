#ifndef _NPY_UMATH_UFUNC_OBJECT_H_
#define _NPY_UMATH_UFUNC_OBJECT_H_

#include <numpy/ufuncobject.h>

#ifdef __cplusplus
extern "C" {
#endif

NPY_NO_EXPORT const char*
ufunc_get_name_cstr(PyUFuncObject *ufunc);

NPY_NO_EXPORT PyObject *
PyUFunc_GetDefaultIdentity(PyUFuncObject *ufunc, npy_bool *reorderable);

NPY_NO_EXPORT PyArrayObject *
numpy_crossjit_reduce_no_callbacks(PyUFuncObject *ufunc, PyArrayObject *arr,
    PyArrayObject *out, int naxes, int *axes, int keepdims);

#ifdef __cplusplus
}
#endif

#endif
