# PF3 Provider-Owned IterationDescriptor

PF3 introduces a versioned iteration contract without changing CPython or the
frozen GEN-AUTO structural signature.  The generic Generator model records:

```text
iteration rank class
outer count and outer shape class
inner extent
logical input slot
operand access kind
inner stride and optional outer stride
broadcast/scalar relation
fresh contiguous output descriptor
```

The initial access kinds are `CONTIGUOUS`, `SCALAR`, `ZERO_STRIDE`, and
`FIXED_STRIDE`.  NumPy owns the concrete byte-stride projection in its
Provider contract.  Truly general stride patterns remain an unsupported
descriptor case, not an implicitly admitted generated mode.

`CONTIGUOUS_SAME_SHAPE` is represented by a flat descriptor with one outer
iteration and a runtime element-count inner extent.  Legacy schema-v1
candidates omit this implied descriptor from canonical serialization.  This
deliberately preserves historical candidate hashes, the frozen DecisionDB,
and generated registry mapping.  New explicit descriptors are serialized and
hashed for later PROFILED modes.

## Compatibility gate

```text
Generator tests:                     66 passed
NumPy generation/build-tool tests:   14 passed
installed NumPy CrossJIT tests:      28 passed
fresh NumPy build:                   481/481
```

Fresh-build GEN-AUTO artifacts remain byte-for-byte identical:

```text
generated C:
5a5bc62c9aed9944408ce29761f89ebd81151f64879549088371b09419dd2857
generated H:
d9703961dff4dea5f3e409120a07a63949cc2dd2362b1e0c0433e849a06f09f4
registry metadata:
3c1feaecac1be71413588d85a6ce8c43b7875b9f64d37d3e4ed266fb4d7ca958
generated object:
de746d55f945686aadde5bb2ed367e0aac7d6ba0ca5dd751c87c92a89b33487f
```

The generated object hash matches the pre-PF3 build object, establishing
identical generated code and assembly for the contiguous configuration.

The complete historical 104-cell float64 `CONTIGUOUS_SAME_SHAPE` primary
matrix was repeated with five independent processes and seven samples:

```text
semantic correctness checks:             104/104
production-policy Provider-active cells:  95/104
generated registry cells:                 78
measured-profitable generic cells:        17
intentional original-policy cells:         9 (scalar-array negative range)
generated-cell speedup GM:             2.074x
old/new generated runtime GM:           1.009x
```

The nine non-active cells are the already-frozen scalar-array negative policy,
not failures.  `old/new = 1.009x` means the PF3 build was approximately 0.9%
faster in aggregate; there is no material regression.

Evidence hashes:

```text
pf3_contiguous_compat.json
947eb0911a2af9813c993dfe40dd596146e0549be1894d8c2425a8358cf96347
pf3_contiguous_compat.csv
f30771bd3001fd806537cb45f86c81a174fb97a7ec6f08552b11890b6244c7b2
```

PF3 passes.  The descriptor is an abstraction and compatibility change only;
new layout execution begins in PF4.
