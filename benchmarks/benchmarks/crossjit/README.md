# CrossJIT fixed-fusion calibration

`run_fixed_fusion_break_even.py` measures ordinary NumPy and forced-admission
CrossJIT fixed fusion in alternating, independent processes.  Main timing
always has detailed Provider diagnostics disabled.  Every CrossJIT cell is
invalid unless Provider counters prove acceptance, publication, execution, and
at least one execution HIT.

The default matrix is float64 `CONTIGUOUS_SAME_SHAPE`, all required element
counts from 16 through 65536, linear regions of length 2–6, a branched DAG,
array–scalar, and scalar–array expressions.  It uses five independent
processes per mode and writes both JSON (including raw samples and execution
evidence) and CSV.

```bash
PYTHON_JIT=1 /path/to/crossjit/venv/bin/python3 \
  benchmarks/benchmarks/crossjit/run_fixed_fusion_break_even.py \
  --python /path/to/crossjit/venv/bin/python3 \
  --output-dir benchmarks/benchmarks/crossjit/results
```

CrossJIT workers set `NPY_CROSSJIT_MIN_WORK=0`.  Results are calibration of
the current precompiled implementation, not production performance claims and
not a profitability policy.  Baseline workers use the same Python and NumPy
build with `PYTHON_JIT=1` and `PYTHON_JIT_CROSSJIT=0`.

The layout fixture layer also supports `CONTIGUOUS_BROADCAST`, `ZERO_STRIDE`,
`INNER_CONTIGUOUS`, and `GENERAL_STRIDED` for later P7 experiments.  Those
layouts are deliberately not calibrated or enabled by P1.

## Generator G0 feasibility oracle

`run_generator_feasibility.py` compiles the benchmark-only L2/L4/L6
straight-line C oracle, runs the complete P1 N grid in independent processes,
joins the measurements to the immutable P1 JSON, and emits compiler
vectorization and assembly evidence:

```bash
/path/to/crossjit/venv/bin/python3 \
  benchmarks/benchmarks/crossjit/run_generator_feasibility.py
```

The quantitative gate decision and methodology are recorded in
`G0_RESULTS.md`.  `g0-build/` contains reproducible compiler artifacts and is
not a source directory.
