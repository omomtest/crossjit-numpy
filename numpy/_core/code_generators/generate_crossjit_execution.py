#!/usr/bin/env python3
"""NumPy domain adapter for extension-neutral execution artifact generation."""
import argparse
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--generator-root', type=Path, required=True)
    p.add_argument('--output-plan', type=Path, required=True)
    p.add_argument('--output-loops', type=Path, required=True)
    p.add_argument('--output-backend', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.generator_root))
    from crossjit_generator.execution import (
        ExecutionLimits, Primitive, ScalarLoop, ScalarType,
        generate_backend_header, generate_execution_header, generate_loop_header)
    arithmetic = (Primitive('ADD', 0, 'left + right'),
                  Primitive('SUBTRACT', 1, 'left - right'),
                  Primitive('MULTIPLY', 2, 'left * right'))
    loops = []
    for width, name in ((64, 'double'), (32, 'float')):
        primitives = arithmetic + (
            Primitive('TRUE_DIVIDE', 3, 'left / right'),
            Primitive('NEGATIVE', 4, '-left'),
            Primitive('MINIMUM', 5, f'crossjit_min_{name}(left, right)'),
            Primitive('MAXIMUM', 6, f'crossjit_max_{name}(left, right)'))
        loops.append(ScalarLoop(name, ScalarType('float', width), primitives))
    for width in (32, 64):
        loops.append(ScalarLoop(f'uint{width}', ScalarType('unsigned', width),
            arithmetic + (Primitive('NEGATIVE', 4, '(TYPE)(0 - left)'),)))
    loops = tuple(loops)
    for path, content in ((args.output_plan, generate_execution_header(ExecutionLimits(), loops)),
                          (args.output_backend, generate_backend_header(index_type='npy_intp')),
                          (args.output_loops, generate_loop_header(loops, index_type='npy_intp', pair_fast_paths=True))):
        if not path.exists() or path.read_text() != content:
            path.write_text(content)


if __name__ == '__main__':
    main()
