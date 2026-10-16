"""Compile Provider-owned, measured complete-region decisions (no kernel code)."""
import argparse
import hashlib
import json
from pathlib import Path


def accepted_keys(policy):
    version = policy.get('schema_version')
    if version != 2 or policy.get('backend_version') != 'native-region-v3':
        raise ValueError('unsupported native policy identity')
    if policy.get('selection_basis') != 'complete-region-original-and-existing':
        raise ValueError('unsupported native selection basis')
    keys = []
    seen = set()
    for row in policy['records']:
        key = row['key']
        if key in seen:
            raise ValueError('duplicate native region key')
        seen.add(key)
        if (not isinstance(key, str) or not key.startswith(('E1|', 'E2|', 'G1|', 'S1|'))
                or len(key) >= 16384 or 'b:3' not in key.split('|')):
            raise ValueError('invalid native region key')
        if row['decision'] != 'ACCEPT':
            continue
        evidence = row['evidence']
        path = Path(evidence['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != evidence['sha256']:
            raise ValueError('native measurement drift')
        measured = json.loads(path.read_text())
        points = [p for p in measured['records'] if p['key'] == key]
        if len(points) != 1:
            raise ValueError('native decision lacks exact complete-path measurement')
        point = points[0]
        if not (point['pairs'] >= 10 and point['ci95'][0] > 1 and point['correct']
                and point['run_mode'] == 'development' and point['complete_region']
                and point['exec_miss'] == 0 and point['exec_error'] == 0):
            raise ValueError('native decision lacks credible positive benefit')
        if not (
                point.get('alternative_pairs', 0) >= 10
                and point.get('alternative_correct')
                and point.get('alternative_complete_region')
                and point.get('alternative_ci95', [0])[0] > 1
                and point.get('alternative_exec_miss') == 0
                and point.get('alternative_exec_error') == 0):
            raise ValueError('native decision does not beat existing complete execution')
        keys.append(key)
    return sorted(keys)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    keys = accepted_keys(json.loads(a.policy.read_text())) if a.policy else []
    body = ',\n'.join('    ' + json.dumps(k, ensure_ascii=True) for k in keys) or '    NULL'
    a.output.write_text('/* Frozen Provider native-region selection. */\n'
                       '#define NUMPY_CROSSJIT_NATIVE_POLICY_COUNT ' + str(len(keys)) + '\n'
                       'static const char *const numpy_crossjit_native_policy_keys[] = {\n'
                       + body + '\n};\n')


if __name__ == '__main__':
    main()
