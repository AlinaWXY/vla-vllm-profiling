"""Keep an independent workload manifest for every strict NCU application pass."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

from profiling.experiments import timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    args, benchmark_args = parser.parse_known_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for index in range(100):
        attempt = args.output_root / f'{index:02d}'
        try:
            attempt.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError('Too many application replay passes')
    record = {'started_at': timestamp(), 'pid': os.getpid(), 'pass_index': index,
              'benchmark_args': benchmark_args}
    (attempt/'attempt.json').write_text(json.dumps(record, indent=2)+'\n')
    sys.argv = ['profiling.bench_pi05', *benchmark_args, '--output', str(attempt/'workload')]
    try:
        runpy.run_module('profiling.bench_pi05', run_name='__main__')
        record['status'] = 'completed'
    except BaseException:
        record['status'] = 'failed'
        raise
    finally:
        record['finished_at'] = timestamp()
        (attempt/'attempt.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
