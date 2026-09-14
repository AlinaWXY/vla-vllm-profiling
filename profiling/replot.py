"""Replot one retained result into a newly allocated experiment directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from profiling.coarse_roofline import plot
from profiling.experiments import timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory allocated with python -m profiling.experiments')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not (output / 'experiment.json').is_file():
        parser.error('Allocate --output using profiling.experiments first.')
    if (output / 'summary.json').exists():
        parser.error('Output already contains a result; allocate a new experiment.')
    summary = json.loads((source / 'summary.json').read_text())
    options = {'xbounds': (10, 2048)}
    if 'sglang' in summary['framework']['path'].lower():
        options.update(
            caption='π0.5 / SGLang 5200508  |  batch 1  |  3 valid cameras  |  prefix 918  |  10 denoising steps',
            footnote='Native fused kernels; eager NCU dispatch. Attention includes QK / softmax / PV. FFN up includes gate / activation.',
            region_label_y=8,
            label_offsets={'vlm': {'QKV': (16,-16), 'O projection': (-100,-28), 'Attention (fused)': (-105,-38)},
                           'action_expert': {'QKV': (25,0), 'FFN down': (30,-26), 'O projection': (-6,-32),
                                             'Attention (fused)': (12,10)}})
    stamp = plot(summary['operators'], output, summary['empirical_ceilings'],
                 [source / 'summary.json', source / 'operators.csv',
                  Path(__file__).resolve(), Path(__file__).with_name('coarse_roofline.py').resolve()],
                 summary['collection_started_at'] + ' — ' + summary['collection_finished_at'], **options)
    summary['plot_stamp'] = stamp
    summary['replot'] = {'source': str(source), 'axis_base': 10, 'measurements_unchanged': True,
                         'replotted_at': timestamp(), 'xbounds': [10,2048], 'ybounds': [4,180]}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    for name in ('operators.csv', 'kernels.csv', 'mapping.csv', 'nvtx_parent_audit.json'):
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    (output / 'README.md').write_text(
        f'# L20 DRAM Roofline replot\n\nSource: {source}\n\n'
        f'Collection: {summary["collection_started_at"]} — {summary["collection_finished_at"]}\n\n'
        f'Plot: {stamp}\n\nMeasurements unchanged; decimal logarithmic axes.\n\n'
        '![Roofline](major_operators.png)\n')
    print(output / 'major_operators.png')


if __name__ == '__main__':
    main()
