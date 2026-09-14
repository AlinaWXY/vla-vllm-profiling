"""Replot existing major-operator measurements with decimal logarithmic ticks."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

root = Path('/fact_data/xinyaowang/vla_vllm_profiling')
sys.path.insert(0, str(root))
from profiling.experiments import file_record, timestamp

source, output = (Path(arg).resolve() for arg in sys.argv[1:3])
summary = json.loads((source / 'summary.json').read_text())
snapshot = output / 'render_source'
snapshot.mkdir(exist_ok=True)
shutil.copy2(root / 'profiling/coarse_roofline.py', snapshot / 'coarse_roofline.py')
spec = importlib.util.spec_from_file_location('decimal_roofline', snapshot / 'coarse_roofline.py')
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
copied = []
for name in ('operators.csv', 'kernels.csv', 'mapping.csv', 'nvtx_parent_audit.json'):
    if (source / name).is_file():
        shutil.copy2(source / name, output / name)
        copied.append(name)
options = {'xbounds': (10, 2048)}
is_sglang = 'sglang' in summary['framework']['path']
if is_sglang:
    options.update(
        caption='π0.5 / SGLang 5200508  |  batch 1  |  3 valid cameras  |  prefix 918  |  10 denoising steps',
        footnote='Native fused kernels; eager NCU dispatch. Attention includes QK / softmax / PV. FFN up includes gate / activation.',
        region_label_y=8,
        label_offsets={'vlm': {'QKV': (16,-16), 'O projection': (-100,-28), 'Attention (fused)': (-105,-38)},
                       'action_expert': {'QKV': (25,0), 'FFN down': (30,-26), 'O projection': (-6,-32),
                                         'Attention (fused)': (12,10)}})
inputs = [source / 'summary.json', *(source / name for name in copied),
          snapshot / 'coarse_roofline.py', Path(__file__).resolve()]
stamp = renderer.plot(summary['operators'], output, summary['empirical_ceilings'], inputs,
                      summary['collection_started_at'] + ' — ' + summary['collection_finished_at'], **options)
old_stamp = summary['plot_stamp']
summary['plot_stamp'] = stamp
summary['replot'] = {'source': str(source), 'axis_base': 10, 'xbounds': [10,2048],
                     'ybounds': [4,180], 'measurements_unchanged': True,
                     'replotted_at': timestamp(), 'inputs': [file_record(path) for path in inputs]}
(output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
readme = (source / 'README.md').read_text().replace(old_stamp, stamp)
readme = readme.replace(f'— {source.parent.name}/{source.name}', f'— {output.parent.name}/{output.name}')
readme = readme.replace('横纵轴为 log2', '横纵轴为 log10，主刻度为 10、10²、10³')
readme = readme.replace('本次仅将坐标改为 2 的幂次刻度。', '本次仅将坐标改为 10 的幂次刻度。')
readme = readme.replace('preprocessing_comparison.json。', f"[preprocessing_comparison.json]({summary['capture']}/preprocessing_comparison.json)。")
readme += f'\n十进制对数坐标重绘：数据源 {source}；测量值与原采集时间保持不变。绘图时间：{stamp}。\n'
(output / 'README.md').write_text(readme)
assert all((source / name).read_bytes() == (output / name).read_bytes() for name in copied)
(output / 'verification.json').write_text(json.dumps({
    'axis_base': 10, 'data_identical_to': str(source), 'measurements_unchanged': True,
    'copied_artifacts': copied, 'verified_at': timestamp()
}, indent=2) + '\n')
print(output / 'major_operators.png')
