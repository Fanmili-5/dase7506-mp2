"""Generic frozen-model official CPU evaluation; references isolated from predictor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
THREADS = ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'GOTO_NUM_THREADS')
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def child():
    snapshot, data, split, tool, output, predictions = sys.argv[2:]
    snap = Path(snapshot).resolve()
    for name in THREADS:
        os.environ[name] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path[:0] = [str(snap / 'code'), str(snap / 'methods')]

    def guard(event, args):
        if event.startswith(('socket.connect', 'socket.bind', 'socket.getaddrinfo')) or event in ('subprocess.Popen', 'os.system'):
            raise PermissionError('Frozen offline CPU prediction/scoring')
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0]))
            if tool == 'predict.py' and path.name == f'gsm8k_{split}.jsonl' and path.parent.name != 'candidates':
                raise PermissionError('Prediction cannot read scoring references')
    sys.addaudithook(guard)
    sys.argv = [str(snap / 'code' / tool), '--split', split, '--data-dir', data, '--output', output]
    if tool == 'predict.py':
        sys.argv += ['--method', str(snap / json.loads((snap / 'freeze.json').read_text())['method']), '--checkpoint', str(snap / 'checkpoint')]
    else:
        sys.argv += ['--predictions', predictions]
    runpy.run_path(sys.argv[0], run_name='__main__')
    assert 'torch' not in sys.modules and 'sklearn' not in sys.modules


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--split', choices=('validation', 'test', 'audit'), required=True)
    p.add_argument('--final-test', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.split == 'test' and not a.final_test:
        p.error('Final frozen test scoring requires --final-test')
    sys.path.insert(0, str(ROOT / 'scripts'))
    from freeze_inference import verify_snapshot
    manifest = verify_snapshot(a.snapshot)
    assert manifest['selection_frozen_before_test']
    freeze_hash = sha(a.snapshot / 'freeze.json')
    a.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH='', PYTHONNOUSERSITE='1')
    env.update({k: '1' for k in THREADS})
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    predictions = a.output / f'{a.split}-predictions.json'
    score = a.output / f'{a.split}-score.json'
    records = []
    started = time.perf_counter()
    try:
        for tool, target in (('predict.py', predictions), ('score_predictions.py', score)):
            cmd = [sys.executable, '-I', str(Path(__file__).resolve()), '--child',
                   str(a.snapshot.resolve()), str(a.data_dir.resolve()), a.split, tool,
                   str(target.resolve()), str(predictions.resolve())]
            tick = time.perf_counter()
            peak = 0
            with (a.output / (tool + '.log')).open('w') as log:
                proc = subprocess.Popen(cmd, env=env, cwd=a.snapshot, stdout=log, stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    try:
                        peak = max(peak, len(list(Path(f'/proc/{proc.pid}/task').iterdir())))
                    except FileNotFoundError:
                        pass
                    # Include the single-threaded monitor in the total limit.
                    if peak + 1 > 4 or time.perf_counter() - started > 30:
                        proc.kill()
                        proc.wait()
                        raise RuntimeError('Local total-thread/30-second budget exceeded')
                    time.sleep(.01)
                if proc.returncode:
                    raise RuntimeError(f'{tool} failed: see saved log')
            records.append({'tool': tool, 'seconds': time.perf_counter() - tick, 'sampled_child_threads': peak,
                            'controller_threads': 1, 'exit_code': proc.returncode})
        total = time.perf_counter() - started
        verify_snapshot(a.snapshot)
        assert sha(a.snapshot / 'freeze.json') == freeze_hash
        official = json.loads(score.read_text())
        cpu = next(x.split(':', 1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name'))
        result = {'state': 'complete', 'split': a.split, 'official_score': official,
                  'official_score_sha256': sha(score), 'snapshot_freeze_sha256': freeze_hash,
                  'complete_cpu_seconds': total, 'asset_bytes': manifest['declared_inference_asset_bytes'],
                  'processes': records, 'cpu_model': cpu, 'reference_machine_qualification': False,
                  'local_resource_pass': total <= 30 and manifest['declared_inference_asset_bytes'] <= 16 * 1024**2,
                  'no_test_tuning_or_error_analysis': True}
        (a.output / 'resources.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        (a.output / 'failure.json').write_text(json.dumps({'state': 'failed', 'error': repr(exc), 'processes': records}) + '\n')
        raise


if __name__ == '__main__':
    child() if len(sys.argv) > 1 and sys.argv[1] == '--child' else main()
