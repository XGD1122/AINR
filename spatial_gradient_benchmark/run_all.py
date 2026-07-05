"""
批量运行所有模型的梯度实验
按顺序对指定 DLPFC 切片运行 AINR, GASTON, SUICA, STINR, GASTON-Mix

用法:
    cd spatial_gradient_benchmark
    python run_all.py --slices 151673 151674 151675 151676 --models all

也可以只运行特定模型:
    python run_all.py --slices 151673 --models gaston suica
"""
import os, sys, subprocess, argparse, time
from datetime import datetime
import pandas as pd

# ============================================================
# 配置
# ============================================================
BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CONFIGS = {
    'ainr': {
        'dir': 'ainr',
        'script': 'run_gradient.py',
        'description': 'AINR (Proposed) - Attention-Guided INR',
        'per_slice': False,  # 一次处理多个切片
    },
    'gaston': {
        'dir': 'gaston',
        'script': 'run_gradient.py',
        'description': 'GASTON - Isodepth-based neural field',
        'per_slice': True,   # 逐切片处理
    },
    'suica': {
        'dir': 'suica',
        'script': 'run_gradient.py',
        'description': 'SUICA - GAE + INR joint framework',
        'per_slice': True,
    },
    'stinr': {
        'dir': 'stinr',
        'script': 'run_gradient.py',
        'description': 'STINR - Pure INR for ST data',
        'per_slice': True,
    },
    'gaston_mix': {
        'dir': 'gaston_mix',
        'script': 'run_gradient.py',
        'description': 'GASTON-Mix - Mixed membership neural field',
        'per_slice': True,
    },
}

# ============================================================
# 参数解析
# ============================================================
parser = argparse.ArgumentParser(description='Run all spatial gradient experiments')
parser.add_argument('--slices', type=str, nargs='+',
                    default=['151673', '151674', '151675', '151676'],
                    help='Slice IDs to process')
parser.add_argument('--models', type=str, nargs='+',
                    default=['all'],
                    help='Models to run: ainr, gaston, suica, stinr, gaston_mix, or all')
args = parser.parse_args()

if 'all' in args.models:
    models_to_run = list(MODEL_CONFIGS.keys())
else:
    models_to_run = [m for m in args.models if m in MODEL_CONFIGS]

slices = args.slices

print("=" * 70)
print("SPATIAL GRADIENT BENCHMARK - RUN ALL EXPERIMENTS")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Slices: {slices}")
print(f"Models: {models_to_run}")
print("=" * 70)

# ============================================================
# 记录运行时信息
# ============================================================
run_log = []

for model_name in models_to_run:
    cfg = MODEL_CONFIGS[model_name]
    model_dir = os.path.join(BENCHMARK_DIR, cfg['dir'])
    script_path = os.path.join(model_dir, cfg['script'])

    if not os.path.exists(script_path):
        print(f"\n[SKIP] {model_name}: script not found at {script_path}")
        continue

    print(f"\n{'='*70}")
    print(f"Running: {cfg['description']}")
    print(f"Script:  {script_path}")
    print(f"{'='*70}")

    try:
        if cfg['per_slice']:
            # 逐切片运行
            for slice_id in slices:
                print(f"\n  --- {model_name}: Slice {slice_id} ---")
                start = time.time()

                cmd = [sys.executable, script_path, '--slice', slice_id]
                result = subprocess.run(cmd, cwd=model_dir, capture_output=False)

                elapsed = time.time() - start
                status = "OK" if result.returncode == 0 else f"FAIL({result.returncode})"
                run_log.append({
                    'Model': model_name, 'Slice': slice_id,
                    'Status': status, 'Time(s)': round(elapsed, 1),
                    'Timestamp': datetime.now().strftime('%H:%M:%S')
                })
                print(f"  {model_name} / {slice_id}: {status} ({elapsed:.1f}s)")

        else:
            # AINR: 一次处理多个切片
            print(f"\n  --- {model_name}: Joint slices {slices} ---")
            start = time.time()

            slice_args = []
            for s in slices:
                slice_args.extend(['--slice_idx', s])
            cmd = [sys.executable, script_path] + slice_args + ['--slice_count', str(len(slices))]

            result = subprocess.run(cmd, cwd=model_dir, capture_output=False)

            elapsed = time.time() - start
            status = "OK" if result.returncode == 0 else f"FAIL({result.returncode})"
            for s in slices:
                run_log.append({
                    'Model': model_name, 'Slice': str(s),
                    'Status': status, 'Time(s)': round(elapsed / len(slices), 1),
                    'Timestamp': datetime.now().strftime('%H:%M:%S')
                })
            print(f"  {model_name}: {status} ({elapsed:.1f}s total)")

    except Exception as e:
        print(f"  ERROR running {model_name}: {e}")
        run_log.append({
            'Model': model_name, 'Slice': 'ALL',
            'Status': f'ERROR: {str(e)[:50]}', 'Time(s)': 0,
            'Timestamp': datetime.now().strftime('%H:%M:%S')
        })

# ============================================================
# 汇总
# ============================================================
print(f"\n{'='*70}")
print("EXPERIMENT SUMMARY")
print(f"{'='*70}")

df_log = pd.DataFrame(run_log)
print(df_log.to_string(index=False))

log_path = os.path.join(BENCHMARK_DIR, "results", "run_log.csv")
df_log.to_csv(log_path, index=False)
print(f"\nRun log saved to: {log_path}")

# 统计
successful = len(df_log[df_log['Status'] == 'OK'])
total = len(df_log)
print(f"\nCompleted: {successful}/{total} experiments")
if successful == total:
    print("All experiments completed successfully!")
else:
    print(f"Warning: {total - successful} experiments failed. Check individual logs.")
