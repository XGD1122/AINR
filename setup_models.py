"""
一键下载和设置所有对比模型的原始仓库

用法:
    cd spatial_gradient_benchmark
    python setup_models.py                # 下载全部 4 个模型
    python setup_models.py --models gaston suica  # 只下载指定模型
"""
import os, sys, subprocess, argparse, shutil

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BENCHMARK_DIR)
MODELS_DIR = os.path.join(BENCHMARK_DIR, "models")  # 所有模型克隆到这里

# 模型仓库信息
MODEL_REPOS = {
    "gaston": {
        "url": "https://github.com/raphael-group/GASTON.git",
        "pip_package": "gaston-spatial",  # 也可通过 pip 安装
        "description": "GASTON: Isodepth-based neural field for spatial transcriptomics",
    },
    "suica": {
        "url": "https://github.com/Szym29/SUICA.git",
        "pip_package": None,  # 只能从源码安装
        "description": "SUICA: GAE + INR for super-high dimensional ST data",
    },
    "stinr": {
        "url": "https://github.com/YisiLuo/STINR.git",
        "pip_package": None,
        "description": "STINR: INR-based spatial transcriptomics (CVPR 2025)",
    },
    "gaston_mix": {
        "url": "https://github.com/raphael-group/GASTON-Mix.git",
        "pip_package": None,
        "description": "GASTON-Mix: Mixture-of-experts neural field",
    },
}


def clone_repo(name, info, target_dir):
    """克隆单个仓库"""
    url = info["url"]
    print(f"\n{'='*60}")
    print(f"Cloning {name}: {info['description']}")
    print(f"  URL: {url}")
    print(f"  Target: {target_dir}")
    print(f"{'='*60}")

    if os.path.exists(target_dir):
        print(f"  Directory already exists. Pulling latest...")
        result = subprocess.run(["git", "-C", target_dir, "pull"],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  [OK] Updated successfully")
        else:
            print(f"  [WARN] Pull failed: {result.stderr[:200]}")
            print(f"  Removing and re-cloning...")
            shutil.rmtree(target_dir)
            result = subprocess.run(["git", "clone", url, target_dir],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  [OK] Re-cloned successfully")
            else:
                print(f"  [FAIL] Clone failed: {result.stderr[:300]}")
                return False
    else:
        result = subprocess.run(["git", "clone", url, target_dir],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  [OK] Cloned successfully")
        else:
            print(f"  [FAIL] Clone failed: {result.stderr[:300]}")
            return False

    # 检查是否有 requirements.txt，提示安装依赖
    req_file = os.path.join(target_dir, "requirements.txt")
    if os.path.exists(req_file):
        print(f"  [INFO] Found requirements.txt - install with:")
        print(f"    pip install -r {req_file}")

    return True


def pip_install(package_name):
    """pip 安装包"""
    print(f"\n  Installing {package_name} via pip...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", package_name],
                          capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  [OK] {package_name} installed successfully")
        return True
    else:
        print(f"  [FAIL] pip install failed: {result.stderr[:300]}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Setup contrast models for gradient benchmark')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['gaston', 'suica', 'stinr', 'gaston_mix'],
                       help='Models to setup')
    parser.add_argument('--use-pip', action='store_true',
                       help='Prefer pip install over git clone when available')
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 70)
    print("SPATIAL GRADIENT BENCHMARK - MODEL SETUP")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Target models: {args.models}")
    print("=" * 70)

    success = {}
    for name in args.models:
        if name not in MODEL_REPOS:
            print(f"\n[FAIL] Unknown model: {name}")
            continue

        info = MODEL_REPOS[name]

        # GASTON 可以通过 pip 安装
        if name == "gaston" and args.use_pip and info["pip_package"]:
            success[name] = pip_install(info["pip_package"])
        else:
            target_dir = os.path.join(MODELS_DIR, name)
            success[name] = clone_repo(name, info, target_dir)

    # ============================================================
    # 汇总
    # ============================================================
    print(f"\n{'='*70}")
    print("SETUP SUMMARY")
    print(f"{'='*70}")
    for name, ok in success.items():
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {name:15s} -> {status}")

    all_ok = all(success.values())
    if all_ok:
        print(f"\nAll models set up successfully!")
        print(f"Models directory: {MODELS_DIR}")
        print(f"\nNext steps:")
        print(f"  1. Install dependencies for each model (see requirements.txt in each dir)")
        print(f"  2. Run experiments: python run_all.py --slices 151673 151674 151675 151676")
    else:
        failed = [n for n, ok in success.items() if not ok]
        print(f"\nSome models failed: {failed}")
        print(f"Please check network connection or try manual download.")

    return all_ok


if __name__ == "__main__":
    main()
