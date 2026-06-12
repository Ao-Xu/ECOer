"""One-shot runner for the ECOer experiment pipeline."""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.classifiers import setup_all_classifiers
from src.preprocessing import load_processed, setup_all_datasets


EXPERIMENTS = [
    "exp1", "exp2", "exp3", "exp4", "exp5",
    "exp6", "exp7", "exp8", "exp9", "exp10",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run ECOer experiments")
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip dataset preprocessing and classifier training",
    )
    parser.add_argument("--only", choices=EXPERIMENTS, help="Run one experiment")
    return parser.parse_args()


def step0_setup():
    print("\n" + "=" * 60)
    print("STEP 0: Dataset preprocessing and classifier training")
    print("=" * 60)
    t0 = time.time()
    setup_all_datasets()
    data_map = {ds: load_processed(ds) for ds in config.DATASETS}
    setup_all_classifiers(data_map)
    print(f"Setup done in {time.time() - t0:.1f}s")


def _run_named(module_name: str, display: str, run_attr: str = "run"):
    print("\n" + "=" * 60)
    print(display)
    print("=" * 60)
    module = __import__(module_name)
    if run_attr == "run":
        module.run()
        return

    old_argv = sys.argv[:]
    sys.argv = [module_name + ".py"]
    try:
        getattr(module, run_attr)()
    finally:
        sys.argv = old_argv


def main():
    args = parse_args()
    total_t0 = time.time()

    if not args.skip_setup and args.only is None:
        step0_setup()

    runners = {
        "exp1": lambda: _run_named("run_exp1_approx", "STEP 1: R2SNN Approximation"),
        "exp2": lambda: _run_named("run_exp2_proximity", "STEP 2: Proximity / Validity"),
        "exp3": lambda: _run_named("run_exp3_quality", "STEP 3: DP / IM / Sparsity"),
        "exp4": lambda: _run_named("run_exp4_ablation", "STEP 4: Ablation Study"),
        "exp5": lambda: _run_named("run_exp5_stats", "STEP 5: Statistical Tests"),
        "exp6": lambda: _run_named("run_exp6_runtime", "STEP 6: Runtime / Hidden-Width / Memory", "main"),
        "exp7": lambda: _run_named("run_exp7_stability", "STEP 7: Ridge Reconstruction Stability", "main"),
        "exp8": lambda: _run_named("run_exp8_distortion", "STEP 8: Metric-Distortion Diagnostic", "main"),
        "exp9": lambda: _run_named("run_exp9_sampling_ablation", "STEP 9: Sampling / Training Ablation", "main"),
        "exp10": lambda: _run_named("run_exp10_rf_specialized", "STEP 10: RF-Specific Sanity Baseline", "main"),
    }

    if args.only:
        runners[args.only]()
    else:
        for exp_name in EXPERIMENTS:
            runners[exp_name]()

    total = time.time() - total_t0
    print("\n" + "=" * 60)
    print(f"EXPERIMENTS COMPLETED in {total / 3600:.2f}h ({total:.0f}s)")
    print(f"Figures saved to: {config.FIGURES_DIR}")
    print(f"Results saved to: {config.RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
