"""
run_all.py — One-shot script to run the full ECOer experiment pipeline.

Usage:
    python run_all.py [--skip-setup] [--only exp1] [--device cuda]

Steps:
  0. Setup: preprocess all 6 datasets, train 18 classifiers
  1. Exp 1: R2SNN approximation comparison
  2. Exp 2: Proximity metrics (ℓ₁, ℓ₂)
  3. Exp 3: Quality metrics (DP, IM, Sparsity)
  4. Exp 4: Ablation study
  5. Exp 5: Statistical significance tests
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.preprocessing import setup_all_datasets, load_processed
from src.classifiers import setup_all_classifiers


def parse_args():
    p = argparse.ArgumentParser(description="Run ECOer experiments")
    p.add_argument("--skip-setup", action="store_true",
                   help="Skip dataset preprocessing and classifier training")
    p.add_argument("--only", default=None,
                   choices=["exp1", "exp2", "exp3", "exp4", "exp5"],
                   help="Run only a specific experiment")
    return p.parse_args()


def step0_setup():
    print("\n" + "="*60)
    print("STEP 0: Dataset preprocessing and classifier training")
    print("="*60)
    t0 = time.time()
    setup_all_datasets()
    data_map = {ds: load_processed(ds) for ds in config.DATASETS}
    setup_all_classifiers(data_map)
    print(f"Setup done in {time.time()-t0:.1f}s")


def step1():
    print("\n" + "="*60)
    print("STEP 1: R2SNN Approximation (Exp 1)")
    print("="*60)
    import run_exp1_approx
    run_exp1_approx.run()


def step2():
    print("\n" + "="*60)
    print("STEP 2: Proximity Metrics (Exp 2)")
    print("="*60)
    import run_exp2_proximity
    run_exp2_proximity.run()


def step3():
    print("\n" + "="*60)
    print("STEP 3: Quality Metrics (Exp 3)")
    print("="*60)
    import run_exp3_quality
    run_exp3_quality.run()


def step4():
    print("\n" + "="*60)
    print("STEP 4: Ablation Study (Exp 4)")
    print("="*60)
    import run_exp4_ablation
    run_exp4_ablation.run()


def step5():
    print("\n" + "="*60)
    print("STEP 5: Statistical Tests (Exp 5)")
    print("="*60)
    import run_exp5_stats
    run_exp5_stats.run()


def main():
    args = parse_args()
    total_t0 = time.time()

    if not args.skip_setup and args.only is None:
        step0_setup()

    runners = {
        "exp1": step1,
        "exp2": step2,
        "exp3": step3,
        "exp4": step4,
        "exp5": step5,
    }

    if args.only:
        runners[args.only]()
    else:
        for fn in runners.values():
            fn()

    total = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETED in {total/3600:.2f}h ({total:.0f}s)")
    print(f"Figures saved to: {config.FIGURES_DIR}")
    print(f"LaTeX figures:    {config.LATEX_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
