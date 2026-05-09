"""Monitor and orchestrate sequential experiment execution."""

import subprocess
import sys
import time
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
EXPERIMENTS = [
    ("A", "run_a.py"),
    ("B", "run_b.py"),
    ("C", "run_c.py"),
    ("D", "run_d.py"),
    ("E", "run_e.py"),
]


def run_experiment(name: str, script: str) -> bool:
    """Run an experiment and wait for completion."""
    
    print(f"\n{'='*70}")
    print(f"Starting Experiment {name}")
    print(f"{'='*70}\n")
    
    script_path = EXAMPLE_DIR / "experiment" / script
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(EXAMPLE_DIR),
        check=False
    )
    
    success = result.returncode == 0
    status = "✓ COMPLETED" if success else f"✗ FAILED (code {result.returncode})"
    print(f"\nExperiment {name}: {status}\n")
    
    return success


def main():
    """Run all experiments sequentially."""
    
    print("\n" + "="*70)
    print("Burgers Discrete Forward - Sequential Experiment Runner")
    print("="*70)
    
    start_time = time.time()
    results = {}
    
    for i, (name, script) in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] Processing Experiment {name}...")
        results[name] = run_experiment(name, script)
        
        if not results[name]:
            print(f"\n⚠️  Experiment {name} failed. Continuing with next...\n")
    
    # Summary
    elapsed = time.time() - start_time
    completed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} Experiment {name}")
    
    print(f"\nCompleted: {completed}/{total}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Results: experiment/实验A through experiment/实验E")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
