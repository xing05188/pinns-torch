from runner import fixed_output_dir, run_experiment


if __name__ == "__main__":
    raise SystemExit(
        run_experiment(
            [
                f"hydra.run.dir=\"{fixed_output_dir('实验A')}\"",
            ]
        )
    )
