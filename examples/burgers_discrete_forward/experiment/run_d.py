from runner import fixed_output_dir, run_experiment


if __name__ == "__main__":
    raise SystemExit(
        run_experiment(
            [
                "q=100",
                "net.layers=[1,50,50,50,101]",
                f"hydra.run.dir=\"{fixed_output_dir('实验D')}\"",
            ]
        )
    )
