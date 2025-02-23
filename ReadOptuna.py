import optuna


def read_study_results(study_name: str, storage_url: str):
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_url
        )

        print("Best trial:")
        print(f"  Trial number: {study.best_trial.number}")
        print(f"  Value (Validation Accuracy): {study.best_trial.value}")
        print("  Parameters:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
        print("\n" + "=" * 50 + "\n")

        print(f"All trials for study '{study_name}':")
        for trial in study.trials:
            print(f"  Trial number: {trial.number}")
            print(f"    Value: {trial.value}")
            print(f"    Parameters:")
            for key, value in trial.params.items():
                print(f"      {key}: {value}")
            print(f"    State: {trial.state}")
            print("-" * 50)

    except Exception as e:
        print(f"Failed to read the study: {e}")


if __name__ == "__main__":
    study_name = "Optimize_hexmagicnet"
    storage_url = "sqlite:///optuna_study.db"

    read_study_results(study_name, storage_url)
