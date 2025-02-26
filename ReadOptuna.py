import optuna


def read_study_results(study_name: str, storage_url: str):
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_url
        )
        
        # print("Best trials:")
        # for trial in study.best_trials:
        #     print(f"Trial #{trial.number}")
        #     print(f"  Values: {trial.values}")
        #     print("  Params:")
        #     for key, value in trial.params.items():
        #         print(f"    {key}: {value}")
                
        filtered = list(filter(lambda x: x.values is not None and x.values[0] > 84, study.best_trials))
        filtered.sort(key=lambda x: x.values[1], reverse=True)
        
        print("Best trials above 84%:")
        for trial in filtered:
            print(f"Trial #{trial.number}")
            print(f"  Values: {trial.values}")
            print("  Params:")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")

        # print(f"All trials for study '{study_name}':")
        # for trial in study.trials:
        #     print(f"  Trial number: {trial.number}")
        #     print(f"    Value: {trial.values}")
        #     print(f"    Parameters:")
        #     for key, value in trial.params.items():
        #         print(f"      {key}: {value}")
        #     print(f"    State: {trial.state}")
        #     print("-" * 50)

    except Exception as e:
        print(f"Failed to read the study: {e}")


if __name__ == "__main__":
    study_name = "Optimize_hexcirclenet_New"
    storage_url = "sqlite:///optuna_study.db"

    read_study_results(study_name, storage_url)
