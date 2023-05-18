# DVCLive with Optuna

A simple repo demonstrating how DVCLive can be used together with Optuna to keep track
of all hyperparameter tuning trials (including associated artifacts)

The virtual environment can be set up quickly by

```
make venv
```

and then activated by

```
source .venv/bin/activate
```

Set up `dvc` by `dvc init` and then run the dvc pipeline by `dvc repro`.

The main script is `dvclive_optuna.py`, the dvc pipeline is defined in `dvc.yaml` and
its parameters in `params.yaml`.

The custom callback used in the main script can be found in `custom_callback.py`.
