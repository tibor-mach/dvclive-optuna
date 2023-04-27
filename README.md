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

## Issues

- Running the dvc pipeline in order to log the Optuna study at the end of training clashes
with logging of trials into separate experiments since the study object is only saved at the end of
the hyperparameter training.
- It would be nice to add logging of plots as well (apparently available now in DVCLive 2.8.)
