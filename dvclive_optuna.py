import time

import dvc.api
import joblib
import optuna
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from dvclive import Live

PARAMS = dvc.api.params_show()


def objective(trial):
    trial_hyperparams = {
        "learning_rate": trial.suggest_float("learning_rate", 0.0001, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 10, 200, step=10),
    }

    clf = GradientBoostingClassifier(**trial_hyperparams)

    clf.fit(x_train, y_train)

    # Save a trained model to a file.
    model_path = PARAMS["paths"]["trial_model"]
    study_path = PARAMS["paths"]["study"]

    with open(model_path, "wb") as f:
        joblib.dump(clf, f)

    with open(study_path, "wb") as f:
        joblib.dump(study, f)

    mean_accuracy = clf.score(x_test, y_test)

    # Log trial metadata and artifacts with DVCLive
    with Live(dir="dvclive", save_dvc_exp=True) as live:
        # log the (non-default) hyperparameters of the trial (identical to trial_hyperparams)
        live.log_params(trial.params)

        # log the resulting metric used for hyperparameter optimization
        live.log_metric("mean_accuracy", mean_accuracy)

        # log the model artifact resulting from the current trial
        live.log_artifact(model_path, name=f"trial-model-{trial.number}", type="model")

        # log the study at current trial
        # note that since the current trial is not yet finished, the output of the objective
        # function will not be available in the study and its status will show up as RUNNING
        # There is no way around that when logging is implemented inside the objective function
        # aside from creating a copy of the study and the latest trial, editing them and logging
        # the resulting study
        live.log_artifact(study_path, name=f"study-at-trial-{trial.number}")

    return mean_accuracy


if __name__ == "__main__":
    x, y = make_classification(n_samples=1000, n_features=50, random_state=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    study = optuna.create_study(direction="maximize", study_name="dvclive-experiments")

    study.optimize(objective, n_trials=5)
    # with open(PARAMS["paths"]["study"], "wb") as f:
    #     joblib.dump(study, f, compress=1)
