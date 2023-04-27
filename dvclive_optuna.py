import time

import dvc.api
import joblib
import optuna
from dvclive.optuna import DVCLiveCallback
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from dvclive import Live

PARAMS = dvc.api.params_show()


def objective(trial):
    X, y = make_classification(n_features=10, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    C = trial.suggest_float("C", 1e-7, 10.0, log=True)
    clf = LogisticRegression(C=C)

    with Live(save_dvc_exp=True) as live:
        live.log_params(trial.params)
        clf.fit(X_train, y_train)

        # Save a trained model to a file.
        with open(PARAMS["paths"]["trial_model"], "wb") as f:
            joblib.dump(clf, f)
        live.log_artifact(PARAMS["paths"]["trial_model"], name=f"trial-{trial.number}")

    time.sleep(1)

    return clf.score(X_test, y_test)


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="dvclive-experiments")
    study.optimize(objective, n_trials=5)
    joblib.dump(study, PARAMS["paths"]["study"], compress=1)
