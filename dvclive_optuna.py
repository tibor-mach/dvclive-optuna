import time

import dvc.api
import joblib
import optuna
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from optuna_artifacts import DVCLiveCallback

PARAMS = dvc.api.params_show()


def objective(trial):
    X, y = make_classification(n_features=10, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    C = trial.suggest_float("C", 1e-7, 10.0, log=True)
    clf = LogisticRegression(C=C)

    clf.fit(X_train, y_train)

    time.sleep(1)

    # let the DVCLiveCallback know path and name to the artifact which should be logged
    # TODO This is very implicit...but I can't see any elegant way around it
    trial.set_user_attr("artifact_path", "dvclive/trial_model.joblib")
    trial.set_user_attr("artifact_name", f"trial-model-{trial.number}")

    # Save a trained model to a file.
    with open(PARAMS["paths"]["trial_model"], "wb") as f:
        joblib.dump(clf, f)

    return clf.score(X_test, y_test)


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="dvclive-experiments")
    study.optimize(objective, n_trials=5, callbacks=[DVCLiveCallback(dir="dvclive")])
    joblib.dump(study, PARAMS["paths"]["study"], compress=1)
