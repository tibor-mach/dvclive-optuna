import dvc.api
import joblib
import optuna
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from custom_callback import CustomOptunaCallback

PARAMS = dvc.api.params_show()

import os


def objective(trial):
    trial_hyperparams = {
        "learning_rate": trial.suggest_float("learning_rate", 0.0001, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 10, 200, step=10),
    }

    clf = GradientBoostingClassifier(**trial_hyperparams)

    clf.fit(x_train, y_train)

    # Save the trained model to a file.
    model_path = os.path.join("dvclive-optuna", "model.pkl")
    with open(model_path, "wb") as f:
        joblib.dump(clf, f)
    # add a "model_path" user attribute to the trial for logging with DVCLive
    trial.set_user_attr("model_path", model_path)

    predictions = clf.predict(x_test)
    accuracy = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)

    return accuracy, precision


if __name__ == "__main__":
    x, y = make_classification(
        n_samples=1000, n_features=50, random_state=PARAMS["seed"]
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=PARAMS["seed"]
    )

    study = optuna.create_study(
        # TPESampler is the default, we only specify sampler here to fix the random seed
        sampler=optuna.samplers.TPESampler(seed=PARAMS["seed"]),
        directions=["maximize", "maximize"],
        study_name="dvclive-experiments",
    )

    # add a "study_path" user attribute to the study for logging with DVCLive
    study_path = os.path.join("dvclive-optuna", "study.pkl")
    study.set_user_attr("study_path", study_path)

    study.optimize(
        objective,
        n_trials=PARAMS["n_trials"],
        callbacks=[
            CustomOptunaCallback(
                save_model=True, save_study=True, metric_name=["recall", "precision"]
            )
        ],
    )
