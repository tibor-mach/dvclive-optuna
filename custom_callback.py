# ruff: noqa: ARG002
import joblib
from dvclive import Live
from dvclive.optuna import DVCLiveCallback


class CustomOptunaCallback(DVCLiveCallback):
    def __init__(
        self, metric_name="metric", save_model=False, save_study=False, **kwargs
    ) -> None:
        kwargs["dir"] = kwargs.get("dir", "dvclive-optuna")
        kwargs.pop("save_dvc_exp", None)
        self.metric_name = metric_name
        self.save_model = save_model
        self.save_study = save_study
        self.live_kwargs = kwargs

    def __call__(self, study, trial) -> None:
        with Live(save_dvc_exp=True, **self.live_kwargs) as live:
            live.log_params(trial.params)
            # the _log_metrics method is inherited from DVCLiveCallback
            self._log_metrics(trial.values, live)

            if self.save_model:
                self._log_model(trial, live)
            if self.save_study:
                self._log_study(study, live)

    def _log_model(self, trial, live):
        model_path = trial.user_attrs.get("model_path")
        # if no model name is provided, the path stem will be used for logging
        model_name = trial.user_attrs.get("model_name")

        if model_path:
            live.log_artifact(model_path, name=model_name, type="model")
        else:
            raise ValueError(
                """
            Model path not provided. Please add a user attribute called "model_path"
            to the Optuna trial in the objective function definition.
            """
            )

    def _log_study(self, study, live):
        study_path = study.user_attrs.get("study_path")

        # if no study name is provided in user attributes, the study_name attribute
        # of the study object will be used. If that is also unspecified,
        # the path stem will be used for logging.
        study_name = study.user_attrs.get("study_name", study.study_name)

        if study_path:
            # In order to include the latest finished trial in the saved study,
            # we only serialize the study here instead of inside the objective function
            with open(study_path, "wb") as f:
                joblib.dump(study, f)

            live.log_artifact(study_path, study_name)

        else:
            raise ValueError(
                """
            Study path not provided. Please add a user attribute called "study_path"
            to the optuna study.
            """
            )
