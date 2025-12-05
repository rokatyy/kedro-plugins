"""MLRun model dataset for Kedro."""
import os
import tempfile
from typing import Any

import joblib
from mlrun import get_dataitem

from .abstract_artifact import MLRunAbstractDataset


class MLRunModel(MLRunAbstractDataset):
    """Dataset for saving/loading models via MLRun."""

    def __init__(
        self,
        key: str | None = None,
        framework: str = "sklearn",
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(key=key, save_args=save_args, load_args=load_args)
        self._framework = framework

    def save(self, data: Any) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.pkl")
            joblib.dump(data, model_path)

            self._ctx_manager.context.log_model(
                key=self._key,
                model_file=model_path,
                framework=self._framework,
                **self._save_args
            )

    def load(self) -> Any:
        artifact = self._ctx_manager.project.get_artifact(self._key)
        target_path = artifact.get_target_path()
        model_file = artifact.model_file
        local_path = get_dataitem(target_path + model_file).local()
        return joblib.load(local_path)

    def _describe(self) -> dict[str, Any]:
        return {
            **super()._describe(),
            "framework": self._framework,
        }
