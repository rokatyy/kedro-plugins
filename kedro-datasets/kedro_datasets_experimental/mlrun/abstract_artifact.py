"""Base class for MLRun datasets."""
from __future__ import annotations

import mlrun.artifacts
from kedro.io import AbstractDataset
from typing import Any, Optional
from .context_manager import MLRunContextManager


class MLRunAbstractDataset(AbstractDataset):
    """Base class for MLRun datasets that provides access to shared context."""

    DEFAULT_LOAD_ARGS: dict[str, Any] = {}
    DEFAULT_SAVE_ARGS: dict[str, Any] = {"index": False}

    def __init__(self,
                 load_args: dict[str, Any] | None = None,
                 save_args: dict[str, Any] | None = None, ) -> None:
        self._ctx_manager = MLRunContextManager()
        self._load_args = {**self.DEFAULT_LOAD_ARGS, **(load_args or {})}
        self._save_args = {**self.DEFAULT_SAVE_ARGS, **(save_args or {})}

    def load(self) -> Optional[mlrun.artifacts.Artifact]:
        return self._ctx_manager.context.get_artifact(**self._load_args)

    def save(self, data: Any) -> None:
        self._ctx_manager.context.log_artifact(item=data, **self._save_args)

    def _describe(self) -> dict[str, Any]:
        return {
            "load_args": self._load_args,
            "save_args": self._save_args,
            "mlrun_project_name": self._ctx_manager.project.name,
        }