from contextlib import contextmanager
from typing import Any

from vllm.model_executor.layers.expert_prefetch.expert_cache import (
    ExpertBuffer,
    ExpertCache,
)

from vllm.model_executor.layers.expert_prefetch.expert_predictor import (
    ExpertPredictorModel,
)

_config: dict[str, Any] | None = None


@contextmanager
def override_config(config):
    global _config
    old_config = _config
    _config = config
    yield
    _config = old_config


def get_config() -> dict[str, Any] | None:
    return _config


__all__ = [
    "ExpertBuffer",
    "ExpertCache",
    "ExpertPredictorModel",
    "override_config",
    "get_config",
]
