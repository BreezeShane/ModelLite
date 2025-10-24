"""
    Definition of decorator.
"""

from typing import Dict, Callable, Any
from omegaconf import DictConfig

# onnx, torchscript, tensorrt, tvm, coreml
_EXPORT_FUNCTION_REGISTRY: Dict[str, Callable[[Any, DictConfig], Any]] = {}

def register_exporter(engine_name: str):
    """ Decorator: Registering exported functions. """
    def decorator(func: Callable[[Any, DictConfig], Any]):
        _EXPORT_FUNCTION_REGISTRY[engine_name] = func
        return func
    return decorator

def get_exporter(engine_name: str) -> Callable[[Any, DictConfig], Any]:
    """ Get the exported function. """
    if engine_name not in _EXPORT_FUNCTION_REGISTRY:
        available = list(_EXPORT_FUNCTION_REGISTRY.keys())
        raise ValueError(f"Unknown Engine: {engine_name}. Available Engines: {available}")
    return _EXPORT_FUNCTION_REGISTRY[engine_name]
