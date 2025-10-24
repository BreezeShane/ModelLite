"""
    Optimizer Configuration Validation.
"""
import warnings
from functools import wraps

from .context_classes import TrainingContextConfig, QuantizationContextConfig

VALID_OPTIMIZERS = {
    "adam": ["lr", "weight_decay", "betas", "eps", "amsgrad"],
    "sgd": ["lr", "momentum", "weight_decay", "nesterov"],
    "adamw": ["lr", "weight_decay", "betas", "eps", "amsgrad"]
}

def validate_optimizer_config(func):
    """ Decorator for validating optimizer configuration. """
    @wraps(func)
    def wrapper(optim_config):
        valid_optimizers = ["adam", "sgd", "adamw", "rmsprop"]
        required_params = {"lr": float}

        if optim_config["name"] not in valid_optimizers:
            raise ValueError(f"Invalid optimizer: {optim_config['name']}")

        for param, param_type in required_params.items():
            if param not in optim_config:
                raise ValueError(f"Missing required parameter: {param}")
            if not isinstance(optim_config[param], param_type):
                raise ValueError(f"Parameter {param} should be {param_type}")

        return func(optim_config)
    return wrapper

def validate_training_config(config: TrainingContextConfig):
    """ Validate Training Configuration. """
    optim_name = config.optimizer.get("name")
    if optim_name not in VALID_OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {optim_name}")

    valid_params = VALID_OPTIMIZERS[optim_name]
    for param in config.optimizer:
        if param not in ["name"] + valid_params:
            warnings.warn(f"Unexpected parameter '{param}' for optimizer '{optim_name}'")

def validate_quantization_config(config: QuantizationContextConfig):
    """ Validate Quantization Configuration. """
    if config.calibration_samples < 10:
        warnings.warn("Very few calibration samples may affect quantization accuracy")

    if config.strategy == "static" and config.calibration_samples < 100:
        warnings.warn("Static quantization typically requires more calibration samples")
