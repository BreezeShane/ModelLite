"""
    Eager Model quantization definitions.
    FX Graph quantization definition.
"""
import copy
from typing import Optional

import torch
from torch.ao.quantization import QConfig

from enumeration import QuantizationStrategy
from schemas.context_classes import QuantizationContextConfig
from .quantize_functions import (
    _fx_graph_quantization,
    _dynamic_quantization,
    _static_quantization,
    _custom_quantization
)

__all__ = [
    "quantize_model"
]

def quantize_model(
        cfg: QuantizationContextConfig,
        original_model,
        test_loader,
        custom_qconfig: Optional[QConfig] = None
    ):
    """
        Quantize a model using specified strategy and configuration.

        Args:
            config: Quantization configuration
            original_model: The model to be quantized
            test_loader: DataLoader for calibration (static quantization)
            custom_qconfig: QConfig object.

        Returns:
            Quantized model

        Raises:
            ValueError: If strategy is invalid
            RuntimeError: If quantization fails
    """
    torch.backends.quantized.engine = cfg.backend
    model = copy.deepcopy(original_model)
    model.eval()
    model = model.to(cfg.device)

    strategy_enum = QuantizationStrategy.from_config(cfg.strategy)
    print(f"Info: Use Quantization Strategy: {strategy_enum}.")

    try:
        if strategy_enum == QuantizationStrategy.FX_GRAPH:
            quantized_model = _fx_graph_quantization(cfg, model, test_loader)
        elif strategy_enum == QuantizationStrategy.DYNAMIC:
            quantized_model = _dynamic_quantization(cfg, model)
        elif strategy_enum == QuantizationStrategy.STATIC:
            quantized_model = _static_quantization(cfg, model, test_loader)
        else:
            quantized_model = _custom_quantization(cfg, model, test_loader, custom_qconfig)

        return quantized_model
    except (RuntimeError, ValueError) as e:
        print(f"Quantization with {strategy_enum} failed with {type(e).__name__}: {e}")
        if strategy_enum != QuantizationStrategy.DYNAMIC:
            print("Falling back to dynamic quantization...")
            return _dynamic_quantization(cfg, model)
        raise
