"""
    Functions of quantization.
"""
from tqdm import tqdm
from typing import Optional

import torch
from torch.ao.quantization import quantize_fx, QConfig, get_default_qconfig_mapping

from schemas.context_classes import QuantizationContextConfig

def _calibrate_quantized_model(cfg: QuantizationContextConfig, model, calibration_loader) -> None:
    """ Calibrate Quantized Model. """
    print(f"Running calibration with {cfg.calibration_samples} samples...")
    model.eval()
    model = model.to(cfg.device)
    calibration_count = 0

    data_iter = iter(calibration_loader)
    with tqdm(total=cfg.calibration_samples, desc="Calibrating") as pbar:
        with torch.no_grad():
            while calibration_count < cfg.calibration_samples:
                try:
                    inputs, _ = next(data_iter)
                except StopIteration:
                    data_iter = iter(calibration_loader)
                    continue
                inputs = inputs.to(cfg.device)
                _ = model(inputs)
                calibration_count += inputs.size(0)

                pbar.update(inputs.size(0))

def _dynamic_quantization(
        cfg: QuantizationContextConfig,
        model
    ):
    """ Dynamic Quantization."""
    quantized_dtype = getattr(torch, cfg.dtype)

    return torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=quantized_dtype
    )

def _static_quantization(
        cfg: QuantizationContextConfig,
        model,
        calibration_loader
    ):
    """ Static Quantization. """
    model.qconfig = torch.quantization.get_default_qconfig(cfg.backend)
    model_prepared = torch.quantization.prepare(model, inplace=False)

    _calibrate_quantized_model(
        cfg=cfg,
        model=model_prepared,
        calibration_loader=calibration_loader
    )

    quantized_model = torch.quantization.convert(model_prepared, inplace=False)
    return quantized_model

def _custom_quantization(
        cfg: QuantizationContextConfig,
        model,
        calibration_loader,
        custom_qconfig: Optional[QConfig] = None
    ):
    """ Custom Quantization, need QConfig."""
    if custom_qconfig is None:
        print("Warning: No received QConfig, using default configuration (static quantization)...")
        model.qconfig = torch.quantization.get_default_qconfig(cfg.backend)
    else:
        model.qconfig = custom_qconfig

    model_prepared = torch.quantization.prepare(model, inplace=False)

    _calibrate_quantized_model(
        cfg=cfg,
        model=model_prepared,
        calibration_loader=calibration_loader
    )

    quantized_model = torch.quantization.convert(model_prepared, inplace=False)
    return quantized_model

def _fx_graph_quantization(
        cfg: QuantizationContextConfig,
        model,
        calibration_loader
    ):
    """
        Use FX Graph Mode to Quantize Model.
    """
    qconfig_mapping = get_default_qconfig_mapping(cfg.backend)

    example_inputs = next(iter(calibration_loader))[0][:1]
    # example_inputs = torch.randn(*cfg.input_size, device=cfg.device)

    model_prepared = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs)

    _calibrate_quantized_model(
        cfg,
        model=model_prepared,
        calibration_loader=calibration_loader
    )

    quantized_model = quantize_fx.convert_fx(model_prepared)
    return quantized_model
