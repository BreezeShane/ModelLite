"""
    Exporter Definitions of Various Engines.
"""
from os.path import join

import torch
from omegaconf import DictConfig, OmegaConf

from .decorator import register_exporter, get_exporter

__all__ = [
    "auto_export"
]

def auto_export(export_config: DictConfig, model):
    """
        Unified model export function,
        automatically select the corresponding exporter according to the configuration.
    """
    exporter = get_exporter(export_config.engine_name)

    return exporter(export_config, model)

@register_exporter("onnx")
def export_to_onnx(cfg: DictConfig, model):
    """
        Export to ONNX Model.
        Not recommended to export (Eager Mode) quantized model
        because some operators are unsupported by ONNX.
    """
    model.eval()
    model.to(cfg.device)
    dummy_input = torch.randn(*cfg.input_shape, device=cfg.device)

    save_path = join(cfg.save_dir, cfg.file_name)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=cfg.export_params,
        opset_version=cfg.opset_version,
        do_constant_folding=cfg.do_constant_folding,
        input_names=cfg.input_names,
        output_names=cfg.output_names,
        dynamic_axes=OmegaConf.to_object(cfg.dynamic_axes)
    )
    print(f"Success: ONNX model has been exported to '{save_path}'.")
    return save_path

@register_exporter("tensorrt")
def export_to_tensorrt(cfg: DictConfig, model):
    """ Export to TensorRT. """
    raise NotImplementedError
    # print(f"TensorRT Model Exported: {cfg.output_path}")
    # return cfg.output_path

@register_exporter("tvm")
def export_to_tvm(cfg: DictConfig, model):
    """ Export to TVM. """
    raise NotImplementedError
    # print(f"TVM Model Exported: {cfg.output_path}")
    # return cfg.output_path
