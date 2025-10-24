"""
    Complex Configuration Data Classes.
"""
from dataclasses import dataclass

from . import data_classes as _data_classes
# from .data_classes import (
#     DebugConfig,
#     DatasetConfig,
#     ModelConfig,
#     TrainingConfig,
#     QuantizationConfig
# )

__all__ = [
    "DatasetContextConfig", "ModelContextConfig",
    "TrainingContextConfig", "QuantizationContextConfig"
]


@dataclass
class DatasetContextConfig(_data_classes.DatasetConfig):
    """ Dataset Config Object in context. """
    seed: int = 42


@dataclass
class ModelContextConfig(_data_classes.ModelConfig):
    """ Model Config Object in context. """
    device: str = "auto"
    num_classes: int = 10
    model_path: str = ""
    save_dir: str = "./saved_models"


@dataclass
class TrainingContextConfig(_data_classes.TrainingConfig):
    """ Training Config Object in context """
    device: str = "cuda"


    # def validate_optimizer(self):
    #     """  Validate Optimizer Configuration. """
    #     if self.optimizer.lr <= 0:
    #         raise ValueError("Learning rate must be positive")

    #     for i, group in enumerate(self.optimizer.param_groups):
    #         if not group.param_names:
    #             raise ValueError(f"Param group {i} must have at least one param_name")
    #         if group.lr is not None and group.lr <= 0:
    #             raise ValueError(f"Learning rate in param group {i} must be positive")

    # def get_optimizer_params(self, model):
    #     """ Generate optimizer parameter group based on configuration. """
    #     param_groups = []

    #     default_params = [p for p in model.parameters()]
    #     if default_params:
    #         param_groups.append({
    #             'params': default_params,
    #             'lr': self.optimizer.lr,
    #             'weight_decay': self.optimizer.weight_decay,
    #         })

    #     for group_config in self.optimizer.param_groups:
    #         group_params = []
    #         for param_name in group_config.param_names:
    #             module = getattr(model, param_name, None)
    #             if module is not None:
    #                 group_params.extend(module.parameters())

    #         if group_params:
    #             group_dict = {'params': group_params}
    #             if group_config.lr is not None:
    #                 group_dict['lr'] = group_config.lr

    #             if group_config.weight_decay is not None:
    #                 group_dict['weight_decay'] = group_config.weight_decay

    #             if group_config.betas is not None:
    #                 group_dict['betas'] = group_config.betas

    #             if group_config.eps is not None:
    #                 group_dict['eps'] = group_config.eps

    #             param_groups.append(group_dict)

    #     return param_groups


@dataclass
class QuantizationContextConfig(_data_classes.QuantizationConfig):
    """ Quantization Config Object in context. """
    device: str = "cpu"


# @dataclass
# class ExportingContextConfig(ExportingConfig):
#     """ Exporting Context Object in context. """
#     opset_version: int = 16
#     do_constant_folding: bool = True
#     export_params: bool = True
#     training: bool = False

#     dynamic_axes: dict = field(default_factory=lambda: {
#         'input': {0: 'batch_size'},
#         'output': {0: 'batch_size'}
#     })

#     operator_export_type: str = "ONNX"  # ONNX, RAW, ONNX_ATEN
#     custom_opsets: dict = field(default_factory=dict)
#     export_modules_as_functions: bool = False

#     metadata: dict = field(default_factory=lambda: {
#         "producer_name": "torch.onnx",
#         "producer_version": "1.0",
#     })


@dataclass
class EvaluationContextConfig(_data_classes.EvaluationConfig):
    """ Evaluation Config Object in context. """
