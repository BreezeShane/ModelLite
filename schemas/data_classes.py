"""
    Data Class Definitions.
"""
from dataclasses import dataclass, field
from typing import Optional

# @dataclass
# class DebugConfig:
#     """ Debugging Configuration. """
#     debug: bool = True
#     quantization_cpu_only: bool = True

#     # Relative global data
#     device: str = "cpu"
#     num_epochs: int = 1
#     batch_size: int = 64


@dataclass
class DatasetConfig:
    """ Dataset Configuration. """
    name: str = "CIFAR10"
    # num_classes: int = 10
    subset_ratio: float = 1
    root_dir: str = "./Dataset"
    batch_size: int = 64
    transform: dict = field(default_factory={
        "resize": 224,
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    })


@dataclass
class ModelConfig:
    """ Model Configuration. """
    name: str = "mobilenet_v3_small"
    pretrained: bool = True


###
### Training Sub Config
###

@dataclass
class ParamGroupConfig:
    """ Parameter group configuration. """
    param_names: list[str] = field(default_factory=list)
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    betas: Optional[list[float]] = None
    eps: Optional[float] = None

@dataclass
class OptimizerConfig:
    """ Optimizer Configuration. """
    name: str
    # _target_: str = "torch.optim.Adam"
    lr: float = 0.001
    weight_decay: float = 0.0001
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1.0e-08
    param_groups: list[ParamGroupConfig] = field(default_factory=list)


@dataclass
class SchedulerConfig:
    """ Scheduler Configuration. """
    name: str
    required_params: dict[str, any] = field(default_factory=dict)
    optional_params: dict[str, any] = field(default_factory=dict)


###
###
###

@dataclass
class TrainingConfig:
    """ Training Configuration. """
    num_epochs: int = 1
    log_dir: str = "./logs"
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig("adam"))
    scheduler: Optional[SchedulerConfig] = None

@dataclass
class QuantizationConfig:
    """ Quantization Config. """
    strategy: str = "fx_graph"
    backend: str = "fbgemm"
    calibration_samples: int = 200
    dtype: str = "qint8"


# @dataclass
# class ExportingConfig:
#     """ Exporting Configuration. """
#     # [batch_size, channels, height, width]
#     input_shape: list[int] = field(default_factory=lambda: [1, 3, 224, 224])
#     input_dtype: str = "float32"
#     input_names: list[str] = field(default_factory=lambda: ["input"])
#     output_names: list[str] = field(default_factory=lambda: ["output"])
#     dynamic_batch: bool = True

#     fuse_layers: bool = True
#     constant_folding: bool = True
#     remove_identity_ops: bool = True
#     compute_precision: str = "fp32" # fp32, fp16, int8, mixed


# @dataclass
# class GlobalConfig:
#     """ Global Configuration. """
#     device: str = "auto" # cpu or cuda
#     seed: int = 42


@dataclass
class EvaluationConfig:
    """ Evaluation Configuration. """
    device: str = "cpu"
    num_runs: int = 50
    num_batches: int = 32
