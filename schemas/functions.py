"""
    Definitions of functions completing the configuration aggregation.
"""
import torch
from omegaconf import DictConfig

from .context_classes import (
    DatasetContextConfig,
    ModelContextConfig,
    TrainingContextConfig,
    QuantizationContextConfig,
    EvaluationContextConfig
)

def _convert_device(device: str):
    _support_device_list = ["auto", "cpu", "gpu", "cuda"]
    # cpu, cuda, ipu, xpu,
    # mkldnn, opengl, opencl,
    # ideep, hip, ve, fpga,
    # maia, xla, lazy, vulkan,
    # mps, meta, hpu, mtia, privateuseone

    if device not in _support_device_list:
        raise ValueError(
            f"Unknown device: {device}, supported devices are {_support_device_list}")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "gpu":
        device = "cuda"
    else:
        pass

    return device

def _validate_seed(seed: any):
    if not hasattr(seed, '__int__'):
        if isinstance(seed, str) and not seed.strip().lstrip('+-').isdigit():
            raise ValueError(
                f"Seed string '{seed}' must contain only digits and optional sign.")
        raise ValueError(
            f"Seed value {seed} of type {type(seed).__name__} cannot be converted to integer.")
    return seed

# def create_global_config(cfg: DictConfig):
#     """ Create context config object of global config. """
#     device = _convert_device(cfg.globals.device)
#     seed = _validate_seed(cfg.globals.seed)

#     return GlobalContextConfig(
#         device=device,
#         seed=seed
#     )

def sync_runtime_status(cfg: DictConfig) -> DictConfig:
    """ Create context config object of global status. """
    actual_device = _convert_device(cfg.globals.device)

    cfg.globals.device = actual_device
    # cfg.evaluation.device = actual_device

    if cfg.run.debug:
        cfg.globals.device = _convert_device(cfg.run.relative.device)
        cfg.data.batch_size = cfg.run.relative.batch_size
        cfg.data.subset_ratio = cfg.run.relative.subset_ratio
        cfg.training.num_epochs = cfg.run.relative.num_epochs
        cfg.quantization.calibration_samples = cfg.run.relative.calibration_samples
        cfg.evaluation.num_runs = cfg.run.relative.num_runs
        cfg.evaluation.num_batches = cfg.run.relative.num_batches
    return cfg

def create_data_context_config(cfg: DictConfig) -> DatasetContextConfig:
    """ Create context config obejct of data. """
    return DatasetContextConfig(
        name=cfg.data.name,
        # num_classes=cfg.data.num_classes,
        subset_ratio=cfg.data.subset_ratio,
        root_dir=cfg.data.root_dir,
        transform=cfg.data.transform,
        batch_size=cfg.data.batch_size,
        seed=cfg.globals.seed,
    )

def create_model_context_config(cfg: DictConfig) -> ModelContextConfig:
    """ Create context config obejct of model. """
    return ModelContextConfig(
        device=cfg.globals.device,
        num_classes=cfg.data.num_classes,
        model_path=cfg.globals.model_dir,
        save_dir=cfg.globals.save_dir,
        name=cfg.model.name,
        pretrained=cfg.model.pretrained
    )

def create_training_context_config(cfg: DictConfig) -> TrainingContextConfig:
    """ Create context config obejct of training. """
    return TrainingContextConfig(
        num_epochs=cfg.training.num_epochs,
        device=cfg.globals.device,
        optimizer=cfg.training.optimizer,
        scheduler=cfg.training.scheduler
    )

def create_quantization_context_config(cfg: DictConfig) -> QuantizationContextConfig:
    """ Create context config obejct of quantization. """
    return QuantizationContextConfig(
        strategy=cfg.quantization.strategy,
        device=cfg.globals.device,
        backend=cfg.quantization.backend,
        calibration_samples=cfg.quantization.calibration_samples,
        dtype=cfg.quantization.dtype
    )

def create_exporting_context_config(cfg: DictConfig) -> DictConfig:
    """
        Create an export context configuration to flatten
        and merge common configuration and engine-specific configuration.
    """
    common_config = {k: v for k, v in cfg.export.items() if k != 'engine'}
    engine_config = dict(cfg.export.engine)

    engine_name = engine_config.pop("name", None)
    merged_config = {**common_config, **engine_config}

    merged_config['engine_name'] = engine_name or cfg.export.engine.name

    return DictConfig(merged_config)

def create_evaluation_context_config(cfg: DictConfig) -> EvaluationContextConfig:
    """ Create context config obejct of evaluation. """
    return EvaluationContextConfig(
        device=cfg.evaluation.device,
        num_runs=cfg.evaluation.num_runs,
        num_batches=cfg.evaluation.num_batches
    )
