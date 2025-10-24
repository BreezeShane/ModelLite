"""
    Model Utilities.
"""
from os import makedirs
from os.path import join, exists, isdir

import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from schemas.context_classes import ModelContextConfig

def load_model(cfg: ModelContextConfig):
    """ Load MobileNet V3 pretrained model. """
    model = None

    if cfg.model_path:
        model = torch.load(cfg.model_path)
    else:
        if cfg.name == "mobilenet_v3_small":
            model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        # TODO: Add more models to support.
        # elif cfg.name == "":
        #     model =
        # else:
        #     pass
    if model:
        return model.to(cfg.device)
    raise RuntimeError("Failed to load model.")

def save_model(cfg: ModelContextConfig, model, file_name_with_suffix: str):
    """ Save the model. """
    if not (exists(cfg.save_dir) and isdir(cfg.save_dir)):
        makedirs(cfg.save_dir)
    save_path = join(cfg.save_dir, file_name_with_suffix)
    torch.save(model.state_dict(), save_path)
    print(f"Success: Model '{file_name_with_suffix}' has been saved!")
