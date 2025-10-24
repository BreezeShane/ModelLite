"""
    Definitions of modifying models.
"""

import torch
from schemas.context_classes import ModelContextConfig

def modify_model(cfg: ModelContextConfig, model):
    """ Modify the classifier of model. """
    # Modify the classifier
    model.classifier[3] = torch.nn.Linear(1024, cfg.num_classes)
    model = model.to(cfg.device)
    return model
