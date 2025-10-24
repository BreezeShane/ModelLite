"""
    Optimizer Configuration Presets.
"""
OPTIMIZER_PRESETS = {
    "adam": {
        "name": "adam",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "betas": [0.9, 0.999]
    },
    "adamw": {
        "name": "adamw",
        "lr": 1e-3,
        "weight_decay": 1e-2
    }
}

SCHEDULER_PRESETS = {
    "cosine": {
        "name": "cosine",
        "T_max": "auto",
        "eta_min": 1e-6
    },
    "step": {
        "name": "step",
        "step_size": 30,
        "gamma": 0.1
    }
}

def apply_preset(config_dict, preset_type):
    """ Apply configuration preset. """
    presets = OPTIMIZER_PRESETS if preset_type == "optimizer" else SCHEDULER_PRESETS
    preset_name = config_dict.get("name")

    if preset_name in presets:
        base_config = presets[preset_name].copy()
        base_config.update(config_dict)
        return base_config
    return config_dict
