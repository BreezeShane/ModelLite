"""
    Global Enumeration.
"""
from enum import Enum


class QuantizationStrategy(Enum):
    """ Quantization strategies """
    FX_GRAPH = "fx_graph"
    STATIC = "static"
    DYNAMIC = "dynamic"
    CUSTOM = "custom"

    @classmethod
    def from_config(cls, strategy_str: str):
        """ Convert string to enumeration safely. """
        try:
            return cls(strategy_str)
        except ValueError:
            print(f"Unknown quantization strategy: {strategy_str}, using FX Graph as default...")
            return cls.FX_GRAPH


class QConfigPreset(Enum):
    """ QConfig Presets. """
    PRESET_HIGH_COMPRESSION = "high_compression"
    PRESET_HIGH_ACCURACY = "high_accuracy"
    PRESET_MOBILE = "mobile"
