"""
    Definitions of QConfig presets.
"""

import torch

from enumeration import QConfigPreset

PRESET_CONFIGS = {
    QConfigPreset.PRESET_HIGH_COMPRESSION: torch.quantization.QConfig(
            activation=torch.quantization.MovingAverageMinMaxObserver.with_args(
                dtype=torch.quint8, averaging_constant=0.9
            ),
            weight=torch.quantization.default_per_channel_weight_observer
        ),
    QConfigPreset.PRESET_HIGH_ACCURACY: torch.quantization.QConfig(
            activation=torch.quantization.HistogramObserver.with_args(
                dtype=torch.quint8, reduce_range=False
            ),
            weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8)
        ),
    QConfigPreset.PRESET_MOBILE: torch.quantization.QConfig(
            activation=torch.quantization.MovingAverageMinMaxObserver.with_args(
                dtype=torch.quint8, averaging_constant=0.95
            ),
            weight=torch.quantization.default_per_channel_weight_observer
        )
}
