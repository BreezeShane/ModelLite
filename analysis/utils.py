"""
    Model analysis and evaluation definitions.
"""

def format_performance_metrics(value, unit=""):
    """ Format performance indicators and select appropriate precision based on the value size. """
    abs_value = abs(value)
    if abs_value < 0.001:
        return f"{value:.6f}{unit}"
    elif abs_value < 0.01:
        return f"{value:.4f}{unit}"
    elif abs_value < 0.1:
        return f"{value:.3f}{unit}"
    else:
        return f"{value:.2f}{unit}"
