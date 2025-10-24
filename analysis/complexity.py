"""
    Analyze the model complexity.
"""
import torch

def get_model_params_and_size(model):
    """ Count the number of parameters and compute model size. """
    state_dict = model.state_dict()
    total_params = 0
    total_bytes = 0
    for tensor in state_dict.values():
        if isinstance(tensor, torch.Tensor):
            total_params += tensor.nelement()
            total_bytes += tensor.nelement() * tensor.element_size()

    # # More accurate method.
    # buffer = BytesIO()
    # torch.save(state_dict, buffer)
    # total_bytes = len(buffer.getvalue())

    return total_params, total_bytes / 1024**2

def analyze_model_complexity(model, model_name="Model"):
    """
    Model Complexity Analysis.
    """

    total_params, current_size = get_model_params_and_size(model)

    fp32_size = total_params * 4 / (1024**2)  # Assume all are FP32
    int8_size = total_params * 1 / (1024**2)  # Assume all are INT8

    model_info = {
        'model_name': model_name,
        'total_parameters': total_params,
        'current_size_mb': current_size,
        'fp32_theoretical_size_mb': fp32_size,
        'int8_theoretical_size_mb': int8_size,
        'compression_ratio': (1 - current_size/fp32_size) * 100 if fp32_size > 0 else 0,
        'parameters_in_millions': total_params / 1e6 # M
    }

    return model_info

def compare_model_size(original_model, quantized_model, model_name="model"):
    """ Compare the sizes of models. """
    _, original_size = get_model_params_and_size(original_model)
    _, quantized_size = get_model_params_and_size(quantized_model)
    compression_ratio = (1 - quantized_size/original_size) * 100 if original_size > 0 else 0

    size_format = "{:.6f}" if original_size < 0.1 or quantized_size < 0.1 else "{:.2f}"
    ratio_format = "{:.6f}" if abs(compression_ratio) < 0.1 else "{:.2f}"

    print(f"\n=== {model_name} Size Comparison ===")
    print(f"Size of original model: {size_format.format(original_size)} MB")
    print(f"Size of quantized model: {size_format.format(quantized_size)} MB")
    print(f"Compression ratio: {ratio_format.format(compression_ratio)}%")

    return original_size, quantized_size, compression_ratio

def print_model_complexity(model_info):
    """ Output Model Complexity. """
    print(f"\n=== {model_info['model_name']} Complexity Analysis ===")
    print(f"Total parameters: {model_info['total_parameters']:,} "
          f"({model_info['parameters_in_millions']:.6f}M)")

    current_size = model_info['current_size_mb']
    fp32_size = model_info['fp32_theoretical_size_mb']
    int8_size = model_info['int8_theoretical_size_mb']

    size_format = "{:.6f}" if current_size < 0.01 else "{:.2f}"

    print(f"Current model size: {size_format.format(current_size)} MB")
    print(f"Theoretical sizes: {size_format.format(fp32_size)} MB (FP32)"
           " / "
           f"{size_format.format(int8_size)} MB (INT8)")

    if model_info['fp32_theoretical_size_mb'] > 0:
        print(f"Compression ratio: {model_info['compression_ratio']:.6f}%")
