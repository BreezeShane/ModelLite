"""
    Test the effect of quantization.
"""
from time import time

import torch

from .complexity import print_model_complexity

def test_fx_quantized_model(quantized_model, test_loader):
    """ Test FX Graph quantization. """
    print("\n=== Testing FX Quantized Model ===")

    if not detect_quantization(quantized_model):
        print("Warning: No quantitative features detected.")
        print("Possible reasons:")
        print("  - Quantification has not really taken effect.")
        print("  - Quantization is implemented in a special way and is difficult to detect.")
        print("  - Model structure is not compatible with quantization.")

    quantized_model.eval()
    quantized_model = quantized_model.to('cpu')

    try:
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                if i >= 2:
                    break
                inputs, labels = inputs.to('cpu'), labels.to('cpu')

                start_time = time.time()
                outputs = quantized_model(inputs)
                inference_time = time.time() - start_time

                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == labels).float().mean()

                print(f"Batch {i+1}: Inference Time {inference_time*1000:.2f}ms, Accuracy: {accuracy.item():.2f}")

        print("✓ FX quantized model inference successful!")
        return True
    except RuntimeError as e:
        print(f"✗ FX quantized model inference failed: {e}")
        return False

def detect_quantization(model) -> bool:
    """ Detect if quantized.  """
    return any(
        'scale' in key or
        'zero_point' in key or
        '_packed_params' in key
        for key in model.state_dict().keys()
    )
    # quant_detected = False
    # for key in state_dict.keys():
    #     print(key)
    #     if 'scale' in key or 'zero_point' in key or '_packed_params' in key:
    #         quant_detected = True
    #         break
    # return quant_detected

def check_quantization_effectiveness(orig_info, quant_info):
    """ Check quantization effectiveness. """
    print("=== Check Quantization Effectiveness ===")
    print_model_complexity(orig_info)
    print_model_complexity(quant_info)

    print("\n=== Quantization Results ===")
    size_reduction = orig_info['current_size_mb'] - quant_info['current_size_mb']
    size_reduction_ratio = (size_reduction / orig_info['current_size_mb']) * 100

    print(f"Size reduction: {size_reduction:.2f} MB ({size_reduction_ratio:.2f}%)")
    print(f"Actual compression vs theoretical: {quant_info['compression_ratio']:.2f}%")

    # quant_layers_existence = check_quantization_status(quantized_model)
    return {
        'original_info': orig_info,
        'quantized_info': quant_info,
        'size_reduction_mb': size_reduction,
        'size_reduction_ratio': size_reduction_ratio,
        # 'quantization_detected': quant_layers_existence > 0
    }

# def check_quantization_status(model):
#     """ Check which layers in model are quantized. """
#     print("\n=== Quantization Status Check ===")

#     quantized_layers = 0
#     total_layers = 0

#     for name, module in model.named_modules():
#         total_layers += 1
#         if isinstance(module, (torch.ao.nn.quantized.Conv2d,
#                              torch.ao.nn.quantized.Linear,
#                              torch.ao.nn.quantized.PReLU)):
#             quantized_layers += 1
#             print(f"✓ Quantized layer: {name} - {type(module).__name__}")
#         elif hasattr(module, 'qconfig') and module.qconfig is not None:
#             print(f"○ Layer with qconfig: {name} - {type(module).__name__}")

#     print(f"\nQuantized layers: {quantized_layers}/{total_layers}")

#     return quantized_layers > 0

def validate_quantization(original_model, quantized_model, test_loader):
    """ Verify the change in model accuracy before and after quantization. """
    original_model.eval()
    quantized_model.eval()

    original_model = original_model.to('cpu')
    quantized_model = quantized_model.to('cpu')

    original_correct = 0
    quantized_correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to('cpu'), labels.to('cpu')
            orig_outputs = original_model(inputs)
            _, orig_predicted = torch.max(orig_outputs.data, 1)

            quant_outputs = quantized_model(inputs)
            _, quant_predicted = torch.max(quant_outputs.data, 1)

            total += labels.size(0)
            original_correct += (orig_predicted == labels).sum().item()
            quantized_correct += (quant_predicted == labels).sum().item()

    original_acc = 100. * original_correct / total
    quantized_acc = 100. * quantized_correct / total
    accuracy_drop = original_acc - quantized_acc

    print(f"Accuracy of original model: {original_acc:.2f}%")
    print(f"Accuracy of quantized model: {quantized_acc:.2f}%")
    print(f"Decreased accuracy: {accuracy_drop:.2f}%")

    return original_acc, quantized_acc, accuracy_drop
