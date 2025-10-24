"""
    Model Benchmarks.
"""
import time
from tqdm import tqdm

import torch

from schemas.context_classes import EvaluationContextConfig
from .quantization_test import validate_quantization
from .complexity import compare_model_size
from .utils import format_performance_metrics

def benchmark_inference_speed(
        cfg: EvaluationContextConfig,
        model,
        test_loader,
    ):
    """ A more comprehensive benchmark, averaging over multiple random batches.W """
    model.eval()
    model = model.to(cfg.device)

    # Warm up to avoid the influence caused by the cost of initialization
    with torch.no_grad():
        for inputs, _ in list(test_loader)[:2]:
            inputs = inputs.to(cfg.device)
            _ = model(inputs)

    # TODO: Unsure whether compile model or not.
    # if hasattr(torch, 'compile'):
    #     model = torch.compile(model, mode="reduce-overhead")

    batch_times = []

    with torch.no_grad():
        for __run in range(cfg.num_runs):
            for batch_idx, (inputs, __labels) in enumerate(test_loader):
                if batch_idx >= cfg.num_batches:
                    break
                inputs = inputs.to(cfg.device)
                start_time = time.time()
                _ = model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()

                batch_time = end_time - start_time
                batch_times.append(batch_time)

    avg_time = sum(batch_times) / len(batch_times)
    std_time = torch.tensor(batch_times).std().item()

    print(f"Tested {len(batch_times)} batches.")
    print(f"Average batch inference time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"Min/Max Time: {min(batch_times)*1000:.2f}/{max(batch_times)*1000:.2f} ms")

    return {
        'avg_batch_time': avg_time,
        'std_batch_time': std_time,
        'min_time': min(batch_times),
        'max_time': max(batch_times),
        'all_times': batch_times,
        'samples_per_second': cfg.num_batches / avg_time
    }

def evaluate_model(model, test_loader, device="cpu"):
    """ Define model evaluation method. """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating accuracy...'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100. * correct / total
    model.train()
    return accuracy

def comprehensive_benchmark(
        cfg: EvaluationContextConfig,
        original_model,
        quantized_model,
        test_loader
    ):
    """ Comprehensive performance analysis. """
    print("=== Comprehensive performance analysis ===")

    print("\n1. Accuracy Comparison:")
    orig_acc, quant_acc, acc_drop = validate_quantization(
        original_model, quantized_model, test_loader)

    print("\n2. Size Comparison:")
    __orig_size, __quant_size, compression = compare_model_size(
        original_model, quantized_model, "MobileNet V3")

    print("\n3. Inference Speed Comparison:")
    print("Original Model:")
    orig_speed = benchmark_inference_speed(cfg, original_model, test_loader)
    print("\nQuantized Model:")
    quant_speed = benchmark_inference_speed(cfg, quantized_model, test_loader)

    speedup = quant_speed['samples_per_second'] / orig_speed['samples_per_second']
    print("\n4. Performance Summary:")
    # print(f"Increased Speed: {speedup:.5f}x")
    # print(f"Decreased Size: {compression:.5f}%")
    # print(f"Decreased Accuracy: {acc_drop:+.5f}%")
    print(f"Increased Speed: {format_performance_metrics(speedup, 'x')}")
    print(f"Decreased Size: {format_performance_metrics(compression, '%')}")
    print(f"Decreased Accuracy: {format_performance_metrics(acc_drop, '%')}")

    return {
        'accuracy_drop': acc_drop,
        'compression_ratio': compression,
        'speedup_factor': speedup,
        'original_accuracy': orig_acc,
        'quantized_accuracy': quant_acc
    }

# def create_inference_demo(quantized_model, test_loader):
#     """ Simple inference demo. """
#     quantized_model.eval()
#     quantized_model = quantized_model.to("cpu")

#     print("\n=== Inference Demo ===")
#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(test_loader):
#             if i >= 3:
#                 break

#             start_time = time.time()
#             outputs = quantized_model(inputs)
#             inference_time = time.time() - start_time

#             _, predicted = torch.max(outputs, 1)
#             accuracy = (predicted == labels).float().mean()

#             print(
#                 f"Sample {i+1}: "
#                 f"Inference Time {inference_time*1000:.2f}ms, "
#                 f"Accuracy: {accuracy.item():.2f}"
#             )
