"""
    Main script to fine-tune and quantize model.
"""
import hydra
from omegaconf import DictConfig, OmegaConf

import torch

from schemas.functions import (
    sync_runtime_status,
    create_data_context_config,
    create_model_context_config,
    create_training_context_config,
    create_quantization_context_config,
    create_exporting_context_config,
    create_evaluation_context_config
)
from dataloader import load_dataset
from model.utils import load_model, save_model
from model.modify import modify_model
from model.train import fine_tune_the_model
from quantization.quantize import quantize_model
from export.engine import auto_export
from analysis.complexity import analyze_model_complexity
from analysis.quantization_test import check_quantization_effectiveness
from analysis.benchmark import comprehensive_benchmark
from analysis.deployment import analyze_model_for_embedded_deployment
from debug import debug_model_parameters

def print_summary_report(results):
    """ Output brief summary result. """
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    print("Quantization Effectiveness: "
        f"{results['quantization_effectiveness']['size_reduction_ratio']:.1f}%"
        " size reduction"
    )
    print(f"Accuracy Drop: {results['benchmark_results']['accuracy_drop']:.2f}%")
    print(f"Speedup: {results['benchmark_results']['speedup_factor']:.2f}x")

@hydra.main(version_base=None, config_path="./hydra_configs", config_name="config")
def main(cfg: DictConfig):
    """ Full process flow. """
    # print(hydra.utils.instantiate(cfg), end="\n\n")
    cfg = sync_runtime_status(cfg)
    print(OmegaConf.to_yaml(cfg))

    data_config = create_data_context_config(cfg)
    train_dataloader, test_dataloader = load_dataset(data_config)

    model_config = create_model_context_config(cfg)
    original_model = load_model(model_config)
    modified_model = modify_model(model_config, original_model)

    print("=== Fine-tune the model ===")
    training_config = create_training_context_config(cfg)
    tuned_model = fine_tune_the_model(
        training_config,
        modified_model,
        train_dataloader,
    )

    # Free the memory.
    del original_model
    del modified_model
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    save_model(model_config, tuned_model, 'finetuned_mobilenetv3.pth')

    print("\n=== Quantize the model ===")
    quantization_config = create_quantization_context_config(cfg)
    quantized_model = quantize_model(quantization_config, tuned_model, test_dataloader)

    if cfg.run.verbose:
        debug_model_parameters(quantized_model, model_name="FX Graph Quantized Model")

    save_model(model_config, quantized_model, 'quantized_mobilenetv3.pth')

    print("\n=== Universal Analysis ===")
    orig_model_info = analyze_model_complexity(
        model=tuned_model,
        model_name="Original Model"
    )
    quant_model_info = analyze_model_complexity(
        model=quantized_model,
        model_name="Quantized Model"
    )
    quant_effectiveness = check_quantization_effectiveness(orig_model_info, quant_model_info)

    # Comprehensive Performance Analysis
    evaluation_config = create_evaluation_context_config(cfg)
    results = comprehensive_benchmark(
        cfg=evaluation_config,
        original_model=tuned_model,
        quantized_model=quantized_model,
        test_loader=test_dataloader
    )

    print("\n=== Deployment Analysis ===")
    deployment_analysis = analyze_model_for_embedded_deployment(
        model_info=quant_model_info,
        target_device="raspberry_pi"
    )

    # create_inference_demo(quantized_model, test_loader)

    print("\n=== Export the Model ===")
    exporting_config = create_exporting_context_config(cfg)
    auto_export(exporting_config, quantized_model)

    summary_result = {
        'quantization_effectiveness': quant_effectiveness,
        'benchmark_results': results,
        'deployment_analysis': deployment_analysis
    }
    print_summary_report(summary_result)

    return summary_result

if __name__ == "__main__":
    main()
