"""
    Analyze whether the model able to be run on devices.
"""
def analyze_model_for_embedded_deployment(
        model_info: dict,
        target_device="raspberry_pi"
    ):
    """
    Comprehensive analysis of the model's deployment suitability,
    including basic metrics and specific device compatibility.
    """
    print(f"\n=== {model_info["model_name"]} Comprehensive Analysis of Embedded Deployment ===")
    device_profiles = {
        "raspberry_pi": {
            "max_memory_mb": 1000,
            "typical_power_w": 3,
            "description": "Raspberry Pi"
        },
        "jetson_nano": {
            "max_memory_mb": 4000,
            "typical_power_w": 10,
            "description": "Jetson Nano"
        },
        "arduino": {
            "max_memory_mb": 256,
            "typical_power_w": 1,
            "description": "Arduino"
        }
    }
    profile = device_profiles.get(target_device, device_profiles["raspberry_pi"])

    # print_model_complexity(model_info)

    memory_usage = model_info['current_size_mb']
    memory_ratio = memory_usage / profile['max_memory_mb']

    if memory_ratio < 0.3:
        memory_status = "Excellent"
    elif memory_ratio < 0.6:
        memory_status = "Acceptable"
    else:
        memory_status = "Difficult"

    suggestions = generate_deployment_suggestions(model_info, memory_ratio, target_device)
    print("\nDeployment Suggestions:")
    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
    else:
        print("  The model is well-suited for deployment on this device.")

    analysis_result = {
        **model_info,
        'target_device': target_device,
        'memory_usage_mb': memory_usage,
        'memory_usage_ratio': memory_ratio,
        'compatibility': memory_status,
        'suggestions': suggestions
    }

    return analysis_result

def generate_deployment_suggestions(model_info, memory_ratio, target_device):
    """ Generate deployment suggestions. """
    suggestions = []

    if memory_ratio > 0.7:
        suggestions.append(
            "Memory usage is high - consider model pruning or choosing a more powerful device.")
    elif memory_ratio > 0.5:
        suggestions.append(
            "Memory usage is moderate - model pruning could further optimize deployment.")

    if model_info['total_parameters'] > 10e6:
        suggestions.append(
            "Model is large - consider knowledge distillation to train a smaller student model.")
    elif model_info['total_parameters'] > 5e6 and target_device == "raspberry_pi":
        suggestions.append(
            "Model is moderately large for Raspberry Pi - monitor inference speed.")

    if model_info['compression_ratio'] < 50:
        suggestions.append(
            "Quantization effect is limited - check if quantization was applied correctly.")
    elif model_info['compression_ratio'] > 70:
        suggestions.append(
            "Good quantization effect - model is well optimized for embedded deployment.")

    if target_device == "arduino":
        if model_info['current_size_mb'] > 100:
            suggestions.append(
                "Model too large for Arduino, consider extreme quantization or model partitioning.")

    return suggestions
