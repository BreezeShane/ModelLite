"""
    Model debugging definitions.
"""

import torch

def verify_model_copy(original, copied):
    """ Verify whether models are different. """
    assert str(original) == str(copied), "Error(Deep Copy): Different Model Structures."

    orig_params, copy_params = original.named_parameters(), copied.named_parameters()

    for (name1, param1), (name2, param2) in zip(orig_params, copy_params):
        assert name1 == name2, f"Error(Deep Copy): Different Parameter Names: {name1} vs {name2}"
        assert torch.allclose(param1, param2, atol=1e-6), f"Different Values: {name1}"

    print("Success: Deep copy works correctly.")


def debug_model_parameters(model, model_name="Model"):
    """ Debug function: print model parameter information. """
    print(f"\n=== {model_name} Parameter Debug ===")
    total_params = 0

    for name, param in model.named_parameters():
        print(f"Parameter: {name}")
        print(f"  Shape: {param.shape}")
        print(f"  Num elements: {param.nelement()}")
        print(f"  Data type: {param.dtype}")
        print(f"  Has _packed_params: {hasattr(param, '_packed_params')}")
        print(f"  Has int_repr: {hasattr(param, 'int_repr')}")
        print(f"  Has q_scale: {hasattr(param, 'q_scale')}")

        total_params += param.nelement()
        print("-" * 50)

    print(f"Total parameters from named_parameters(): {total_params}")
    return total_params
