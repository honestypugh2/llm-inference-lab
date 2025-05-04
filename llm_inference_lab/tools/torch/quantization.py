"""
This module and its functions are intended to be used with HuggingFace transformers model
for quantization. Dynamic quantization in PyTorch converts a float point
model to a quantized model with int8 or float16 for the weights and activations.
"""
import torch


####################################
# Helpers for dynamic quantization #
####################################
def dynamic_quantization(model: torch.nn.Module) -> torch.nn.Module:
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return model
