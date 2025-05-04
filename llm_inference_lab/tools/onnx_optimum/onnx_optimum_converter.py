"""
This module and its functions are intended to be used with ONNX Runtime and
HuggingFace transformers models that are supported by Optimum ONNX.
Supported model architectures can be found
at https://huggingface.co/docs/optimum/exporters/onnx/overview.
Dynamic quantization converts a float point
model to a quantized model with int8 or float16 for the weights and activations.
"""

from optimum.onnxruntime import ORTQuantizer, ORTModelForSeq2SeqLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig

########################################
# Helpers for exporting to ONNX format #
########################################
def export_to_seq2seq_onnx(model_location: str, onnx_model_location: str) -> None:
    onnx_model = ORTModelForSeq2SeqLM.from_pretrained(model_location, export=True)
    onnx_model.save_pretrained(onnx_model_location)


###################################
# Helpers for loading ONNX models #
###################################
def load_seq2seq_onnx_model(onnx_model_location: str):
    onnx_model = ORTModelForSeq2SeqLM.from_pretrained(onnx_model_location, export=False)
    return onnx_model


####################################
# Helpers for dynamic quantization #
####################################
def export_to_seq2seq_onnx_quantized(
    onnx_model_location: str, quantized_onnx_model_location: str
) -> None:
    """To quantize sequence2sequence models, you have to quantize each model's
    component individually. Currently the ORTQuantizer class do not support
    multi-file models such as the sequence2sequence models.
    This function requires the model location of the ONNX model
    (ORTModelForSeq2SeqLM) and uses the ONNX model to create the quantized components,
    encoder, decoder and decoder with past key values.
    Each of the quantized components are placed in a list.
    Then the quantization strategy is defined
    by creating the configuration using AutoQuantizationConfig.
    The quantized model components are saved
    to the quantized ONNX model location.

    Args:
        onnx_model_location (str): The model location of the ONNX model
        quantized_onnx_model_location (str): The model location of the quantized ONNX model

    """

    encoder_quantizer = ORTQuantizer.from_pretrained(
        onnx_model_location, file_name="encoder_model.onnx"
    )

    decoder_quantizer = ORTQuantizer.from_pretrained(
        onnx_model_location, file_name="decoder_model.onnx"
    )

    decoder_wp_quantizer = ORTQuantizer.from_pretrained(
        onnx_model_location, file_name="decoder_with_past_model.onnx"
    )

    quantizer = [encoder_quantizer, decoder_quantizer, decoder_wp_quantizer]

    dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

    for quant in quantizer:
        quant.quantize(
            save_dir=quantized_onnx_model_location, quantization_config=dqconfig
        )
