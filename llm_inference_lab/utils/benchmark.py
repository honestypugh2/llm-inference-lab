"""
This module and its functions are intended to be used to run benchmarks for text summarization.
Defaults to Pegasus model architecture. To benchmarkk DistilBart, add the import
statement from minted.tools.infra.constants import distilbart_model.
"""
import os
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PegasusForConditionalGeneration,
    BartForConditionalGeneration,
    PegasusTokenizerFast,
    BartTokenizerFast,
    BatchEncoding,
)
from optimum.onnxruntime import ORTModelForSeq2SeqLM

from .models import pegasus_model

def measure_inference_time(model: PegasusForConditionalGeneration
    | BartForConditionalGeneration, 
    tokenizer: PegasusTokenizerFast | BartTokenizerFast,
    inputs: BatchEncoding
   ) -> float:
        # Record start time
        start_time = time.time()
        
        # Run inference
        outputs = model.generate(
        inputs["input_ids"]
        )

        _ = tokenizer.batch_decode(outputs,skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # Record end time and calculate latency
        end_time = time.time()
        latency = (end_time - start_time) 
        return latency


def measure_inference_latency(
    model: PegasusForConditionalGeneration
    | BartForConditionalGeneration
    | ORTModelForSeq2SeqLM,
    inputs: BatchEncoding,
    use_gpu: bool = True,
    num_samples: int = 100,
    num_warmups: int = 10,
) -> float:
    """
    Takes a model, the input_ids extracted from a BatchEncoding, a boolean to use_gpu,
    the num_samples to get average inference time per sequence and num_warmups.
    This is intended to provide the inference latency per sequence.
    """
    with torch.inference_mode():
        for _ in range(num_warmups):
            _ = model.generate(
                inputs.get("input_ids")
            )
    if use_gpu:
        torch.cuda.synchronize()

    with torch.inference_mode():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model.generate(
                inputs.get("input_ids")
            )
            if use_gpu:
                torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave


def get_model_size(model, temp_dir: str = "/llm-inference-lab") -> int:
    """
    Provides the model size of a torch-based model.
    """
    model_dir = os.path.join(temp_dir, "temp")
    torch.save(model.state_dict(), model_dir)
    size = os.path.getsize(model_dir)
    os.remove(model_dir)

    return size


def get_onnx_seq2seq_model(
    model_name: str = pegasus_model.path,
    provider: str = "CPUExecutionProvider",
):
    """
    Takes a saved ONNX model location, to include a quantized ONNX model,
    and the execution provider for the ONNX model. When use_gpu = True,
    provider must be set to CUDAExecutionProvider. Quantized models can not run
    on CUDAExecutionProvider.
    """
    model = ORTModelForSeq2SeqLM.from_pretrained(
        model_name,
        export=False,
        provider=provider,
    )

    return model


def get_seq2seq_model(model_name: str = pegasus_model.path):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return model, tokenizer


def dynamic_quantization(model: torch.nn.Module):
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return model


def prepare_seq2seq_inputs(
    text: str,
    tokenizer: PegasusTokenizerFast | BartTokenizerFast,
    padding: bool = False,
    device: str = "cuda:0",
) -> BatchEncoding:

    inputs = tokenizer(
        text, return_tensors="pt", padding=padding, truncation=True
    )
    if torch.cuda.is_available() and device != "cpu":
        inputs = inputs.to(device)

    return inputs


def move_inputs_to_device(
    inputs: BatchEncoding, device: str = "cuda:0"
) -> BatchEncoding:

    inputs_cuda = inputs.to(device)
    return inputs_cuda


def run_seq2seq(
    model: PegasusForConditionalGeneration
    | BartForConditionalGeneration
    | ORTModelForSeq2SeqLM,
    tokenizer: PegasusTokenizerFast | BartTokenizerFast,
    text: str,
    device: str = "cuda:0",
) -> list[str]:

    inputs = prepare_seq2seq_inputs(text=text, tokenizer=tokenizer, device=device)

    if torch.cuda.is_available() and device != "cpu":
        inputs = move_inputs_to_device(inputs, device=device)
        model = model.to(device)

    outputs = model.generate(
        inputs.get("input_ids"))

    answer = tokenizer.batch_decode(
        outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return answer
