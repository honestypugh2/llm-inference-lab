"""
This module and its functions are intended to be used with ray for storing model weights in ray
object storage for fast loading and reduced memory usage. It does this by taking advantage of ray
zero copy capability for numpy arrays as well as torch's ability to directly use numpy arrays as
tensors (no conversion is required).
"""
from typing import Iterable
import copy
import ray
from ray import ObjectRef
import torch
from torch.nn import Module, Parameter
import numpy as np


NamedTensorT = tuple[str, torch.Tensor]
NamedTensorIterT = Iterable[NamedTensorT]
NumpyDictT = dict[str, np.ndarray]
NumpyModuleDictT = dict[str, NumpyDictT]


#######################################################
# Helpers for preparing models for ray object storage #
#######################################################
def get_model_weights_as_numpy_arrays(model: Module) -> list[NumpyModuleDictT]:
    """
    Returns a list containing the named paramters and buffers for each module in the model as numpy
    arrays.
    """
    numpy_dicts = []
    for _module_name, module in model.named_modules():
        numpy_dicts.append(module_tensors_to_numpy_dict(module))
    return numpy_dicts


def module_tensors_to_numpy_dict(module: Module) -> NumpyModuleDictT:

    params = named_tensors_to_numpy_dict(module.named_parameters(recurse=False))
    buffers = named_tensors_to_numpy_dict(module.named_buffers(recurse=False))
    return {"parameters": params, "buffers": buffers}


def named_tensors_to_numpy_dict(name_tensor_iter: NamedTensorIterT) -> NumpyDictT:
    numpy_dict = {}
    for name, torch_tensor in name_tensor_iter:
        numpy_tensor = tensor_to_numpy(torch_tensor)
        numpy_dict[name] = numpy_tensor
    return numpy_dict


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return torch.clone(tensor).detach().numpy()


def remove_model_weights(model: Module) -> Module:
    model_copy = copy.deepcopy(model)
    for _module_name, module in model_copy.named_modules():
        remove_tensor_attributes(module, module.named_parameters(recurse=False))
        remove_tensor_attributes(module, module.named_buffers(recurse=False))
    model_copy.train(False)
    return model_copy


def remove_tensor_attributes(module: Module, named_tensor_iter: NamedTensorIterT):
    for attr_name, _attr_value in named_tensor_iter:
        setattr(module, attr_name, None)


#####################################################
# Helpers for loading models for ray object storage #
#####################################################


def load_model_from_numpy_dicts(
    empty_model: Module, numpy_dicts: list[NumpyModuleDictT]
) -> None:
    """
    Takes an empty torch model and a list of numpy dicts weights corresponding to the
    parameters/buffers (weights) of each layer in the model and loads them into the empty model.
    This is intended to be used in conjunction with the remove_model_weights and
    get_model_weights_as_numpy_arrays functions above.
    """
    with torch.inference_mode():
        empty_modules = [module for _module_name, module in empty_model.named_modules()]
        for module, numpy_dict in zip(empty_modules, numpy_dicts):
            load_module_from_numpy_dict(module, numpy_dict)


def load_module_from_numpy_dict(module: Module, numpy_dict: NumpyModuleDictT) -> None:
    load_module_parameters_from_numpy(module, numpy_dict["parameters"])
    load_module_buffers_from_numpy(module, numpy_dict["buffers"])


def load_module_parameters_from_numpy(
    module: Module, named_params_dict: NumpyDictT
) -> None:
    for name, numpy_array in named_params_dict.items():
        module.register_parameter(name, Parameter(torch.as_tensor(numpy_array)))


def load_module_buffers_from_numpy(module, named_buffers_dict: NumpyDictT) -> None:
    for name, numpy_array in named_buffers_dict.items():
        module.register_buffer(name, torch.as_tensor(numpy_array))


def load_model_from_ray(model_ref: ObjectRef) -> Module:
    model, weights = ray.get(model_ref)  # pyright: ignore[reportPrivateImportUsage]
    load_model_from_numpy_dicts(model, weights)
    return model
