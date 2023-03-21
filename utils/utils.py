import torch, os, gc
import numpy as np


def calculate_total_params(our_model):
    """Calculate the size of total network."""
    total_parameters = 0
    for variable in our_model.parameters():
        shape = variable.size()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters

    return total_parameters


def tuple_data(data):
    """将yaml读取到的字符数据转换成元组"""
    return eval(repr(data).replace('\'', ''))


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()