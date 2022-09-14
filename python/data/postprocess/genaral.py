

import numpy as np
from data.base import OpBase, op_register


@op_register
class Reshape(OpBase):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, **kwargs):
        for k, v in self.kwargs.items():
            kwargs[k] = np.reshape(kwargs[k], v)
        return kwargs


@op_register("Transpose")
class Transpose(OpBase):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, **kwargs):
        for k, v in self.kwargs.items():
            kwargs[k] = np.transpose(kwargs[k], v)
        return kwargs


@op_register
class ReOutputsKey(OpBase):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, **kwargs):
        for k, v in self.kwargs.items():
            kwargs[k] = kwargs[v]
            kwargs.pop(v)
        return kwargs

