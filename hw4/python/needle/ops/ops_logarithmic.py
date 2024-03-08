from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        return Z - logsumexp(Z)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        axes = list(range(len(z.shape)))
        z = node.inputs[0]
        z_max_dim = Tensor(z.get_outputs().max(axes, keepdims=True), device=z.device, requires_grad=False)
        z_exp = exp(z + (-z_max_dim).broadcast_to(z.shape))
        z_exp_sum = summation(z_exp, axes=axes)
        grad_z_exp_sum = 1 / z_exp_sum
        ori_shape = z.shape
        sum_shape = range(len(z.shape)) if axes is None else axes
        now_shape = list(ori_shape)
        for i in sum_shape:
            now_shape[i] = 1
        return (1 - reshape(grad_z_exp_sum, now_shape).broadcast_to(ori_shape) * z_exp) * out_grad
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxz = Z.max(self.axes, keepdims=True)
        ret = array_api.log(array_api.exp(Z - maxz.broadcast_to(Z.shape)).sum(axis=self.axes, keepdims=True)) + maxz
        if self.axes is None:
            axes = list(range(len(Z.shape)))
        elif isinstance(self.axes, int):
            axes = [self.axes]
        else:
            axes = list(self.axes)
        
        if self.axes is not None:
            out_shape = [size for i, size in enumerate(Z.shape) if i not in axes]
        else:
            out_shape = [1]
        
        return ret.reshape(tuple(out_shape))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        z_max_dim = Tensor(z.get_outputs().max(self.axes, keepdims=True), device=z.device, requires_grad=False)
        z_exp = exp(z + (-z_max_dim).broadcast_to(z.shape))
        z_exp_sum = summation(z_exp, axes=self.axes)
        grad_z_exp_sum = out_grad / z_exp_sum
        ori_shape = z.shape
        sum_shape = range(len(z.shape)) if self.axes is None else self.axes
        now_shape = list(ori_shape)
        for i in sum_shape:
            now_shape[i] = 1
        return reshape(grad_z_exp_sum, now_shape).broadcast_to(ori_shape) * z_exp
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

