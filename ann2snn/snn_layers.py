import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

steps = 16
aa = 0.5  # 梯度近似项
tau = 1.0  # 漏电常数 tau


def config_snn_param(args):
    global steps, tau
    steps = args.timesteps
    tau = args.tau


class SpikeAct(torch.autograd.Function):
    """ 定义脉冲激活函数，并根据论文公式进行梯度的近似。
        Implementation of the spiking activation function with an approximation of gradient.
    """

    @staticmethod
    def forward(ctx, input, Vth):
        ctx.save_for_backward(input)
        # if input = u > Vth then output = 1
        output = torch.gt(input, Vth)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # hu is an approximate func of df/du
        hu = abs(input) < aa
        hu = hu.float() / (2 * aa)

        # 默认返回所有forward参数个数的loss，即为每一个输入参数（input和Vth）返回loss，由于只需要给input
        # 返回，其他的就直接返回None
        return grad_input * hu, None


spikeAct = SpikeAct.apply


def state_update(u_t_n1, o_t_n1, W_mul_o_t1_n, Vth):
    u_t1_n1 = u_t_n1 - Vth * o_t_n1 + W_mul_o_t1_n
    # u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n  # zero模式下tau=1.0与subtraction模式结果相同
    o_t1_n1 = spikeAct(u_t1_n1, Vth)
    return u_t1_n1, o_t1_n1


class SpikeTensor:
    def __init__(self, data, timesteps):
        """
        wrapper for pytorch Tensor.
        data shape: [b, c, h, w, t]
        """
        self.data = data
        self.timesteps = timesteps
        self.b = self.data.size(0)
        self.chw = self.data.size()[1:3]

    def size(self, *args):
        """
        wrapper for self.data.size()
        """
        return self.data.size(*args)

    def view(self, *args):
        """
        wrapper for self.data.view()
        args: [b, c*h*w]
        """
        return SpikeTensor(self.data.view(*args, self.timesteps), self.timesteps)

    def firing_ratio(self):
        """
        calculate the firing ratio over the timesteps which approximate the number in ANN
        """
        firing_ratio = torch.sum(self.data, dim=2) / steps
        return firing_ratio


class tdLayer(nn.Module):
    """将普通的层转换到时间域上。输入张量需要额外带有时间维，此处时间维在数据的最后一维上。前传时，对该时间维中的每一个时间步的数据都执行一次普通层的前传。
        Converts a common layer to the time domain. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data. When forwarding, a normal layer forward is performed for each time step of the data in that time dimension.

    Args:
        layer (nn.Module): 需要转换的层。
            The layer needs to convert.
        bn (nn.Module): 如果需要加入BN，则将BN层一起当做参数传入。
            If batch-normalization is needed, the BN layer should be passed in together as a parameter.
    """

    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = layer
        self.bn = bn

    def forward(self, x):
        if isinstance(x, SpikeTensor):
            x = x.data
            isSpikeTensor = True
        else:
            isSpikeTensor = False

        x_ = torch.zeros(self.layer(x[..., 0]).shape + (steps,), device=x.device)
        # x_ = WrappedTensor(torch.zeros(self.layer(x[..., 0]).shape + (steps,))).to(x.device)
        for step in range(steps):
            x_[..., step] = self.layer(x[..., step])

        if self.bn is not None:
            x_ = self.bn(x_)

        if isSpikeTensor:
            x_ = SpikeTensor(x_, steps)

        return x_


class LIFSpike(nn.Module):
    """对带有时间维度的张量进行一次LIF神经元的发放模拟，可以视为一个激活函数，用法类似ReLU。
        Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """

    def __init__(self, vth=1):
        super(LIFSpike, self).__init__()
        self.Vth = vth

    def forward(self, x):
        if isinstance(x, SpikeTensor):
            x = x.data
            isSpikeTensor = True
        else:
            isSpikeTensor = False

        u = torch.zeros(x.shape[:-1], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        # u = WrappedTensor(torch.zeros(x.shape[:-1])).to(x.device)
        # out = WrappedTensor(torch.zeros(x.shape)).to(x.device)

        for step in range(steps):
            u, out[..., step] = state_update(u, out[..., max(step-1, 0)], x[..., step], self.Vth)

        if isSpikeTensor:
            out = SpikeTensor(out, steps)

        return out
