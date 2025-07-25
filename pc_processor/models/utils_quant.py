import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BinaryActivation_Attention(nn.Module):
    def __init__(self, num_head, nbits_a=4, **kwargs):
        super(BinaryActivation_Attention, self).__init__()
        self.num_head = num_head
        self.alpha = nn.Parameter(torch.ones([num_head]))
        self.zero_point = nn.Parameter(torch.zeros([num_head]))
        self.register_buffer('init_state', torch.zeros(1))
        self.nbits = nbits_a

    def grad_scale(self, x, scale):
        y = x
        y_grad = x * scale
        return y.detach() - y_grad.detach() + y_grad

    def round_pass(self, x):
        y = x.round()
        y_grad = x
        return y.detach() - y_grad.detach() + y_grad

    def forward(self, x):
        if self.alpha is None:
            return x

        Qn = 0
        Qp = 2 ** (self.nbits - 1) - 1
        zero_point = self.zero_point
        alpha = self.alpha

        alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        x = self.round_pass(x / alpha + zero_point).clamp(Qn, Qp)
        x = (x - zero_point) * alpha

        return x


############################################################################################

def BinaryQuantizer(input):
    out = torch.sign(input)
    return out

class HardBinaryConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size, kernel_size) * 0.001, requires_grad=True)

    def forward(self, x):
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(self.weight), dim=3, keepdim=True), dim=2, keepdim=True), dim=1, keepdim=True)
        C=scaling_factor.shape[0]
        binary_weights = torch.sign(self.weight)
        y = F.conv2d(x, binary_weights, None, self.stride, self.padding)
        y *= scaling_factor.reshape(1,C,1,1)
        return y


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(out_chn), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


def act_quant_fn(input):
    input = BinaryQuantizer(input)

    return input


class QuantizeConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(QuantizeConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)
        # Initialize weights as small random values
        self.weight = nn.Parameter(torch.randn(self.weight.shape) * 0.001, requires_grad=True)

    def forward(self, input):
        real_weights = self.weight
        # Compute the scaling factor as the mean of absolute values of real_weights
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights), dim=3, keepdim=True), dim=2, keepdim=True), dim=1, keepdim=True)
        binary_weights = torch.sign(real_weights)

        # Perform the convolution using the binary weights
        output = F.conv2d(input, binary_weights, None, self.stride, self.padding, self.dilation, self.groups)
        output = output * scaling_factor.reshape([-1, 1, 1])
        return output


class HardBinaryConv1(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=1, stride=1, padding=0):
        super(HardBinaryConv1, self).__init__()
        self.stride = stride
        self.padding = padding
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.rand(self.shape) * 0.001, requires_grad=True)

    def forward(self, x):
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(self.weight), dim=3, keepdim=True),
                                               dim=2, keepdim=True), dim=1, keepdim=True)
        binary_weights = torch.sign(self.weight)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)
        y *= scaling_factor.reshape([-1, 1, 1])
        return y


# class QuantizeLinear(nn.Linear):
#     def __init__(self, in_features, out_features, bias=True):
#         super(QuantizeLinear, self).__init__(in_features, out_features, bias)
#         self.weight.data.uniform_(-0.001, 0.001)  # 初始化权重为小随机数
#
#     def forward(self, input):
#         binary_weights = self.weight.sign()
#         return F.linear(input, binary_weights, self.bias)

class QuantizeLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(QuantizeLinear, self).__init__()
        self.conv = HardBinaryConv(in_features, out_features)
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_features, 1, 1))
        else:
            self.bias = None

    def forward(self, input):
        # Assume input shape is (batch_size, sequence_length, features)
        # We need to reshape it to (batch_size * sequence_length, features, 1, 1) for the convolution
        batch_size, sequence_length, features = input.shape
        # Reshape for convolution
        B, N, C = input.shape
        input = input.reshape(B, C, int(math.sqrt(N)), int(math.sqrt(N)))
        # Perform the 1x1 convolution
        output = self.conv(input)
        b, c, h, w = output.shape
        # Reshape back to the original sequence format with new feature size
        output = output.reshape(B, N, c)
        return output