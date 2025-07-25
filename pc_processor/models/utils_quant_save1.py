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
        # print(self.alpha.shape, self.zero_point.shape)

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
        if self.training and self.init_state == 0:
            # The init alpha for activation is very very important as the experimental results shows.
            # Please select a init_rate for activation.
            # self.alpha.data.copy_(x.max() / 2 ** (self.nbits - 1) * self.init_rate)

            Qn = 0
            Qp = 2 ** (self.nbits - 1) - 1

            self.alpha.data.copy_(2 * x.abs().mean(dim=-1).mean(dim=-1).mean(dim=0) / math.sqrt(Qp))
            self.zero_point.data.copy_(
                self.zero_point.data * 0.9 + 0.1 * x.detach().min(dim=-1)[0].min(dim=-1)[0].min(dim=0)[
                    0] - self.alpha.data * Qn)
            self.init_state.fill_(1)

        Qn = 0
        Qp = 2 ** (self.nbits - 1) - 1

        g = self.num_head / math.sqrt(x.numel() * Qp)

        # Method1:
        zero_point = (self.zero_point.round() - self.zero_point).detach() + self.zero_point
        alpha = self.grad_scale(self.alpha, g)
        zero_point = self.grad_scale(zero_point, g)
        alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        x = self.round_pass(x / alpha + zero_point).clamp(Qn, Qp)
        x = (x - zero_point) * alpha

        return x


############################################################################################
class BinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        input = input[0]
        # indicate_small = (input < -1).float()
        # indicate_big = (input > 1).float()
        indicate_leftmid = ((input >= -1) & (input <= 0)).float()
        indicate_rightmid = ((input > 0) & (input <= 1)).float()

        grad_input = (indicate_leftmid * (2 + 2 * input) + indicate_rightmid * (2 - 2 * input)) * grad_output.clone()
        return grad_input


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights, 1)) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights), dim=3, keepdim=True), dim=2, keepdim=True),
                                    dim=1, keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        # print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(out_chn), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


def act_quant_fn(input):
    input = BinaryQuantizer.apply(input)

    return input


class QuantizeConv2d(nn.Conv2d):
    def __init__(self, *kargs, bias=True):
        # 调用父类 nn.Conv2d 的构造函数
        super(QuantizeConv2d, self).__init__(*kargs, bias=bias)
        # def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=0, dilation=1, bias=True, *kargs):
        # 初始化权重量化器和激活量化器
        self.weight_quantizer = BinaryQuantizer
        self.act_quantizer = BinaryQuantizer

    def forward(self, input):
        # 获取实际的卷积核权重
        real_weights = self.weight
        # 计算缩放因子
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights), dim=3, keepdim=True), dim=2, keepdim=True),
                                    dim=1, keepdim=True)
        # 对权重进行去均值操作
        real_weights = real_weights - real_weights.mean([1, 2, 3], keepdim=True)
        # 对权重进行方差归一化操作
        real_weights = real_weights / (torch.sqrt(real_weights.var([1, 2, 3], keepdim=True) + 1e-5) / 2 / np.sqrt(2))
        # 计算权重的绝对值期望
        EW = torch.mean(torch.abs(real_weights))
        # 计算量化阈值 Q_tau
        Q_tau = (- EW * np.log(2 - 2 * 0.92)).detach().cpu().item()
        # 将缩放因子从计算图中分离出来
        scaling_factor = scaling_factor.detach()
        # 生成二值化的权重（无梯度）
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        # 对权重进行截断操作
        cliped_weights = torch.clamp(real_weights, -Q_tau, Q_tau)
        # 计算最终的量化权重
        weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        # 对输入进行激活量化
        input = self.act_quantizer.apply(input)
        # 进行卷积操作
        out = nn.functional.conv2d(input, weight, stride=self.stride, padding=self.padding, dilation=self.dilation)
        # 如果存在偏置，则加上偏置
        if not self.bias is None:
            out = out + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return out


# 对权重和激活进行二值化
# class QuantizeLinear(nn.Module):
#     def __init__(self, in_channels, out_channels, bias=False):
#         super(QuantizeLinear, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.weight_quantizer = BinaryQuantizer  # 假设 BinaryQuantizer 已经定义
#
#     def forward(self, input):
#         # 将 input 从 (N, L) 转换为 (N, C, H, W)，其中 C=in_channels, H=W=1
#         input = input.unsqueeze(-1).unsqueeze(-1)
#
#         # 量化权重
#         scaling_factor = torch.mean(abs(self.conv.weight), dim=1, keepdim=True)
#         scaling_factor = scaling_factor.detach()
#         real_weights = self.conv.weight - torch.mean(self.conv.weight, dim=-1, keepdim=True)
#         real_weights = real_weights / (torch.sqrt(real_weights.var(dim=-1, keepdim=True) + 1e-5) / 2 / np.sqrt(2))
#         EW = torch.mean(torch.abs(real_weights))
#         Q_tau = (-EW * np.log(2 - 2 * 0.92)).detach().cpu().item()
#         scaling_factor = scaling_factor.detach()
#         binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
#         cliped_weights = torch.clamp(real_weights, -Q_tau, Q_tau)
#         weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
#
#         # 使用量化权重进行 1x1 卷积
#         out = nn.functional.conv2d(input, weight, bias=self.conv.bias, stride=1, padding=0)
#
#         # 将输出从 (N, C_out, 1, 1) 转换回 (N, C_out)
#         out = out.squeeze(-1).squeeze(-1)
#
#         return out

class SymQuantizer(torch.autograd.Function):
    """
        uniform quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            if input.ndimension() <= 3:
                max_input = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_input = torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        s = (2 ** (num_bits - 1) - 1) / max_input
        output = torch.round(input * s).div(s)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None

class TwnQuantizer(torch.autograd.Function):
    """Ternary Weight Networks (TWN)
    Ref: https://arxiv.org/abs/1605.04711
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param input: tensor to be ternarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            m = input.norm(p=1).div(input.nelement())
            thres = 0.7 * m
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = (mask * input).abs().sum() / mask.sum()
            result = alpha * pos - alpha * neg
        else: # row-wise only for embed / weight
            n = input[0].nelement()
            m = input.data.norm(p=1, dim=1).div(n)
            thres = (0.7 * m).view(-1, 1).expand_as(input)
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = ((mask * input).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
            result = alpha * pos - alpha * neg

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class QuantizeLinear(nn.Linear):
    def __init__(self,  *kargs, bias=False, config=None):
        super(QuantizeLinear, self).__init__(*kargs, bias=bias)
        self.weight_bits = 1
        self.input_bits = 1
        self.recu = config.recu
        if self.weight_bits == 1:
            self.weight_quantizer = BinaryQuantizer
        elif self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
        elif self.weight_bits < 32:
            self.weight_quantizer = SymQuantizer
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))

        if self.input_bits == 1:
            self.act_quantizer = BinaryQuantizer
        elif self.input_bits == 2:
            self.act_quantizer = TwnQuantizer
            self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
        elif self.input_bits < 32:
            self.act_quantizer = SymQuantizer
            self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))


    def forward(self, input):
        if self.weight_bits == 1:
            scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
            scaling_factor = scaling_factor.detach()
            real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
            if self.recu:
                #print(scaling_factor, flush=True)

                real_weights= real_weights/(torch.sqrt(real_weights.var(dim=-1, keepdim=True) + 1e-5) / 2 / np.sqrt(2))
                EW = torch.mean(torch.abs(real_weights))
                Q_tau = (- EW * np.log(2-2*0.92)).detach().cpu().item()
                scaling_factor = scaling_factor.detach()
                binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
                cliped_weights = torch.clamp(real_weights, -Q_tau, Q_tau)
                weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
                #print(binary_weights, flush=True)
            else:
                scaling_factor = scaling_factor.detach()
                binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
                cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
                weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        elif self.weight_bits < 32:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)
        else:
            weight = self.weight

        if self.input_bits == 1:
            input = self.act_quantizer.apply(input)

        out = nn.functional.linear(input, weight)

        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out