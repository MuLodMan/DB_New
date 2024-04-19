import math
import torch
from torch import Tensor
from torchvision.ops import deform_conv2d
from torch.nn.modules.utils import _pair
from torchvision.utils import _log_api_usage_once
from torch.nn import Module,Parameter,init
from typing import Optional

t1 = _pair(3)
class DeformConv2d(Module):
    """
    See :func:`deform_conv2d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        _log_api_usage_once(self)

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = Parameter(
            torch.empty(out_channels, in_channels // groups, self.kernel_size[0], self.kernel_size[1])
        )

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, offset: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
            offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width]):
                offsets to be applied for each position in the convolution kernel.
            mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width, out_height, out_width]):
                masks to be applied for each position in the convolution kernel.
        """
        return deform_conv2d(
            input,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"{self.in_channels}"
            f", {self.out_channels}"
            f", kernel_size={self.kernel_size}"
            f", stride={self.stride}"
        )
        s += f", padding={self.padding}" if self.padding != (0, 0) else ""
        s += f", dilation={self.dilation}" if self.dilation != (1, 1) else ""
        s += f", groups={self.groups}" if self.groups != 1 else ""
        s += ", bias=False" if self.bias is None else ""
        s += ")"

        return s