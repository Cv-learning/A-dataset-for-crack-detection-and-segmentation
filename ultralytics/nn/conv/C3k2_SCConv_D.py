import torch
import torch.nn.functional as F
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class SCConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1, groups: int = 1, pooling_r: int = 2):
        super().__init__()
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding),
            Conv(c1=in_channels, c2=out_channels, k=1, g=groups)
        )
        self.k3 = nn.Sequential(
            Conv(c1=in_channels, c2=out_channels, k=1, g=groups),
            nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        )
        self.k4 = nn.Sequential(
            Conv(c1=in_channels, c2=out_channels, k=1, g=groups),
            Conv(c1=out_channels, c2=out_channels, k=3, g=groups, p=autopad(kernel_size, padding, dilation), d=dilation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.k3(x)
        k2 = self.k2(x)
        k4 = self.k4(x)
        out = torch.sigmoid(identity) * k2 + k4
        return out


class C3k2_SCConv_D(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, c_: int = None, shortcut: bool = True, g: int = 1, e: float = 0.25, dropout_rate: float = 0.0): # Add dropout_rate parameter
        super().__init__()
        c_ = c_ or int(c2 * e)  # Calculate intermediate channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1, 1) # Adjust output channels for cv3

        self.m = nn.ModuleList()
        SCConv_instance = SCConv(c_, c_, 3, 1, 1) # Create one SCConv instance
        for i in range(n):
            self.m.append(nn.Sequential(
                SCConv_instance, # Reuse the same SCConv instance
                Conv(c_, c_, 3, 1, g=g)
            ))

        # Add Dropout layer
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None  # No dropout if rate is 0

        self.n = n
        self.shortcut = shortcut  # Store for potential use

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.cv1(x)
        y2 = self.cv2(x)

        # Apply shortcut if enabled (though logic might be different from standard C3)
        # For C3k2, it's more about concatenation, so shortcut is typically False
        # If you want to add residual connection: y2 = y2 + x if self.shortcut and y2.shape == x.shape else y2

        ys = [y1, y2]
        for i in range(self.n):
            # Apply module and append result
            ys.append(self.m[i](ys[-1]))

        out = self.cv3(torch.cat(ys, 1))
        if self.dropout is not None:
            out = self.dropout(out) # Apply dropout after the final convolution of the block
        return out