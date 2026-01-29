import torch
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

#   Efficient up-convolution block (EUCB)
class EUCB(nn.Module):
    def __init__(self, in_channels, scale_factor=2, mode="nearest"):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.mode = mode
        self.scale_factor = scale_factor
        
        # Upsample + Depthwise Convolution
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=self.scale_factor, mode=self.mode),
            Conv(self.in_channels, self.in_channels, 3, g=self.in_channels, s=1)  # Depthwise conv
        )
        
        # Pointwise Convolution
        self.pwc = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        x = self.up_dwc(x)
        x = self.channel_shuffle(x)
        x = self.pwc(x)
        x = self.bn(x)
        return x
    
    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // 4  # Using 4 groups as default, can be adjusted
        
        # Reshape to group channels
        x = x.view(batchsize, 4, channels_per_group, height, width)
        # Transpose and shuffle
        x = torch.transpose(x, 1, 2).contiguous()
        # Reshape back
        x = x.view(batchsize, -1, height, width)
        return x