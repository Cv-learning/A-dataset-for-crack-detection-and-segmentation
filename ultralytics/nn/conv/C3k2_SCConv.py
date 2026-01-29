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
    """
    Self-Calibrated Convolution (SCConv) module.
    This module enhances feature representation by calibrating spatial features.
    """
 
    def __init__(self, c1, c2, s=1, d=1, g=1, pooling_r=4, act=True):
        """
        Initialize SCConv module.
        
        Args:
            c1 (int): Input channels
            c2 (int): Output channels
            s (int): Stride
            d (int): Dilation rate
            g (int): Groups
            pooling_r (int): Pooling ratio for k2 branch
            act (bool): Whether to use activation function
        """
        super(SCConv, self).__init__()
        # k2 branch: pooling -> conv -> norm
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r), 
                    Conv(c1, c2, k=3, d=d, g=g, act=act)
                    )
        # k3 branch: direct conv
        self.k3 = Conv(c1, c2, k=3, d=d, g=g, act=act)
        # k4 branch: conv with stride
        self.k4 = Conv(c2, c2, k=3, s=s, d=d, g=g, act=act) # Note: input to k4 is c2 (output of k3*x_gate)

    def forward(self, x):
        """
        Forward pass of SCConv.
        
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            Tensor: Output tensor after SCConv operation
        """
        identity = x

        # Process k2 branch and upsample to match identity size
        k2_out = self.k2(x)
        k2_upsampled = F.interpolate(k2_out, size=identity.shape[2:], mode='bilinear', align_corners=False)
        
        # Create attention gate: sigmoid(identity + upsampled_k2)
        gate = torch.sigmoid(identity + k2_upsampled)
        
        # Apply gate to k3 output: k3_output * gate
        k3_out = self.k3(x)
        gated_output = k3_out * gate
        
        # Final conv (k4)
        out = self.k4(gated_output)

        return out


class Bottleneck(nn.Module):
    """Standard bottleneck block."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        Initialize bottleneck block.
        
        Args:
            c1 (int): Input channels
            c2 (int): Output channels
            shortcut (bool): Whether to use shortcut connection
            g (int): Groups
            k (tuple): Kernel sizes for two conv layers
            e (float): Expansion ratio
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    

class Bottleneck_SCConv(Bottleneck):
    """Bottleneck block using SCConv for the second convolution."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, pooling_r=4):
        """
        Initialize bottleneck with SCConv.
        
        Args:
            c1 (int): Input channels
            c2 (int): Output channels
            shortcut (bool): Whether to use shortcut connection
            g (int): Groups
            k (tuple): Kernel sizes (Note: k[1] is used for SCConv)
            e (float): Expansion ratio
            pooling_r (int): Pooling ratio for SCConv
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        # Use SCConv for the second layer instead of standard Conv
        self.cv2 = SCConv(c_, c2, s=1, d=1, g=g, pooling_r=pooling_r)
        # The add logic remains the same as parent class

    def forward(self, x):
        """Forward pass with SCConv."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Initialize C2f module.
        
        Args:
            c1 (int): Input channels
            c2 (int): Output channels
            n (int): Number of bottleneck blocks
            shortcut (bool): Whether bottleneck blocks use shortcut
            g (int): Groups for convolutions
            e (float): Expansion ratio
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        # Default uses standard Bottleneck
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        # Split the output of cv1 into two parts
        y = list(self.cv1(x).chunk(2, 1))
        # Apply each bottleneck module in sequence
        for m in self.m:
            y.append(m(y[-1]))
        # Concatenate all parts and apply final conv
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize C3 module.
        
        Args:
            c1 (int): Input channels
            c2 (int): Output channels
            n (int): Number of bottleneck blocks
            shortcut (bool): Whether bottleneck blocks use shortcut
            g (int): Groups for convolutions
            e (float): Expansion ratio
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        # Default uses standard Bottleneck
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k_SCConv(C3):
    """C3 module with SCConv bottleneck blocks."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3, pooling_r=4):
        """
        Initialize C3k_SCConv module.
        
        Args:
            c1 (int): Input channels
            c2 (int): Output channels
            n (int): Number of bottleneck blocks
            shortcut (bool): Whether bottleneck blocks use shortcut
            g (int): Groups for convolutions
            e (float): Expansion ratio
            k (int): Kernel size for bottleneck convolutions
            pooling_r (int): Pooling ratio for SCConv
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # Replace standard Bottlenecks with Bottleneck_SCConv
        self.m = nn.Sequential(*(Bottleneck_SCConv(c_, c_, shortcut, g, k=(k, k), e=1.0, pooling_r=pooling_r) for _ in range(n)))


class C3k2_SCConv(C2f):
    """
    C2f (C3k2) module using SCConv bottleneck blocks for crack detection.
    Combines the efficiency of C2f with the attention mechanism of SCConv.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, pooling_r=4):
        """
        Initialize C3k2_SCConv module.
        
        Args:
            c1 (int): Input channels
            c2 (int): Output channels
            n (int): Number of bottleneck blocks
            c3k (bool): If True, use C3k_SCConv blocks instead of Bottleneck_SCConv
            e (float): Expansion ratio
            g (int): Groups for convolutions
            shortcut (bool): Whether bottleneck blocks use shortcut
            pooling_r (int): Pooling ratio for SCConv
        """
        # Initialize parent C2f class
        super().__init__(c1, c2, n, shortcut, g, e)
        
        # Replace the bottleneck modules with SCConv-enhanced versions
        if c3k:
            # Use C3k_SCConv blocks (each containing multiple Bottleneck_SCConv)
            # Note: This creates a nested structure which might be too complex
            # The original C2f structure expects simple Bottleneck-like blocks
            # So this path might need specific implementation if desired
            self.m = nn.ModuleList(C3k_SCConv(self.c, self.c, 2, shortcut, g, pooling_r=pooling_r) for _ in range(n))
        else:
            # Use simple Bottleneck_SCConv blocks, which is more typical for C2f structure
            self.m = nn.ModuleList(Bottleneck_SCConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, pooling_r=pooling_r) for _ in range(n))

    def forward(self, x):
        """Forward pass through C3k2_SCConv layer."""
        # Reuse the parent's forward logic
        return super().forward(x)

# Optional: A more direct version if c3k=True is intended to mean using C3k structure within C2f-like processing
class C3k2_SCConv_Direct(nn.Module):
    """
    Alternative C3k2_SCConv implementation with clearer structure.
    This version explicitly implements the C2f-like split and merge logic with SCConv.
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, pooling_r=4):
        """
        Initialize direct C3k2_SCConv module.
        
        Args:
            c1 (int): Input channels
            c2 (int): Output channels
            n (int): Number of bottleneck blocks
            shortcut (bool): Whether bottleneck blocks use shortcut
            g (int): Groups for convolutions
            e (float): Expansion ratio
            pooling_r (int): Pooling ratio for SCConv
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        # Use SCConv-enhanced bottleneck blocks
        self.m = nn.ModuleList(
            Bottleneck_SCConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, pooling_r=pooling_r) 
            for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through the direct C3k2_SCConv layer."""
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))