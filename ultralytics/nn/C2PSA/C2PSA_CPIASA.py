import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from einops import rearrange


class ComplexIFFT(nn.Module):
    def __init__(self):
        super(ComplexIFFT, self).__init__()

    def forward(self, real, imag):
        x_complex = torch.complex(real, imag)
        x_ifft = fft.ifft2(x_complex, dim=(-2, -1))  # 2D IFFT
        return x_ifft.real  


class Conv1x1(nn.Module):
    def __init__(self, in_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, padding=0,groups=in_channels*2)

    def forward(self, x):
        return self.conv(x)



class ComplexFFT(nn.Module):
    def __init__(self):
        super(ComplexFFT, self).__init__()

    def forward(self, x):
        x_fft = fft.fft2(x, dim=(-2, -1))  
        real = x_fft.real  
        imag = x_fft.imag 
        return real, imag

class Stage2_fft(nn.Module):
    def __init__(self, in_channels):
        super(Stage2_fft, self).__init__()
        self.c_fft = ComplexFFT()
        self.conv1x1 = Conv1x1(in_channels)
        self.c_ifft = ComplexIFFT()

    def forward(self, x):
        real, imag = self.c_fft(x)

        combined = torch.cat([real, imag], dim=1)
        conv_out = self.conv1x1(combined)

        out_channels = conv_out.shape[1] // 2
        real_out = conv_out[:, :out_channels, :, :]
        imag_out = conv_out[:, out_channels:, :, :]

        output = self.c_ifft(real_out, imag_out)

        return output

class spr_sa(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim,hidden_dim,3,1,1,groups=dim),
            nn.Conv2d(hidden_dim,hidden_dim,1,1,0)
        )
        self.act =nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)


    def forward(self, x):
        x = self.conv_0(x)
        x1= F.adaptive_avg_pool2d(x, (1, 1))
        x1 = F.softmax(x1, dim=1)
        x=x1*x
        x = self.act(x)
        x = self.conv_1(x)
        return x

class CPIASA(nn.Module):
    def __init__(self, dim, num_heads=4, bias=True):
        super(CPIASA, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.spr_sa=spr_sa(dim//2,2)
        self.linear_0 = nn.Conv2d(dim, dim , 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.qkv = nn.Conv2d(dim//2, dim//2 * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim//2 * 3, dim//2 * 3, kernel_size=3, stride=1, padding=1, groups=dim//2 * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        # Use GroupNorm instead of BatchNorm2d here to avoid batch-size restrictions
        # when the spatial resolution becomes 1x1 during stride probing.
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim // 2, dim // 8, kernel_size=1),
            nn.GroupNorm(1, dim // 8),  # safe for 1x1 feature maps and small batch sizes
            nn.GELU(),
            nn.Conv2d(dim // 8, dim // 2, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 16, kernel_size=1),
            nn.GroupNorm(1, dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1),
        )
        self.fft=Stage2_fft(in_channels=dim)
        self.gate = nn.Sequential(
            nn.Conv2d(dim//2, dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, 1, kernel_size=1),  # 输出动态 K
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        y, x = self.linear_0(x).chunk(2, dim=1)

        y_d = self.spr_sa(y)

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape
        gate_output = self.gate(x).view(b, -1).mean()
        # Handle NaN/Inf values: replace with default value 0.5 (middle of sigmoid range)
        if torch.isnan(gate_output) or torch.isinf(gate_output):
            gate_output = torch.tensor(0.5, device=x.device, dtype=x.dtype)
        # Clamp to valid range [0, 1]
        gate_output = torch.clamp(gate_output, 0.0, 1.0)
        # Convert to Python float and compute dynamic_k, ensuring it's at least 1 and at most C
        gate_value = float(gate_output.item())
        dynamic_k = max(1, min(int(C * gate_value), C))
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        mask = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=dynamic_k, dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))

        attn = attn.softmax(dim=-1)
        out1 = (attn @ v)
        out2 = (attn @ v)
        out3 = (attn @ v)
        out4 = (attn @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out_att = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # Frequency Adaptive Interaction Module (FAIM)
        # stage1
        # C-Map (before sigmoid)
        channel_map = self.channel_interaction(out_att)
        # S-Map (before sigmoid)
        spatial_map = self.spatial_interaction(y_d)

        # S-I
        attened_x = out_att * torch.sigmoid(spatial_map)
        # C-I
        conv_x = y_d * torch.sigmoid(channel_map)

        x = torch.cat([attened_x, conv_x], dim=1)
        out = self.project_out(x)
        # stage 2
        out=self.fft(out)
        return out
class Attention(nn.Module):


    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x
 
def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        c = self.conv(x)
        c = self.bn(c)
        c = self.act(c)
        return c
 
    
 
 
class C2PSA(nn.Module):


    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
    
class PSABlock(nn.Module):
 

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x 
    
 
 
class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
 
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
 
    def forward(self, x):
        b, c, h, w = x.shape
 
        x = x.view(b, c, h * w).permute(0, 2, 1)  # (b, h*w, c)
 
        qkv = self.qkv(x).reshape(b, h * w, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
 
        key = F.softmax(k, dim=-1)
        query = F.softmax(q, dim=-2)
        context = key.transpose(-2, -1) @ v
        x = (query @ context).reshape(b, h * w, c)
 
        x = self.proj(x)
 
        x = x.permute(0, 2, 1).view(b, c, h, w)
 
        return x
 
 
class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels,
                                   padding=kernel_size // 2)
        self.relu = nn.ReLU()
 
    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = x + residual
        x = self.relu(x)
        return x
 
 
 
class CPIASABlock(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        
        self.attn = CPIASA(c)
 
 
class C2PSA_CPIASA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        
        self.m = nn.Sequential(*(CPIASABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

 
 


