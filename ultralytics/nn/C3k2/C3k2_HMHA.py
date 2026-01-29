import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
 
 

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Inter_CacheModulation(nn.Module):
    def __init__(self, in_c=3):
        super(Inter_CacheModulation, self).__init__()

        self.align = nn.AdaptiveAvgPool2d(in_c)
        self.conv_width = nn.Conv1d(in_channels=in_c, out_channels=2 * in_c, kernel_size=1)
        self.gatingConv = nn.Conv1d(in_channels=in_c, out_channels=in_c, kernel_size=1)

    def forward(self, x1, x2):
        C = x1.shape[-1]
        x2_pW = self.conv_width(self.align(x2) + x1)
        scale, shift = x2_pW.chunk(2, dim=1)
        x1_p = x1 * scale + shift
        x1_p = x1_p * F.gelu(self.gatingConv(x1_p))
        return x1_p


class Intra_CacheModulation(nn.Module):
    def __init__(self, embed_dim=48):
        super(Intra_CacheModulation, self).__init__()

        self.down = nn.Conv1d(embed_dim, embed_dim // 2, kernel_size=1)
        self.up = nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=1)
        self.gatingConv = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1)

    def forward(self, x1, x2):
        x_gated = F.gelu(self.gatingConv(x2 + x1)) * (x2 + x1)
        x_p = self.up(self.down(x_gated))
        return x_p


class ReGroup(nn.Module):
    def __init__(self, groups=[1, 1, 2, 4]):
        super(ReGroup, self).__init__()
        self.gourps = groups

    def forward(self, query, key, value):
        C = query.shape[1]
        channel_features = query.mean(dim=0)
        correlation_matrix = torch.corrcoef(channel_features)

        mean_similarity = correlation_matrix.mean(dim=1)
        _, sorted_indices = torch.sort(mean_similarity, descending=True)

        query_sorted = query[:, sorted_indices, :]
        key_sorted = key[:, sorted_indices, :]
        value_sorted = value[:, sorted_indices, :]

        query_groups = []
        key_groups = []
        value_groups = []
        start_idx = 0
        total_ratio = sum(self.gourps)
        group_sizes = [int(ratio / total_ratio * C) for ratio in self.gourps]

        for group_size in group_sizes:
            end_idx = start_idx + group_size
            query_groups.append(query_sorted[:, start_idx:end_idx, :])
            key_groups.append(key_sorted[:, start_idx:end_idx, :])
            value_groups.append(value_sorted[:, start_idx:end_idx, :])
            start_idx = end_idx

        return query_groups, key_groups, value_groups


def CalculateCurrentLayerCache(x, dim=128, groups=[1, 1, 2, 4]):
    lens = len(groups)
    ceil_dim = dim  # * max_value // sum_value
    for i in range(lens):
        qv_cache_f = x[i].clone().detach()
        qv_cache_f = torch.mean(qv_cache_f, dim=0, keepdim=True).detach()
        update_elements = F.interpolate(qv_cache_f.unsqueeze(1), size=(ceil_dim, ceil_dim), mode='bilinear',
                                        align_corners=False)
        c_i = qv_cache_f.shape[-1]

        if i == 0:
            qv_cache = update_elements * c_i // dim
        else:
            qv_cache = qv_cache + update_elements * c_i // dim

    return qv_cache.squeeze(1)

class HMHA(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super(HMHA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(4, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.group = [1, 2, 2, 3]

        self.intra_modulator = Intra_CacheModulation(embed_dim=dim)

        self.inter_modulator1 = Inter_CacheModulation(in_c=1 * dim // 8)
        self.inter_modulator2 = Inter_CacheModulation(in_c=2 * dim // 8)
        self.inter_modulator3 = Inter_CacheModulation(in_c=2 * dim // 8)
        self.inter_modulator4 = Inter_CacheModulation(in_c=3 * dim // 8)
        self.inter_modulators = [self.inter_modulator1, self.inter_modulator2, self.inter_modulator3,
                                 self.inter_modulator4]

        self.regroup = ReGroup(self.group)
        self.dim = dim

    def forward(self, x ):
        b, c, h, w = x.shape
        qv_cache = None
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')

        qu, ke, va = self.regroup(q, k, v)
        attScore = []
        tmp_cache = []
        for index in range(len(self.group)):
            query_head = qu[index]
            key_head = ke[index]

            query_head = torch.nn.functional.normalize(query_head, dim=-1)
            key_head = torch.nn.functional.normalize(key_head, dim=-1)

            attn = (query_head @ key_head.transpose(-2, -1)) * self.temperature[index, :, :]
            attn = attn.softmax(dim=-1)

            attScore.append(attn)  # CxC
            t_cache = query_head.clone().detach() + key_head.clone().detach()
            tmp_cache.append(t_cache)

        tmp_caches = torch.cat(tmp_cache, 1)
    
        out = []
        if qv_cache is not None:
            if qv_cache.shape[-1] != c:
                qv_cache = F.adaptive_avg_pool2d(qv_cache, c)
        for i in range(4):
            if qv_cache is not None:
                inter_modulator = self.inter_modulators[i]
                attScore[i] = inter_modulator(attScore[i], qv_cache) + attScore[i]
                out.append(attScore[i] @ va[i])
            else:
                out.append(attScore[i] @ va[i])

        update_factor = 0.9
        if qv_cache is not None:

            update_elements = CalculateCurrentLayerCache(attScore, c, self.group)
            qv_cache = qv_cache * update_factor + update_elements * (1 - update_factor)
        else:
            qv_cache = CalculateCurrentLayerCache(attScore, c, self.group)
            qv_cache = qv_cache * update_factor

        out_all = torch.concat(out, 1)
        # Intra Modulation
        out_all = self.intra_modulator(out_all, tmp_caches) + out_all

        out_all = rearrange(out_all, 'b  c (h w) -> b c h w', h=h, w=w)
        out_all = self.project_out(out_all)
        return out_all


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()

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


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


 



class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C2f_HMHA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_HMHA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
        
class Bottleneck_HMHA(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.Attention = HMHA(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))

class C3k_HMHA(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_HMHA(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

class C3k2_HMHA(C2f_HMHA):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_HMHA(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_HMHA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
