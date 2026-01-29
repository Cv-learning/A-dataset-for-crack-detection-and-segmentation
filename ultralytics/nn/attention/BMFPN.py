import torch
import torch.nn as nn
import math

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

# --- MCA Components ---
class MCAGate(nn.Module):
    """MCA Gate module for channel and spatial attention."""
    def __init__(self, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, L)
        avg_out = torch.mean(x, dim=1, keepdim=True) # (B, 1, L)
        avg_out = self.conv(avg_out) # (B, 1, L)
        out = self.sigmoid(avg_out) # (B, 1, L)
        return x * out

class MCALayer(nn.Module):
    """MCA (Multi-scale Cross-dimensional Attention) Layer."""
    def __init__(self, inp, no_spatial=True):
        """Constructs a MCA module.
        Args:
            inp: Number of channels of the input feature maps
            no_spatial: whether to build channel dimension interactions
        """
        super(MCALayer, self).__init__()
        self.no_spatial = no_spatial

        lambd = 1.5
        gamma = 1
        temp = round(abs((math.log2(inp) - gamma) / lambd))
        kernel = temp if temp % 2 else temp - 1

        self.h_cw = MCAGate(3) # Attention along C and W for H
        self.w_hc = MCAGate(3) # Attention along H and C for W
        if not no_spatial:
            self.c_hw = MCAGate(kernel) # Attention along H and W for C

    def forward(self, x):
        B, C, H, W = x.size()

        # Attention along Height (H): Permute to (B, H, C, W), apply gate to C*W dim, permute back
        x_h = x.permute(0, 2, 1, 3).contiguous().view(B, H, C * W) # (B, H, C*W)
        x_h = self.h_cw(x_h) # Apply attention
        x_h = x_h.view(B, H, C, W).permute(0, 2, 1, 3).contiguous() # (B, C, H, W)

        # Attention along Width (W): Permute to (B, W, H, C), apply gate to H*C dim, permute back
        x_w = x.permute(0, 3, 2, 1).contiguous().view(B, W, H * C) # (B, W, H*C)
        x_w = self.w_hc(x_w) # Apply attention
        x_w = x_w.view(B, W, H, C).permute(0, 3, 2, 1).contiguous() # (B, C, H, W)

        if not self.no_spatial:
            # Attention along Channel (C): Permute to (B, C, H*W), apply gate to H*W dim, permute back
            x_c = x.view(B, C, H * W) # (B, C, H*W)
            x_c = self.c_hw(x_c) # Apply attention
            x_c = x_c.view(B, C, H, W) # (B, C, H, W)
            x_out = 1 / 3 * (x_c + x_h + x_w)
        else:
            x_out = 1 / 2 * (x_h + x_w)

        return x_out

# --- BiFPN Components ---
class BiFPNAdd(nn.Module):
    """BiFPN Add layer with learnable weights."""
    def __init__(self, n, epsilon=1e-4):
        super(BiFPNAdd, self).__init__()
        self.n = n
        self.epsilon = epsilon
        # Initialize weights for each input feature map
        self.w = nn.Parameter(torch.ones(n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        # x is a list of feature maps, e.g., [x0, x1, x2]
        if self.n == 1:
            return x[0]
        
        # Softmax normalization for weights
        w = self.w
        weight = torch.softmax(w, dim=0) # Shape: (n,)

        # Weighted sum of features
        weighted_features = [weight[i] * x[i] for i in range(self.n)]
        out = sum(weighted_features)
        
        return out

# --- BMFPN Module ---
class BMFPN(nn.Module):
    """
    BMFPN (BiFPN + MCA) module.
    Combines weighted feature fusion from BiFPN with multi-dimensional attention from MCA.
    Designed to replace Concat operations in the neck of YOLO architectures.
    """
    def __init__(self, in_channels_list, out_channels, n_inputs, epsilon=1e-4):
        """
        Args:
            in_channels_list (list): List of input channel dimensions for each feature map to be fused.
            out_channels (int): Number of output channels after fusion and processing.
            n_inputs (int): Number of input feature maps to be fused (e.g., 2 or 3).
            epsilon (float): Epsilon for numerical stability in weight normalization.
        """
        super(BMFPN, self).__init__()
        self.n_inputs = n_inputs
        self.out_channels = out_channels
        self.epsilon = epsilon

        # Ensure all input channels are mapped to the same number of channels before fusion
        self.adjust_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1, bias=False) if ch != out_channels else nn.Identity()
            for ch in in_channels_list
        ])
        # BatchNorm after each conv
        self.adjust_bns = nn.ModuleList([
            nn.BatchNorm2d(out_channels) if ch != out_channels else nn.Identity()
            for ch in in_channels_list
        ])

        # BiFPN-style weighted addition layer
        self.weighted_add = BiFPNAdd(n_inputs, epsilon)

        # MCA attention layer applied after fusion
        self.mca_attention = MCALayer(out_channels, no_spatial=False)

        # Optional: Final convolution to refine the fused features
        self.refine_conv = nn.Conv2d(out_channels, out_channels, 3, padding=autopad(3), bias=False)
        self.refine_bn = nn.BatchNorm2d(out_channels)
        self.refine_act = nn.SiLU() # Or nn.ReLU(), depending on YOLO11's default

    def forward(self, inputs):
        """
        Args:
            inputs: A list of feature maps, e.g., [feat_small, feat_medium, feat_large]
                    The order should correspond to the order in in_channels_list.
        """
        if len(inputs) != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} inputs, got {len(inputs)}")

        # Adjust channels of each input feature map
        adjusted_features = []
        for i, (x, conv, bn) in enumerate(zip(inputs, self.adjust_convs, self.adjust_bns)):
            x_adj = bn(conv(x))
            # Interpolate if spatial sizes differ (optional, depends on specific neck structure)
            # if x_adj.shape[-2:] != inputs[0].shape[-2:]: # Example: match to first feature map's size
            #     x_adj = F.interpolate(x_adj, size=inputs[0].shape[-2:], mode='nearest')
            adjusted_features.append(x_adj)

        # Apply weighted addition (BiFPN mechanism)
        fused_feature = self.weighted_add(adjusted_features)

        # Apply MCA attention (MCA mechanism)
        attended_feature = self.mca_attention(fused_feature)

        # Refine the fused and attended feature
        out = self.refine_bn(self.refine_conv(attended_feature))
        out = self.refine_act(out)

        return out

# --- Alternative: BMFPN_Concat2/3 style modules (if you prefer replacing Concat directly) ---
# These are simpler and follow the exact style of your BiFPN_Concat2/3 examples.

class BMFPN_Concat2(nn.Module):
    """BMFPN equivalent for fusing 2 features, replacing Concat."""
    def __init__(self, in_channels_list, out_channels, dimension=1):
        super(BMFPN_Concat2, self).__init__()
        self.d = dimension # Concatenation dimension (usually 1 for channels)
        self.out_channels = out_channels

        # Channel adjustment layers - adjust each input to out_channels
        self.adjust_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1, bias=False) if ch != out_channels else nn.Identity()
            for ch in in_channels_list
        ])
        self.adjust_bns = nn.ModuleList([
            nn.BatchNorm2d(out_channels) if ch != out_channels else nn.Identity()
            for ch in in_channels_list
        ])

        # BiFPN weights
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

        # MCA attention - after concatenation, input will be out_channels * 2
        self.mca_attention = MCALayer(out_channels * 2, no_spatial=False) 

        # Final conv to bring back to desired output channels
        self.final_conv = nn.Conv2d(out_channels * 2, out_channels, 1, bias=False)
        self.final_bn = nn.BatchNorm2d(out_channels)
        self.final_act = nn.SiLU()

    def forward(self, x):
        if len(x) != 2:
             raise ValueError(f"BMFPN_Concat2 expects 2 inputs, got {len(x)}")

        # Adjust channels
        x1 = self.adjust_bns[0](self.adjust_convs[0](x[0]))
        x2 = self.adjust_bns[1](self.adjust_convs[1](x[1]))

        # Apply BiFPN weights before concat
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        x1_weighted = weight[0] * x1
        x2_weighted = weight[1] * x2

        # Concatenate
        x_cat = torch.cat([x1_weighted, x2_weighted], dim=self.d) # dim=self.d is usually 1

        # Apply MCA attention
        x_attended = self.mca_attention(x_cat)

        # Reduce channels back to desired output
        out = self.final_bn(self.final_conv(x_attended))
        out = self.final_act(out)

        return out

class BMFPN_Concat3(nn.Module):
    """BMFPN equivalent for fusing 3 features, replacing Concat."""
    def __init__(self, in_channels_list, out_channels, dimension=1):
        super(BMFPN_Concat3, self).__init__()
        self.d = dimension
        self.out_channels = out_channels

        # Channel adjustment layers - adjust each input to out_channels
        self.adjust_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1, bias=False) if ch != out_channels else nn.Identity()
            for ch in in_channels_list
        ])
        self.adjust_bns = nn.ModuleList([
            nn.BatchNorm2d(out_channels) if ch != out_channels else nn.Identity()
            for ch in in_channels_list
        ])

        # BiFPN weights
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

        # MCA attention - after concatenation, input will be out_channels * 3
        self.mca_attention = MCALayer(out_channels * 3, no_spatial=False) 

        # Final conv to bring back to desired output channels
        self.final_conv = nn.Conv2d(out_channels * 3, out_channels, 1, bias=False)
        self.final_bn = nn.BatchNorm2d(out_channels)
        self.final_act = nn.SiLU()

    def forward(self, x):
        if len(x) != 3:
             raise ValueError(f"BMFPN_Concat3 expects 3 inputs, got {len(x)}")

        # Adjust channels
        x1 = self.adjust_bns[0](self.adjust_convs[0](x[0]))
        x2 = self.adjust_bns[1](self.adjust_convs[1](x[1]))
        x3 = self.adjust_bns[2](self.adjust_convs[2](x[2]))

        # Apply BiFPN weights before concat
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        x1_weighted = weight[0] * x1
        x2_weighted = weight[1] * x2
        x3_weighted = weight[2] * x3

        # Concatenate
        x_cat = torch.cat([x1_weighted, x2_weighted, x3_weighted], dim=self.d) # dim=self.d is usually 1

        # Apply MCA attention
        x_attended = self.mca_attention(x_cat)

        # Reduce channels back to desired output
        out = self.final_bn(self.final_conv(x_attended))
        out = self.final_act(out)

        return out