import torch
import torch.nn as nn


class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation in MogaNet."""

    def __init__(self, embed_dims, mlp_hidden_dims, kernel_size=3, act_layer=nn.GELU, ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()
        self.embed_dims = embed_dims
        self.mlp_hidden_dims = mlp_hidden_dims

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=self.mlp_hidden_dims, kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.mlp_hidden_dims, out_channels=self.mlp_hidden_dims, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=True, groups=self.mlp_hidden_dims)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(
            in_channels=mlp_hidden_dims, out_channels=embed_dims, kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.mlp_hidden_dims, out_channels=1, kernel_size=1)
        self.sigma = nn.Parameter(
            1e-5 * torch.ones((1, mlp_hidden_dims, 1, 1)), requires_grad=True)
        self.decompose_act = act_layer()

    def feat_decompose(self, x):
        x = x + self.sigma * (x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
