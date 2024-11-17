import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from model.ctrgcn import Model
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, dim_feat, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_feat // num_heads

        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim_feat * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim_feat, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = self.forward_attention(q, k, v)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_attention(self, q, k, v):
        B, _, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C * self.num_heads)
        return x

class Transformer_block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=1., mlp_out_ratio=1.,
                 qkv_bias=True, drop=0.1, attn_drop=0.1, act_layer=nn.ReLU,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.attn = Attention(dim, dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_out_dim = int(dim * mlp_out_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       out_features=mlp_out_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class Fusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.maxpooling = nn.AdaptiveMaxPool1d(1)

    def forward(self, xt, xs):

        xt = self.maxpooling(xt.permute(0, 2, 1)).permute(0, 2, 1)
        xs = self.maxpooling(xs.permute(0, 2, 1)).permute(0, 2, 1)

        output = torch.cat([xt, xs], dim=1)

        return output


class STEncoder(nn.Module):
    """Two branch MG encoder"""

    def __init__(self, t_input_size, s_input_size, hidden_size, num_head) -> None:
        super().__init__()
        self.d_model = hidden_size
        self.gcn = Model()

        # temporal and spatial branch embedding layers
        self.t_embedding = nn.Sequential(
            nn.Linear(t_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
        )
        self.s_embedding = nn.Sequential(
            nn.Linear(s_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
        )

        self.s_encoder = Transformer_block(hidden_size, num_head)
        self.t_encoder = Transformer_block(hidden_size, num_head)

    def forward(self, x):
        # N, C, T, V, M

        xt, xs = self.gcn(x)

        xt = self.t_embedding(xt)  # temporal domain
        xs = self.s_embedding(xs)  # spatial domain

        vt = self.t_encoder(xt)
        vs = self.s_encoder(xs)

        vt = vt.amax(dim=1)
        vs = vs.amax(dim=1)

        return vt, vs


class PretrainingEncoder(nn.Module):
    """multi_granularity network + projectors"""

    def __init__(self, t_input_size, s_input_size, hidden_size, num_head, num_class=60):
        super(PretrainingEncoder, self).__init__()

        self.d_model = hidden_size

        self.st_encoder = STEncoder(t_input_size, s_input_size, hidden_size, num_head)

        # temporal domain level feature projector
        self.t_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

        # spatial domain level feature projector
        self.s_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

    def forward(self, x):

        vt, vs = self.st_encoder(x)

        # projection
        zs = self.s_proj(vs)
        zt = self.t_proj(vt)

        return zt, zs


class DownstreamEncoder(nn.Module):
    """multi_granularity network + classifier"""

    def __init__(self, t_input_size, s_input_size, hidden_size, num_head, num_class=60):
        super(DownstreamEncoder, self).__init__()

        self.d_model = hidden_size

        self.st_encoder = STEncoder(t_input_size, s_input_size, hidden_size, num_head)

        # linear classifier
        self.fc = nn.Linear(2 * self.d_model, num_class)

    def forward(self, x, knn_eval=False):

        vt, vs = self.st_encoder(x)

        v = torch.cat([vt, vs], dim=1)

        if knn_eval:  # return last layer features during  KNN evaluation (action retrieval)
            return v
        else:
            return self.fc(v)
