# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from .attention import flash_attention
from torch.utils.checkpoint import checkpoint
from ovi.distributed_comms.communications import all_gather, all_to_all_4D
from ovi.distributed_comms.parallel_states import nccl_info, get_sequence_parallel_state


def gradient_checkpointing(module: nn.Module, *args, enabled: bool, **kwargs):
    if enabled:
        return checkpoint(module, *args, use_reentrant=False, **kwargs)
    else:
        return module(*args, **kwargs)


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000, freqs_scaling=1.0):
    assert dim % 2 == 0
    pos =  torch.arange(max_seq_len)
    freqs = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    freqs = freqs_scaling * freqs
    freqs = torch.outer(pos, freqs)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

@torch.amp.autocast('cuda', enabled=False)
def rope_apply_1d(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2 ## b l h d
    c_rope = freqs.shape[1]  # number of complex dims to rotate
    assert c_rope <= c, "RoPE dimensions cannot exceed half of hidden size"
    
    # loop over samples
    output = []
    for i, (l, ) in enumerate(grid_sizes.tolist()):
        seq_len = l
        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2)) # [l n d//2]
        x_i_rope = x_i[:, :, :c_rope] * freqs[:seq_len, None, :]  # [L, N, c_rope]
        x_i_passthrough = x_i[:, :, c_rope:]  # untouched dims
        x_i = torch.cat([x_i_rope, x_i_passthrough], dim=2)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).bfloat16()

@torch.amp.autocast('cuda', enabled=False)
def rope_apply_3d(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    
    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).bfloat16()

@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    x_ndim = grid_sizes.shape[-1]
    if x_ndim == 3:
        return rope_apply_3d(x, grid_sizes, freqs)
    else:
        return rope_apply_1d(x, grid_sizes, freqs)

class ChannelLastConv1d(nn.Conv1d):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x


class ConvMLP(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ChannelLastConv1d(dim,
                                    hidden_dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)
        self.w2 = ChannelLastConv1d(hidden_dim,
                                    dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)
        self.w3 = ChannelLastConv1d(dim,
                                    hidden_dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.bfloat16()).type_as(x) * self.weight.bfloat16()

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.bfloat16()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        # optional sequence parallelism
        # self.world_size = get_world_size()
        self.use_sp = get_sequence_parallel_state()
        if self.use_sp:
            self.sp_size = nccl_info.sp_size
            self.sp_rank = nccl_info.rank_within_group
            assert self.num_heads % self.sp_size == 0, \
                f"Num heads {self.num_heads} must be divisible by sp_size {self.sp_size}"
    # query, key, value function
    def qkv_fn(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        q, k, v = self.qkv_fn(x)
        if self.use_sp:
            # print(f"[DEBUG SP] Doing all to all to shard head")
            q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = all_to_all_4D(k, scatter_dim=2, gather_dim=1)
            v = all_to_all_4D(v, scatter_dim=2, gather_dim=1) # [B, L, H/P, C/H]
        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)
        if self.use_sp: 
            # print(f"[DEBUG SP] Doing all to all to shard sequence")
            x = all_to_all_4D(x, scatter_dim=1, gather_dim=2) # [B, L/P, H, C/H]
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    def qkv_fn(self, x, context):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        return q, k, v

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        q, k, v = self.qkv_fn(x, context)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 additional_emb_length=None):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.additional_emb_length = additional_emb_length

    def qkv_fn(self, x, context):
        context_img = context[:, : self.additional_emb_length]
        context = context[:, self.additional_emb_length :]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)

        return q, k, v, k_img, v_img


    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        q, k, v, k_img, v_img = self.qkv_fn(x, context)

        if self.use_sp:
            # print(f"[DEBUG SP] Doing all to all to shard head")
            q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)  
            k = torch.chunk(k, self.sp_size, dim=2)[self.sp_rank]
            v = torch.chunk(v, self.sp_size, dim=2)[self.sp_rank]
            k_img = torch.chunk(k_img, self.sp_size, dim=2)[self.sp_rank]
            v_img = torch.chunk(v_img, self.sp_size, dim=2)[self.sp_rank]
            
        # [B, L, H/P, C/H]
        # k_img: [B, L, H, C/H]
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)
        if self.use_sp: 
            # print(f"[DEBUG SP] Doing all to all to shard sequence")
            x = all_to_all_4D(x, scatter_dim=1, gather_dim=2) # [B, L/P, H, C/H]
            
        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}

class ModulationAdd(nn.Module):
    def __init__(self, dim, num):
        super().__init__()
        self.modulation = nn.Parameter(torch.randn(1, num, dim) / dim**0.5)

    def forward(self, e):
        return self.modulation.bfloat16() + e.bfloat16()

class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 additional_emb_length=None):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        if cross_attn_type == 'i2v_cross_attn':
            assert additional_emb_length is not None, "additional_emb_length should be specified for i2v_cross_attn"
            self.cross_attn = WanI2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps, additional_emb_length)
        else:
            assert additional_emb_length is None, "additional_emb_length should be None for t2v_cross_attn"
            self.cross_attn = WanT2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps, )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        # self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        # self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.modulation = ModulationAdd(dim, 6)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.bfloat16
        assert len(e.shape) == 4 and e.size(2) == 6 and e.shape[1] == x.shape[1], f"{e.shape}, {x.shape}"
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            e = self.modulation(e).chunk(6, dim=2)
        assert e[0].dtype == torch.bfloat16

        # self-attention
        y = self.self_attn(
            self.norm1(x).bfloat16() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens,
            grid_sizes,
            freqs)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            x = x + y * e[2].squeeze(2)

        # cross-attention & ffn
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(
                self.norm2(x).bfloat16() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)

        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L, C]
        """
        assert e.dtype == torch.bfloat16
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            e = (self.modulation.bfloat16().unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=3)
        
        # head
        x = self.head(
            self.norm(x).bfloat16() * (1 + e[1].squeeze(3)) + e[0].squeeze(3))

        return x

    def unravel(self, x, grid_sizes):
        r"""
        Args:
            x(Tensor): Shape [B, L, C_out * prod(patch_size)]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
        Returns:
            list of tensors, each has shape [C_out, F * patch_t, H * patch_h, W * patch_w]
            or [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            # v is [F H w] F * H * 80, 100, it was right padded by 20. 
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        # out is list of [C F H W]
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        if self.is_video_type:
            assert isinstance(self.patch_embedding, nn.Conv3d), f"Patch embedding for video should be a Conv3d layer, got {type(self.patch_embedding)}"\
            nn.init.xavier_uniform_(self.patch_embedding.weight)
            if self.patch_embedding.bias is not None:
                nn.init.zeros_(self.patch_embedding.bias)

        # zero out the head
        if self.head is not None:
            nn.init.zeros_(self.head.weight)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)


class WanTransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        cross_attn_type='t2v_cross_attn',
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        additional_emb_length=None
    ):
        super().__init__()

        self.attn_block = WanAttentionBlock(
            cross_attn_type=cross_attn_type,
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            additional_emb_length=additional_emb_length
        )

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context=None,
        context_lens=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            context(Tensor): Shape [B, L1, C], context embedding for cross attention
            context_lens(Tensor): Shape [B], context lengths for cross attention
        """

        x = self.attn_block(
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
        )

        return x

class WanTransformer(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        dim=768,
        num_layers=12,
        num_heads=12,
        ffn_dim=3072,
        cross_attn_type='t2v_cross_attn',
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        is_video_type=False,
        patch_size=(1, 1, 1),
        additional_emb_length=None
    ):
        super().__init__()
        self.is_video_type = is_video_type
        if is_video_type:
            # Patch embedding for video data
            assert patch_size == (1, 2, 2)
            self.patch_embedding = nn.Conv3d(4, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        else:
            # Patch embedding for image data
            assert patch_size == (1, 1, 1)
            self.patch_embedding = None

        # Transformer blocks
        self.layers = nn.ModuleList([
            WanTransformerBlock(
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                cross_attn_type=cross_attn_type,
                qk_norm=qk_norm,
                cross_attn_norm=cross_attn_norm,
                eps=eps,
                additional_emb_length=additional_emb_length,
            ) for _ in range(num_layers)
        ])
        
        # Head
        self.head = Head(dim, 4, patch_size, eps=eps)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context=None,
        context_lens=None,
        grad_checkpointing=False,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            context(Tensor): Shape [B, L1, C], context embedding for cross attention
            context_lens(Tensor): Shape [B], context lengths for cross attention
        """
        if self.is_video_type:
            x = self.patch_embedding(x).flatten(2).transpose(1, 2)

        for layer in self.layers:
            x = gradient_checkpointing(
                layer,
                x=x,
                e=e,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=freqs,
                context=context,
                context_lens=context_lens,
                enabled=grad_checkpointing,
            )

        x = self.head(x, e)
        x = self.head.unravel(x, grid_sizes)
        return x