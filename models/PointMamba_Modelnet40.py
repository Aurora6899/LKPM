import torch
import ocnn
import dwconv
from ocnn.octree import Octree
from typing import Optional, List
from torch.utils.checkpoint import checkpoint
from typing import Optional
import torch
from torch import nn
from torch.cuda.amp import autocast
import copy
from mamba_ssm import Mamba

class OctreeT(Octree):

    def __init__(self, octree: Octree, patch_size: int = 24, dilation: int = 4,
                 nempty: bool = True, max_depth: Optional[int] = None,
                 start_depth: Optional[int] = None, **kwargs):
        super().__init__(octree.depth, octree.full_depth)
        self.__dict__.update(octree.__dict__)

        self.nempty = nempty
        self.max_depth = max_depth or self.depth
        self.start_depth = start_depth or self.full_depth
        self.invalid_mask_value = -1e3
        assert self.start_depth > 1

        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError(f"Invalid patch_size value: {patch_size}. It must be a positive integer.")
        if not isinstance(dilation, int) or dilation <= 0:
            raise ValueError(f"Invalid dilation value: {dilation}. It must be a positive integer.")

        self.block_num = patch_size * dilation
        self.nnum_t = self.nnum_nempty.float() if nempty else self.nnum.float()


        self.nnum_a = ((self.nnum_t / self.block_num).ceil() * self.block_num).int()


        num = self.max_depth + 1
        self.batch_idx = [None] * num
        self.build_t()

    def build_t(self):
        for d in range(self.start_depth, self.max_depth + 1):
            self.build_batch_idx(d)

    def build_batch_idx(self, depth: int):
        batch = self.batch_id(depth, self.nempty)
        self.batch_idx[depth] = self.patch_partition(batch, depth)

    def patch_partition(self, data: torch.Tensor, depth: int, fill_value=0):
        num = self.nnum_a[depth] - self.nnum_t[depth].int()

        if num < 0:
            raise ValueError(
                f"Invalid num value: {num}. self.nnum_a[{depth}]: {self.nnum_a[depth]}, self.nnum_t[{depth}]: {self.nnum_t[depth].int()}")
        if num == 0:
            num = 1
        tail = data.new_full((num,) + data.shape[1:], fill_value)
        return tail
    def get_key_from_octree(self, depth: int):
        if self.nempty:
            keys_at_depth = self.keys[depth][self.nempty_mask(depth)]
        else:
            keys_at_depth = self.keys[depth]
        return keys_at_depth



class MLP(torch.nn.Module):

    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, activation=torch.nn.GELU,
                 drop: float = 0.0, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features

        self.fc1 = torch.nn.Linear(self.in_features, self.hidden_features)
        self.act = activation()
        self.fc2 = torch.nn.Linear(self.hidden_features, self.out_features)
        self.drop = torch.nn.Dropout(drop, inplace=True)

    def forward(self, data: torch.Tensor):
        data = self.fc1(data)
        data = self.act(data)
        data = self.drop(data)
        data = self.fc2(data)
        data = self.drop(data)
        return data

class OctreeDWConvBn(torch.nn.Module):

    def __init__(self, in_channels: int, kernel_size: List[int] = [3],
                 stride: int = 1, nempty: bool = False):
        super().__init__()
        self.conv = dwconv.OctreeDWConv(
            in_channels, kernel_size, nempty, use_bias=False)
        self.bn = torch.nn.BatchNorm1d(in_channels)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.conv(data, octree, depth)
        out = self.bn(out)
        return out

class RPE(torch.nn.Module):

    def __init__(self, patch_size: int, num_heads: int, dilation: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dilation = dilation
        self.pos_bnd = self.get_pos_bnd(patch_size)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def get_pos_bnd(self, patch_size: int):
        return int(0.8 * patch_size * self.dilation ** 0.5)

    def xyz2idx(self, xyz: torch.Tensor):
        mul = torch.arange(3, device=xyz.device) * self.rpe_num
        xyz = xyz.clamp(-self.pos_bnd, self.pos_bnd)
        idx = xyz + (self.pos_bnd + mul)
        return idx

    def forward(self, xyz):
        idx = self.xyz2idx(xyz)
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out

    def extra_repr(self) -> str:
        return 'num_heads={}, pos_bnd={}, dilation={}'.format(
            self.num_heads, self.pos_bnd, self.dilation)  # noqa

    def extra_repr(self) -> str:
        return 'dim={}, patch_size={}, num_heads={}, dilation={}'.format(
            self.dim)  # noqa


class PointMambaBlock(torch.nn.Module):
    def __init__(self, dim: int,
                 proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
                 activation: torch.nn.Module = torch.nn.GELU, **kwargs):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.mamba = OctreeMamba(dim, proj_drop)
        self.drop_path = ocnn.nn.OctreeDropPath(drop_path, nempty)
        self.cpe = OctreeDWConvBn(dim, nempty=nempty)

    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        data = self.cpe(data, octree, depth) + data
        attn = self.mamba(self.norm1(data), octree, depth)
        data = data + self.drop_path(attn, octree, depth)
        return data

class PointMambaStage(torch.nn.Module):

    def __init__(self, dim: int,
                 proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
                 activation: torch.nn.Module = torch.nn.GELU, interval: int = 6,
                 use_checkpoint: bool = True, num_blocks: int = 2,
                 pim_block=PointMambaBlock, **kwargs):
        super().__init__()
        self.num_blocks = num_blocks
        self.use_checkpoint = use_checkpoint
        self.interval = interval  # normalization interval
        self.num_norms = (num_blocks - 1) // self.interval

        self.blocks = torch.nn.ModuleList([pim_block(
            dim=dim
            , proj_drop=proj_drop,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            nempty=nempty, activation=activation) for i in range(num_blocks)])


    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        for i in range(self.num_blocks):
            if self.use_checkpoint and self.training:
                data = checkpoint(self.blocks[i], data, octree, depth)
            else:
                data = self.blocks[i](data, octree, depth)

        return data


class PatchEmbed(torch.nn.Module):

    def __init__(self, in_channels: int = 3, dim: int = 96, num_down: int = 2,
                 nempty: bool = True, **kwargs):
        super().__init__()
        self.num_stages = num_down
        self.delta_depth = -num_down
        channels = [int(dim * 2 ** i) for i in range(-self.num_stages, 1)]

        self.convs = torch.nn.ModuleList([ocnn.modules.OctreeConvBnRelu(
            in_channels if i == 0 else channels[i], channels[i], kernel_size=[3],
            stride=1, nempty=nempty) for i in range(self.num_stages)])
        self.downsamples = torch.nn.ModuleList([ocnn.modules.OctreeConvBnRelu(
            channels[i], channels[i + 1], kernel_size=[2], stride=2, nempty=nempty)
            for i in range(self.num_stages)])
        self.proj = ocnn.modules.OctreeConvBnRelu(
            channels[-1], dim, kernel_size=[3], stride=1, nempty=nempty)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        for i in range(self.num_stages):
            depth_i = depth - i
            data = self.convs[i](data, octree, depth_i)
            data = self.downsamples[i](data, octree, depth_i)
        data = self.proj(data, octree, depth_i - 1)
        return data


class Downsample(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: List[int] = [2], nempty: bool = True):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(out_channels)
        self.conv = ocnn.nn.OctreeConv(in_channels, out_channels, kernel_size,
                                       stride=2, nempty=nempty, use_bias=True)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        data = self.conv(data, octree, depth)
        data = self.norm(data)
        return data


class PointMamba_m40(torch.nn.Module):

    def __init__(self, in_channels: int,
                 channels: List[int] = [192],
                 num_blocks: List[int] = [2],
                 drop_path: float = 0.5,
                 nempty: bool = True, stem_down: int = 2, **kwargs):
        super().__init__()
        self.nempty = nempty
        self.num_stages = len(num_blocks)
        self.stem_down = stem_down
        drop_ratio = torch.linspace(0, drop_path, sum(num_blocks)).tolist()

        self.patch_embed = PatchEmbed(in_channels, channels[0], stem_down, nempty)
        self.layers = torch.nn.ModuleList([PointMambaStage(
            dim=channels[i],
            drop_path=drop_ratio[sum(num_blocks[:i]):sum(num_blocks[:i + 1])],
            nempty=nempty, num_blocks=num_blocks[i], )
            for i in range(self.num_stages)])
        self.downsamples = torch.nn.ModuleList([Downsample(
            channels[i], channels[i + 1], kernel_size=[2],
            nempty=nempty) for i in range(self.num_stages - 1)])

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        data = self.patch_embed(data, octree, depth)
        depth = depth - self.stem_down  # current octree depth
        octree = OctreeT(octree, patch_size=24, dilation=4, nempty=self.nempty,
                         max_depth=depth, start_depth=depth - self.num_stages + 1)
        features = {}
        for i in range(self.num_stages):
            depth_i = depth - i
            data = self.layers[i](data, octree, depth_i)
            features[depth_i] = data

            if i < self.num_stages - 1:
                data = self.downsamples[i](data, octree, depth_i)


        return features

class OctreeMamba(torch.nn.Module):
    def __init__(self, dim: int, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim

        self.bi_pixel_mamba = BiPixelMambaLayer(dim=dim)
        self.bi_window_mamba = BiWindowMambaLayer(dim=dim)

        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        N, C = data.shape
        B = 1
        L = N
        data = data.unsqueeze(0).permute(0, 2, 1)  # [1, C, N]
        data = self.bi_pixel_mamba(data,depth)  # [B, C, L]
        data = self.bi_window_mamba(data,depth)  # [B, C, L]
        data = data.permute(0, 2, 1).squeeze(0)  # [N, C]
        data = self.proj(data)
        data = self.proj_drop(data)
        return data

class BiPixelMambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba_forw = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=True,
            bimamba_type="v2",
        )

    @autocast(enabled=False)
    def forward(self, x,depth):
        # x: [B, C, L]
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C, L = x.shape
        assert C == self.dim

        if depth == 5:
            patch_size = 64
        elif depth == 4:
            patch_size = 64
        elif depth == 3:
            patch_size = 64
        else:
            raise ValueError(f"Unsupported depth value: {depth}")

        if L % patch_size != 0:
            pad_len = patch_size - (L % patch_size)
            start_idx = L - (L % patch_size) - pad_len
            start_idx = max(start_idx, 0)
            borrowed_points = x[:, :, start_idx:start_idx + pad_len]
            x = torch.cat([x, borrowed_points], dim=2)
            L_padded = L + pad_len
        else:
            L_padded = L

        num_patches = L_padded // patch_size
        x_div = x.reshape(B, C, num_patches, patch_size)
        x_div = x_div.permute(0, 3, 1, 2).contiguous()
        x_div = x_div.view(B * patch_size, C, num_patches)

        x_flat = x_div.transpose(1, 2)  # [NB, n_tokens, C]
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba_forw(x_norm)
        x_out = x_mamba

        x_out = x_out.reshape(B, patch_size, num_patches, C)
        x_out = x_out.permute(0, 3, 1, 2).contiguous()
        x_out = x_out.reshape(B, C, L_padded)

        if L_padded != L:
            x_out = x_out[:, :, :L]
        out = x_out + x[:, :, :L]
        return out

class BiWindowMambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        self.mamba_forw = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=True,
            bimamba_type="v2",
        )

    @autocast(enabled=False)
    def forward(self, x,depth):
        # x: [B, C, L]
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        B, C, L = x.shape
        assert C == self.dim

        if depth == 5:
            patch_size = 64
        elif depth == 4:
            patch_size = 64
        elif depth == 3:
            patch_size = 64
        else:
            raise ValueError(f"Unsupported depth value: {depth}")

        if L % patch_size != 0:
            pad_len = patch_size - (L % patch_size)
            start_idx = L - (L % patch_size) - pad_len
            start_idx = max(start_idx, 0)
            borrowed_points = x[:, :, start_idx:start_idx + pad_len]
            x = torch.cat([x, borrowed_points], dim=2)
            L_padded = L + pad_len
        else:
            L_padded = L


        pool_layer = nn.AvgPool1d(kernel_size=patch_size, stride=patch_size)
        x_div = pool_layer(x)  # [B, C, L_padded // p]

        x_flat = x_div.transpose(1, 2)  # [B, n_tokens, C]
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba_forw(x_norm)
        x_out= x_mamba

        unpool_layer = nn.Upsample(scale_factor=patch_size, mode='nearest')
        x_out = x_out.transpose(1, 2)  # [B, C, n_tokens]
        x_out = unpool_layer(x_out)
        x_out = x_out[:, :, :L_padded]

        if L_padded != L:
            x_out = x_out[:, :, :L]
        out = x_out + x[:, :, :L]
        return out
