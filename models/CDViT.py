"""
Modifed from Timm. https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, Mlp, Block


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim

        # self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.reduction = nn.ModuleList()
        for d in range(len(dim)):
            tmp = []
            tmp.append(nn.Linear(4 * dim[d], 2 * dim[d], bias=False))
            self.reduction.append(nn.Sequential(*tmp))

        # self.up = nn.Linear(dim, 2 * dim, bias=False)
        self.up = nn.ModuleList()
        for d in range(len(dim)):
            tmp = []
            tmp.append(nn.Linear(dim[d], 2 * dim[d], bias=False))
            self.up.append(nn.Sequential(*tmp))

        # self.norm = norm_layer(4 * dim)
        self.norm = nn.ModuleList()
        for d in range(len(dim)):
            tmp = []
            tmp.append(norm_layer(4 * dim[d]))
            self.norm.append(nn.Sequential(*tmp))

    def forward(self, x):
        """
        x: B, H*W, C
        """
        outs = []
        for i in range(len(self.dim)):
            tmp = x[i]
            tokens = tmp[:, 0:1, :]
            tmp = tmp[:, 1:, :]
            H, W = self.input_resolution[i], self.input_resolution[i]
            B, L, D = tmp.shape
            assert L == H * W, "input feature has wrong size"
            assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
            tmp = tmp.view(B, H, W, D)

            x0 = tmp[:, 0::2, 0::2, :]  # B H/2 W/2 C
            x1 = tmp[:, 1::2, 0::2, :]  # B H/2 W/2 C
            x2 = tmp[:, 0::2, 1::2, :]  # B H/2 W/2 C
            x3 = tmp[:, 1::2, 1::2, :]  # B H/2 W/2 C
            tmp = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
            tmp = tmp.view(B, -1, 4 * D)  # B H/2*W/2 4*C

            tmp = self.norm[i](tmp)
            tmp = self.reduction[i](tmp)
            tokens = self.up[i](tokens)
            tmp = torch.cat((tokens, tmp), dim=1)
            outs.append(tmp)

        return outs

    def extra_repr(self):
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, multi_conv=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if multi_conv:
            if patch_size[0] == 12:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=3, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
                )
            elif patch_size[0] == 16:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
                )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class fusionAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., selfornot=True):
        super().__init__()
        self.self = selfornot
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, n):
        B, N, C = x.shape

        if self.self:#  self-attention
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            q = self.wq(x[:, 0:n, ...]).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv = self.wkv(x[:, n:, ...]).reshape(B, N-n, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, n, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class fusionAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True, selfornot=True):
        super().__init__()
        self.selfornot = selfornot
        self.norm1 = norm_layer(dim)
        self.attn = fusionAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop=attn_drop, proj_drop=drop, selfornot=selfornot)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.fuse_attention2 = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1, bias=False), nn.BatchNorm2d(1), nn.ReLU(),
            nn.Conv2d(1, 1, 1)
        )
        self.fuse_attention3 = nn.Sequential(
            nn.Conv2d(3, 1, 3, padding=1, bias=False), nn.BatchNorm2d(1), nn.ReLU(),
            nn.Conv2d(1, 1, 1)
        )


    def forward(self, x, n):
        # cls token passing
        for i in range(len(x)):
            if i == 0:
                tmp = x[i][:, 0:1, ...]
                tok = x[i][:, 1:, ...]
            else:
                tmp = torch.cat((tmp, x[i][:, 0:1, ...]), 1)
                tok = torch.cat((tok, x[i][:, 1:, ...]), 1)
        tmp = tmp.unsqueeze(0).permute(0, 2, 1, 3)
        if len(x) == 2:
            fuse_attention = torch.sigmoid(self.fuse_attention2(tmp)).permute(0, 2, 1, 3).squeeze(0)
        elif len(x) == 3:
            fuse_attention = torch.sigmoid(self.fuse_attention3(tmp)).permute(0, 2, 1, 3).squeeze(0)
        x = torch.cat((fuse_attention, tok), 1)

        if self.selfornot:
            x = x + self.drop_path(self.attn(self.norm1(x), n))
        else:
            x = x[:, 0:n, ...] + self.drop_path(self.attn(self.norm1(x), n))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class MultiScaleBlock(nn.Module):

    def __init__(self, dim, patches, depth, num_heads, input_resolution, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, upsample=None, selfornot=True):
        super().__init__()
        num_branches = len(dim)
        self.num_branches = num_branches
        # different branch could have different embedding size, the first one is the base
        self.blocks0 = nn.ModuleList()
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(
                    Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                          drop=drop, attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer))
            if len(tmp) != 0:
                self.blocks0.append(nn.Sequential(*tmp))
        if len(self.blocks0) == 0:
            self.blocks0 = None

        self.blocks1 = nn.ModuleList()
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(
                    Block(dim=dim[d]*2, num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                          drop=drop, attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer))
            if len(tmp) != 0:
                self.blocks1.append(nn.Sequential(*tmp))
        if len(self.blocks1) == 0:
            self.blocks1 = None

        self.fusion = fusionAttentionBlock(
            dim=dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale,drop=drop,
            attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer, has_mlp=False, selfornot=selfornot)

        # patch merging layer
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None


    def forward(self, x):
        ###### cls token are passed between hierachical ViT blocks
        outs_b0 = []
        for index in range(self.num_branches):
            if index == 0:
                outs_b0.append(self.blocks0[index](x[index]))
            else:
                outs_b0.append(self.blocks0[index](torch.cat((outs_b0[index - 1][:, 0:1, ...], x[index][:, 1:, ...]), 1)))
        ######

        ###### Bi-directional integration in
        outs_tsr = []
        for i in range(self.num_branches):  # Tread-level -> Seasonal-level -> Residual-level
            _, n, _ = outs_b0[i].shape
            if i == 2:
                tokens_joint = outs_b0[i]
                tsr_fusion_out = self.blocks0[i](tokens_joint)[:, :n, ...]
            if i == 1:
                tokens_joint = [outs_b0[i], outs_b0[i + 1]]
                tsr_fusion_out = self.fusion(tokens_joint, n)[:, :n, ...]
            elif i == 0:
                tokens_joint = [outs_b0[i], outs_b0[i + 1], outs_b0[i + 2]]
                tsr_fusion_out = self.fusion(tokens_joint, n)[:, :n, ...]
            outs_tsr.append(tsr_fusion_out)

        outs_rst = []
        for i in range(self.num_branches):  # Residual-level -> Seasonal-level -> Tread-level
            _, n, _ = outs_tsr[i].shape
            if i == 0:
                tokens_joint = outs_tsr[i]
                rst_fusion_out = self.blocks0[i](tokens_joint)[:, :n, ...]
            if i == 1:
                tokens_joint = [outs_tsr[i], outs_tsr[i - 1]]
                rst_fusion_out = self.fusion(tokens_joint, n)[:, :n, ...]
            elif i == 2:
                tokens_joint = [outs_tsr[i], outs_tsr[i - 1], outs_tsr[i - 2]]
                rst_fusion_out = self.fusion(tokens_joint, n)[:, :n, ...]
            outs_rst.append(rst_fusion_out)
        ###### Bi-directional integration out
        outs = []
        for i in range(self.num_branches):
            outs.append(outs_b0[i] + outs_rst[i])

        if self.upsample is not None:
            outs = self.upsample(outs)

        # Residual-level out
        out = self.blocks1[2](outs[2])

        return out


def _compute_num_patches(img_size, patches):
    return [int(i // p) * int(i // p) for i, p in zip(img_size, patches)]


class _AttentionModule_cls(nn.Module):
    def __init__(self):
        super(_AttentionModule_cls, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(2, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU(),
            nn.Conv2d(2, 2, 3, dilation=2, padding=2, groups=1, bias=False), nn.BatchNorm2d(2), nn.ReLU(),
            nn.Conv2d(2, 2, 1, bias=False), nn.BatchNorm2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(2, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU(),
            nn.Conv2d(2, 2, 3, dilation=3, padding=3, groups=1, bias=False), nn.BatchNorm2d(2), nn.ReLU(),
            nn.Conv2d(2, 2, 1, bias=False), nn.BatchNorm2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(2, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU(),
            nn.Conv2d(2, 2, 3, dilation=4, padding=4, groups=1, bias=False), nn.BatchNorm2d(2), nn.ReLU(),
            nn.Conv2d(2, 1, 1, bias=False), nn.BatchNorm2d(1)
        )
        self.down = nn.Sequential(
            nn.Conv2d(2, 1, 1, bias=False), nn.BatchNorm2d(1)
        )

    def forward(self, x):
        block1 = F.relu(self.block1(x) + x, True)
        block2 = F.relu(self.block2(block1) + block1, True)
        block3 = torch.sigmoid(self.block3(block2) + self.down(block2))
        return block3
class cls_fusion(nn.Module):
    def __init__(self, embed_dim, num_classes, num_branches):
        super().__init__()
        self.num_branches = num_branches
        self.embed_dim = embed_dim[2]

        self.refine2_hl_cls = nn.Sequential(
            nn.Conv2d(2, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU(),
            nn.Conv2d(1, 1, 3, padding=1, groups=1, bias=False), nn.BatchNorm2d(1), nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False), nn.BatchNorm2d(1)
        )
        self.attention2_hl_cls = _AttentionModule_cls()

        self.refine1_hl_cls = nn.Sequential(
            nn.Conv2d(2, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU(),
            nn.Conv2d(1, 1, 3, padding=1, groups=1, bias=False), nn.BatchNorm2d(1), nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False), nn.BatchNorm2d(1)
        )
        self.attention1_hl_cls = _AttentionModule_cls()

        self.head = nn.Linear(self.embed_dim * 2, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):

        cls_bin = x[0].unsqueeze(0).unsqueeze(1)
        cls_sea = x[1].unsqueeze(0).unsqueeze(1)
        cls_win = x[2].unsqueeze(0).unsqueeze(1)

        refine2_hl_cls_0 = F.relu(self.refine2_hl_cls(torch.cat((cls_bin, cls_sea), 1)) + cls_bin, True)
        refine2_hl_cls_0 = (1 + self.attention2_hl_cls(torch.cat((cls_bin, cls_sea), 1))) * refine2_hl_cls_0
        refine2_hl_cls_1 = F.relu(self.refine2_hl_cls(torch.cat((refine2_hl_cls_0, cls_sea), 1)) + refine2_hl_cls_0, True)
        refine2_hl_cls_1 = (1 + self.attention2_hl_cls(torch.cat((refine2_hl_cls_0, cls_sea), 1))) * refine2_hl_cls_1

        refine1_hl_cls_0 = F.relu(self.refine1_hl_cls(torch.cat((refine2_hl_cls_1, cls_win), 1)) + refine2_hl_cls_1, True)
        refine1_hl_cls_0 = (1 + self.attention1_hl_cls(torch.cat((refine2_hl_cls_1, cls_win), 1))) * refine1_hl_cls_0
        refine1_hl_cls_1 = F.relu(self.refine1_hl_cls(torch.cat((refine1_hl_cls_0, cls_win), 1)) + refine1_hl_cls_0, True)
        refine1_hl_cls_1 = (1 + self.attention1_hl_cls(torch.cat((refine1_hl_cls_0, cls_win), 1))) * refine1_hl_cls_1

        x = self.head(refine1_hl_cls_1.squeeze(0).squeeze(0))

        return x
####

class CDViT(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(8, 16), in_chans=3, num_classes=1000, embed_dim=(192, 384),
                 depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]), num_heads=(6, 12), mlp_ratio=(2., 2., 4.), qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None,
                 norm_layer=nn.LayerNorm, multi_conv=False, clsFusion=False, selfornot=True):
        super().__init__()

        self.num_classes = num_classes
        if not isinstance(img_size, list):
            img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.clsFusion = clsFusion

        num_patches = _compute_num_patches(img_size, patch_size)
        self.num_branches = len(patch_size)

        self.patch_embed = nn.ModuleList()
        self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])
        for im_s, p, d in zip(img_size, patch_size, embed_dim):
            self.patch_embed.append(PatchEmbed(img_size=im_s, patch_size=p, in_chans=in_chans, embed_dim=d, multi_conv=multi_conv))

        self.cls_token = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, embed_dim[i])) for i in range(self.num_branches)])
        self.pos_drop = nn.Dropout(p=drop_rate)

        # MultiScaleBlock
        input_resolution = [int(math.sqrt(num_patches[i])) for i in range(len(num_patches))]
        total_depth = sum([sum(x[-2:]) for x in depth])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]  # stochastic depth decay rule
        dpr_ptr = 0
        self.MultiScaleBlock = nn.ModuleList()
        for idx, block_cfg in enumerate(depth):
            curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(embed_dim, num_patches, block_cfg, num_heads=num_heads, input_resolution=input_resolution,
                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                  attn_drop=attn_drop_rate, drop_path=dpr_, norm_layer=norm_layer,
                                  upsample=PatchMerging, selfornot=selfornot)
            dpr_ptr += curr_depth
            self.MultiScaleBlock.append(blk)

        self.cls_fusion = cls_fusion(embed_dim=embed_dim, num_classes=num_classes, num_branches=self.num_branches)
        self.norm = norm_layer(embed_dim[0] * 2)
        self.head = nn.Linear(embed_dim[0] * 2, num_classes) if num_classes > 0 else nn.Identity()

        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                trunc_normal_(self.pos_embed[i], std=.02)
            trunc_normal_(self.cls_token[i], std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B, C, H, W = x.shape
        xs = []
        for i in range(self.num_branches):
            x_ = torch.nn.functional.interpolate(x, size=(self.img_size[i], self.img_size[i]), mode='bicubic') if H != self.img_size[i] else x
            tmp = self.patch_embed[i](x_)
            cls_tokens = self.cls_token[i].expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            tmp = torch.cat((cls_tokens, tmp), dim=1)
            tmp = tmp + self.pos_embed[i]
            tmp = self.pos_drop(tmp)
            xs.append(tmp)

        for MultiScaleBlock in self.MultiScaleBlock:
            xs = MultiScaleBlock(xs)

        # NOTE: was before branch token section, move to here to assure all branch token are before layer norm
        xs = self.norm(xs)
        out = xs[:, 0]

        return out

    def forward(self, x):
        x = self.forward_features(x)
        out = self.head(x)
        return out

