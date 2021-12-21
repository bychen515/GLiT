import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed, HybridEmbed,_cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
from mix_attention import Mix_Attention, Vit_Attention, Conv_Attention

__all__ = [
    'glit_tiny_patch16_224',
]

def count_mixit_flops(dim, v_head, e_v, e_c, k_size, mlp_r,embed_number=197):
    c_head = 3 - v_head
    dim_v = dim * (v_head) / 3
    dim_c = dim * (c_head) / 3

    flopsv1 = dim_v * dim_v * e_v * 3 * embed_number
    flopsv2 = embed_number * embed_number * dim_v * e_v * 2
    flopsv3 = dim_v * e_v * dim_v * embed_number

    flopsc1 = dim_c * dim_c * e_c * embed_number
    flopsc2 = dim_c * e_c * k_size * embed_number
    flopsc3 = dim_c * e_c * dim_c * embed_number

    flopsmlp = 2 * mlp_r * dim * dim * 197

    return (flopsv1 + flopsv2 + flopsv3 + flopsc1 + flopsc2 + flopsc3 + flopsmlp) / 1e6


class Block_mix(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, V_head=3, C_head=0, e_v=1, e_c=0, k_size=31): #GELU
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.flops = count_mixit_flops(dim, V_head, e_v, e_c, k_size, mlp_ratio)
        if V_head == 0:
            self.attn = Conv_Attention(dim=dim, num_heads=num_heads, e_c=e_c, k_size=k_size)
        elif C_head == 0:
            self.attn = Vit_Attention(dim=dim, num_heads=num_heads, e_v=e_v)
        else:
            self.attn = Mix_Attention(dim=dim, num_heads=num_heads, v_head=V_head, e_v=e_v, e_c=e_c, k_size=k_size)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RetrainTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.flops_trans = 0
        base_arch = [3, 0, 1, 3, 0, 1, 3, 1, 0, 3, 0, 3]
        arch = [1, 4, 8, 4, 8, 8, 3, 6, 8, 4, 5, 8]

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList()
        for i in range(depth):
            candidates = nn.ModuleList()
            if base_arch[i] == 0:
                count_num = 0
                for k in [17, 31, 45]:
                    for e in [1, 2, 4]:
                        if count_num == arch[i]:
                            candidates.append(
                                Block_mix(
                                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                                    V_head=base_arch[i], C_head=3 - base_arch[i], e_v=1, e_c=e, k_size=k))
                        count_num += 1
            elif base_arch[i] == 3:
                count_num = 0
                for e in [0.5, 1, 2]:
                    for mlp_r in [2, 4, 6]:
                        if count_num == arch[i]:
                            candidates.append(
                                Block_mix(
                                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_r, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                                    V_head=base_arch[i], C_head=3 - base_arch[i], e_v=e, e_c=0, k_size=31))
                        count_num += 1

            else:
                count_num = 0
                for e in [1, 2, 4]:
                    for mlp_r in [2, 4, 6]:
                        if count_num == arch[i]:
                            candidates.append(
                                Block_mix(
                                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_r, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                                    V_head=base_arch[i], C_head=3 - base_arch[i], e_v=1, e_c=e, k_size=31))
                        count_num += 1
            self.blocks.append(candidates)
            self.flops_trans += candidates[0].flops

        self.norm = norm_layer(embed_dim)


        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.flops_embed = in_chans * embed_dim * (patch_size ** 2) * ((img_size / patch_size) ** 2) / 1e6
        self.flops_head = embed_dim * num_classes / 1e6
        self.flops = self.flops_embed + self.flops_trans + self.flops_head
        print(f"Flops: {self.flops:.1f}M")

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
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk_idx, blk in enumerate(self.blocks):
            x = blk[0](x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def glit_tiny_patch16_224(pretrained=False, **kwargs):
    model = RetrainTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


if __name__ == '__main__':
    model = glit_tiny_patch16_224()
    in_pic = torch.rand([1, 3, 224, 224])
    cls = model(in_pic)
    print(f"output shape: {cls.shape}")


