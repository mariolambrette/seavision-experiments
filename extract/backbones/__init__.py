"""Backbone registry. One entry per backbone in plan section 5.3.

All seven are implemented. The registry is the single place the set of
backbones is defined.
"""
from .birder_adapter import BirderAdapter
from .dinov3_adapter import DINOv3Adapter
from .openclip_adapter import OpenClipAdapter
from .siglip2_adapter import Siglip2Adapter

REGISTRY = {
    "clip_laion2b": lambda: OpenClipAdapter(
        "clip_laion2b", "ViT-L-14", "laion2b_s32b_b82k"),
    "bioclip2": lambda: OpenClipAdapter(
        "bioclip2", "ViT-L-14", "hf-hub:imageomics/bioclip-2"),
    "bioclip25_huge": lambda: OpenClipAdapter(
        "bioclip25_huge", "ViT-H-14", "hf-hub:imageomics/bioclip-2.5-vith14"),
    "dinov3_vitl16": lambda: DINOv3Adapter(
        "dinov3_vitl16", "facebook/dinov3-vitl16-pretrain-lvd1689m",
        no_weights_cfg=dict(hidden_size=1024, num_hidden_layers=24,
                            num_attention_heads=16, intermediate_size=4096)),
    "dinov3_vith16plus": lambda: DINOv3Adapter(
        "dinov3_vith16plus", "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        no_weights_cfg=dict(hidden_size=1280, num_hidden_layers=32,
                            num_attention_heads=20, intermediate_size=5120)),
    # 224 px checkpoint; other sizes interpolate unless the config maps a
    # size to one of birder's 252/336 px checkpoints (a resolution decision).
    "bio_dino": lambda: BirderAdapter(
        "bio_dino", "vit_reg4_so150m_p14_ls",
        "vit_reg4_so150m_p14_ls_dino-v2-bio-224px", 224),
    # birder's separately trained 336 px checkpoint: native at 336, but
    # different weights. Run beside bio_dino at 336 (interpolated) so the
    # two ways of reaching 576 tokens can be compared (decided 5 Oct 2026).
    "bio_dino_ckpt336": lambda: BirderAdapter(
        "bio_dino_ckpt336", "vit_reg4_so150m_p14_ls",
        "vit_reg4_so150m_p14_ls_dino-v2-bio-336px", 336),
    # "size" is a patch budget for this model (NaFlex), not a square side
    "siglip2_naflex": lambda: Siglip2Adapter(
        "siglip2_naflex", "google/siglip2-so400m-patch16-naflex",
        no_weights_cfg=dict(hidden_size=1152, intermediate_size=4304,
                            num_hidden_layers=27, num_attention_heads=16,
                            patch_size=16, num_patches=256)),
}


def get_adapter(name: str):
    if name not in REGISTRY:
        raise KeyError(f"no adapter for {name!r}; implemented: "
                       f"{sorted(REGISTRY)}")
    return REGISTRY[name]()
