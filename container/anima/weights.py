"""Loads Anima checkpoints into plain fp32 state dicts.

DiT: model.diffusion_model.* (or diffusion_model.* / bare keys) in bf16, fp16
or fp32, or ComfyUI "comfy_quant" int8_tensorwise+convrot layers (ComfyUI
Kitchen converter): those are int8 rows with per-row scales, stored after a
block-diagonal Hadamard rotation of the input dimension, which is undone
here. The rotation is lossless; the int8 rounding is not, so the bf16 release
of a checkpoint is always the better source.

Text encoder: Qwen3-0.6B, keys ending in model.layers.* / model.norm /
model.embed_tokens under any prefix.
"""
from __future__ import annotations

import json
import math

import torch
from safetensors import safe_open

DIT_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "")


def _hadamard(n: int) -> torch.Tensor:
    """Normalized regular Hadamard (kron powers of the 4x4 one), as
    comfy_kitchen's ConvRot builds it. Symmetric and orthogonal."""
    if n < 4 or round(math.log(n, 4)) != math.log(n, 4):
        raise ValueError(f"ConvRot group size must be a power of 4, got {n}")
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float64)
    h = h4
    while h.shape[0] < n:
        h = torch.kron(h, h4)
    return h / math.sqrt(n)


def _dequant_int8(q: torch.Tensor, scale: torch.Tensor, conf: dict) -> torch.Tensor:
    if conf.get("format") != "int8_tensorwise":
        raise ValueError(f"unsupported comfy_quant format: {conf}")
    w = q.to(torch.float64) * scale.to(torch.float64)
    if conf.get("convrot"):
        g = int(conf["convrot_groupsize"])
        out_f, in_f = w.shape
        w = (w.reshape(out_f, in_f // g, g) @ _hadamard(g).T).reshape(out_f, in_f)
    return w.float()


def load_dit(path: str) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        prefix = next(p for p in DIT_PREFIXES if any(k.startswith(p + "blocks.0.") for k in keys))
        names = [k for k in keys if k.startswith(prefix)]
        base = {k[len(prefix):] for k in names}
        sd = {}
        for k in sorted(base):
            if k.endswith((".comfy_quant", ".weight_scale")):
                continue
            t = f.get_tensor(prefix + k)
            stem = k[: -len(".weight")] if k.endswith(".weight") else None
            if stem and stem + ".comfy_quant" in base:
                conf = json.loads(bytes(f.get_tensor(prefix + stem + ".comfy_quant").tolist()))
                t = _dequant_int8(t, f.get_tensor(prefix + stem + ".weight_scale"), conf)
            elif t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                raise ValueError(f"{k}: fp8 checkpoints are not supported; use the bf16 release")
            sd[k] = t.float()
    if "llm_adapter.embed.weight" not in sd:
        raise ValueError("not an Anima DiT (no llm_adapter): " + path)
    return sd


def load_te(path: str) -> dict[str, torch.Tensor]:
    """Qwen3-0.6B text encoder -> keys model.layers.*, model.norm.weight,
    model.embed_tokens.weight."""
    sd = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            i = k.find("model.")
            if i < 0 or not k[i:].startswith(("model.layers.", "model.norm.", "model.embed_tokens.")):
                continue
            sd[k[i:]] = f.get_tensor(k).float()
    need = ("model.embed_tokens.weight", "model.norm.weight", "model.layers.27.mlp.down_proj.weight")
    missing = [k for k in need if k not in sd]
    if missing:
        raise ValueError(f"not a Qwen3-0.6B text encoder ({path}); missing {missing}")
    if sd["model.embed_tokens.weight"].shape != (151936, 1024):
        raise ValueError("unexpected Qwen3 vocabulary shape: %s" % (tuple(sd["model.embed_tokens.weight"].shape),))
    return sd
