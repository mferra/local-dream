"""Exports an Anima checkpoint to the ONNX graphs + token table Local Dream's
QNN package needs.

  python export_onnx.py --dit DIT.safetensors --te TE.safetensors --out DIR
      [--residual_scale 32] [--only unet_part1,unet_part2,clip]

Writes DIR/<graph>/model.onnx (+ external weight data) for unet_part1,
unet_part2 and clip, and DIR/token_emb.bin: the Qwen3 input embedding table
as raw fp16 [151936, 1024], which the app looks prompt tokens up in.
"""
from __future__ import annotations

import argparse
import gc
import os

import torch

import anima_model as am
from weights import load_dit, load_te

torch.set_grad_enabled(False)


def export(model, args, path, inputs, outputs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.onnx.export(
        model,
        args,
        path,
        input_names=inputs,
        output_names=outputs,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print("wrote", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit", required=True)
    ap.add_argument("--te", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--residual_scale", type=float, default=32.0)
    ap.add_argument("--only", default="unet_part1,unet_part2,clip")
    a = ap.parse_args()
    only = set(a.only.split(","))
    os.makedirs(a.out, exist_ok=True)

    dit = load_dit(a.dit)
    sample = torch.randn(1, am.LATENT_C, 1, am.LATENT_H, am.LATENT_W)
    ctx = torch.randn(1, am.SEQ_TEXT, 1024)
    t = torch.tensor([0.5])

    if "unet_part1" in only:
        m = am.DitPart1(dit, a.residual_scale).eval()
        export(m, (sample, ctx, t), f"{a.out}/unet_part1/model.onnx",
               ["sample", "encoder_hidden_states", "timestamp"], ["hidden", "emb"])
        del m
        gc.collect()

    if "unet_part2" in only:
        m = am.DitPart2(dit, a.residual_scale).eval()
        hidden = torch.randn(1, am.TOKENS_H * am.TOKENS_W, am.DIT_DIM)
        emb = torch.randn(1, 1, am.DIT_DIM)
        export(m, (hidden, emb, ctx, t), f"{a.out}/unet_part2/model.onnx",
               ["hidden", "emb", "context", "timestamp"], ["output"])
        del m
        gc.collect()

    te = load_te(a.te)
    if "clip" in only:
        m = am.AnimaClip(te, dit).eval()
        inp = torch.randn(1, am.SEQ_TEXT, 1024)
        ids = torch.zeros(1, am.SEQ_TEXT, dtype=torch.int32)
        mask = torch.ones(1, am.SEQ_TEXT)
        export(m, (inp, ids, mask, mask), f"{a.out}/clip/model.onnx",
               ["input_embedding", "t5_ids", "t5_mask", "qwen_mask"], ["context"])
        del m
        gc.collect()

    te["model.embed_tokens.weight"].half().numpy().tofile(f"{a.out}/token_emb.bin")
    print("wrote", f"{a.out}/token_emb.bin")


if __name__ == "__main__":
    main()
