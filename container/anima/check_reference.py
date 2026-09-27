"""Checks anima_model against ComfyUI's Anima implementation and measures
what the FP16 export has to survive.

  python check_reference.py --comfyui DIR --dit DIT.safetensors [--te TE.safetensors]

DiT: same sample/timestep/context through ComfyUI (fp32) and through
DitPart1 -> DitPart2 (fp32, residual_scale 1): reports the relative error,
the residual stream's per-block abs max (the value residual_scale must bring
under FP16's range) and, with --fp16, the error of the FP16 run at the chosen
residual scale.

Text encoder (with --te): ComfyUI's Qwen3 + LLM adapter vs AnimaClip on a real
tokenized prompt, fp32, plus the Qwen hidden-state abs max.
"""
from __future__ import annotations

import argparse
import gc
import sys

import torch

import anima_model as am
from weights import load_dit, load_te

torch.manual_seed(0)
torch.set_grad_enabled(False)


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def import_comfy(root):
    sys.path.insert(0, root)
    import comfy.options

    # comfy.cli_args only parses argv when enabled; force the CPU device.
    sys.argv = [sys.argv[0], "--cpu"]
    comfy.options.enable_args_parsing()
    import comfy.ops  # noqa: F401
    import comfy.ldm.anima.model as anima  # noqa: F401
    return sys.modules["comfy.ops"], anima


def comfy_dit(comfy_ops, anima, sd):
    cfg = dict(
        max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16,
        patch_spatial=2, patch_temporal=1, model_channels=2048, concat_padding_mask=True,
        crossattn_emb_channels=1024, pos_emb_cls="rope3d", pos_emb_learnable=True,
        pos_emb_interpolation="crop", min_fps=1, max_fps=30, use_adaln_lora=True,
        adaln_lora_dim=256, num_blocks=28, num_heads=16, extra_per_block_abs_pos_emb=False,
        rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0,
        rope_t_extrapolation_ratio=1.0, extra_h_extrapolation_ratio=1.0,
        extra_w_extrapolation_ratio=1.0, extra_t_extrapolation_ratio=1.0,
        rope_enable_fps_modulation=False,
    )
    m = anima.Anima(**cfg, device="cpu", dtype=torch.float32,
                    operations=comfy_ops.disable_weight_init)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "pos_embedder" not in k]
    if missing or unexpected:
        raise SystemExit(f"ComfyUI load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return m.eval()


def run_dit(p1, p2, sample, ctx, t, trace=None):
    emb, lora = p1.time(t)
    x = p1.x_embedder(p1.patchify(sample))
    rope = (p1.cos, p1.sin)
    for blk in list(p1.blocks) + list(p2.blocks):
        x = blk(x, emb, lora, ctx, rope)
        if trace is not None:
            trace.append(x.abs().max().item())
    hidden = x
    return hidden, p2(hidden, emb, ctx, t) if trace is None else None


def check_dit(args, comfy):
    sd = load_dit(args.dit)
    sample = torch.randn(1, am.LATENT_C, 1, am.LATENT_H, am.LATENT_W)
    ctx = torch.randn(1, am.SEQ_TEXT, 1024)
    t = torch.tensor([args.timestep])

    ref_model = comfy_dit(*comfy, sd)
    ref = ref_model(sample, t, ctx)
    del ref_model
    gc.collect()

    p1, p2 = am.DitPart1(sd, 1.0).eval(), am.DitPart2(sd, 1.0).eval()
    hidden, emb = p1(sample, ctx, t)
    out = p2(hidden, emb, ctx, t)
    print(f"[dit] fp32 split vs ComfyUI: rel err {rel(out, ref):.2e}")

    trace = []
    run_dit(p1, p2, sample, ctx, t, trace)
    print("[dit] residual abs max per block:", " ".join(f"{v:.0f}" for v in trace))
    print(f"[dit] residual abs max overall: {max(trace):.1f} (fp16 max 65504)")
    del p1, p2
    gc.collect()

    if args.fp16:
        s = args.residual_scale
        q1 = am.DitPart1(sd, s).eval().half()
        q2 = am.DitPart2(sd, s).eval().half()
        h16, e16 = q1(sample.half(), ctx.half(), t.half())
        out16 = q2(h16, e16, ctx.half(), t.half()).float()
        bad = (~torch.isfinite(out16)).sum().item()
        print(f"[dit] fp16 (residual_scale {s}) vs ComfyUI: rel err {rel(out16.nan_to_num(), ref):.2e}, non-finite {bad}")


def check_clip(args, comfy):
    import comfy.text_encoders.anima as te_mod
    import comfy.sd1_clip  # noqa: F401
    dit_sd = load_dit(args.dit)
    te_sd = load_te(args.te)
    prompt = "masterpiece, best quality, 1girl, standing in a field of flowers, sunset"
    tok = te_mod.AnimaTokenizer()
    pairs = tok.tokenize_with_weights(prompt)
    qwen_ids = [p[0] for p in pairs["qwen3_06b"][0]]
    t5_ids = [p[0] for p in pairs["t5xxl"][0]]

    # ComfyUI reference: Qwen3 last hidden state -> llm_adapter (unpadded)
    comfy_ops, anima = comfy
    import comfy.text_encoders.llama as llama
    q = llama.Qwen3_06B({}, dtype=torch.float32, device="cpu", operations=comfy_ops.disable_weight_init)
    q.load_state_dict(te_sd, strict=False)
    ids = torch.tensor([qwen_ids])
    hid = q.model(ids)[0]
    adapter = anima.LLMAdapter(operations=comfy_ops.disable_weight_init)
    adapter.load_state_dict({k[len("llm_adapter."):]: v for k, v in dit_sd.items() if k.startswith("llm_adapter.")})
    ref = adapter(hid, torch.tensor([t5_ids]))
    ref = torch.nn.functional.pad(ref, (0, 0, 0, am.SEQ_TEXT - ref.shape[1]))

    # Ours, fed exactly as the app feeds clip.bin
    clip = am.AnimaClip(te_sd, dit_sd).eval()
    emb = te_sd["model.embed_tokens.weight"]
    n, m = len(qwen_ids), len(t5_ids)
    pad_ids = qwen_ids + [151643] * (am.SEQ_TEXT - n)
    inp = emb[torch.tensor(pad_ids)][None]
    qmask = torch.tensor([[1.0] * n + [0.0] * (am.SEQ_TEXT - n)])
    t5 = torch.tensor([t5_ids + [0] * (am.SEQ_TEXT - m)], dtype=torch.int32)
    t5mask = torch.tensor([[1.0] * m + [0.0] * (am.SEQ_TEXT - m)])
    out = clip(inp, t5, t5mask, qmask)
    print(f"[clip] qwen tokens {n}, t5 tokens {m}")
    print(f"[clip] Qwen hidden abs max (before final norm): {clip_hidden_max(clip, inp):.1f}")
    print(f"[clip] fp32 vs ComfyUI: rel err {rel(out, ref):.2e}")
    if args.fp16:
        out16 = clip.half()(inp.half(), t5, t5mask.half(), qmask.half()).float()
        print(f"[clip] fp16 vs ComfyUI: rel err {rel(out16.nan_to_num(), ref):.2e}, "
              f"non-finite {(~torch.isfinite(out16)).sum().item()}")


def clip_hidden_max(clip, inp):
    x = inp
    mx = 0.0
    for layer in clip.layers:
        x = layer(x, clip.q_cos, clip.q_sin, clip.causal)
        mx = max(mx, x.abs().max().item())
    return mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfyui", required=True)
    ap.add_argument("--dit", required=True)
    ap.add_argument("--te")
    ap.add_argument("--timestep", type=float, default=0.7)
    ap.add_argument("--residual_scale", type=float, default=32.0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--skip_dit", action="store_true")
    args = ap.parse_args()
    comfy = import_comfy(args.comfyui)
    if not args.skip_dit:
        check_dit(args, comfy)
    if args.te:
        check_clip(args, comfy)


if __name__ == "__main__":
    main()
