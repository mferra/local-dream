"""Export-oriented Anima model for Local Dream's QNN packages.

Anima = Cosmos-Predict2 2B DiT + LLM adapter, with a Qwen3-0.6B text encoder.
This is a from-scratch re-implementation written for ONNX -> QNN (FP16 on the
Hexagon HTP), not for training or general inference:

  * only plain ops (MatMul, Softmax, LayerNormalization, elementwise), static
    shapes, fixed 1024x1024 canvas (128x128 latent, 64x64 tokens);
  * graph I/O names/shapes match the published packages and the app's
    QnnModel.hpp (executeAnimaClip / executeAnimaUnetPart1 / Part2);
  * FP16 safety: the DiT residual stream grows past what FP16 holds (ComfyUI
    keeps it in FP32 for the same reason), so it is carried divided by
    `residual_scale`. The factor is folded into the weights that write into
    the residual (x_embedder, every output_proj and mlp.layer2), so it costs
    no ops, and every norm that reads the residual is scale invariant.
    RMSNorms over Qwen's hidden states pre-divide by a constant so the
    mean-of-squares cannot overflow FP16 either.

Checked numerically against ComfyUI's implementation by check_reference.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# Canvas the app runs Anima at (Config.hpp): 1024x1024 px -> 128x128 latent.
LATENT_C, LATENT_H, LATENT_W = 16, 128, 128
PATCH = 2
TOKENS_H, TOKENS_W = LATENT_H // PATCH, LATENT_W // PATCH  # 64 x 64
SEQ_TEXT = 512  # anima_text_seq_len == anima_qwen_seq_len

DIT_DIM, DIT_HEADS, DIT_BLOCKS = 2048, 16, 28
DIT_HEAD_DIM = DIT_DIM // DIT_HEADS  # 128
SPLIT_BLOCK = 14  # part1 = blocks[:14], part2 = blocks[14:] (same as published)
# The residual stream peaks near 4e4 (last block) in FP32. Carried / 32 it
# stays ~1.2e3; the norms further divide by 8 before squaring.
DEFAULT_RESIDUAL_SCALE = 32.0
DIT_NORM_PRESCALE = 8.0


def rms_norm(x, weight, eps, prescale=1.0):
    """RMSNorm; `prescale` divides x first so x*x stays inside FP16."""
    if prescale != 1.0:
        x = x * (1.0 / prescale)
        eps = eps / (prescale * prescale)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def layer_norm(x, eps=1e-6, prescale=DIT_NORM_PRESCALE):
    """LayerNorm is scale invariant, so dividing first only keeps the
    variance's squares inside FP16 however the HTP computes them."""
    return F.layer_norm(x * (1.0 / prescale), (x.shape[-1],), eps=eps / (prescale * prescale))


def attention(q, k, v, bias=None):
    """q,k,v: [B, heads, L, hd]. bias: additive [B, 1, 1|Lq, Lk] or None."""
    scores = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(q.shape[-1]))
    if bias is not None:
        scores = scores + bias
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def key_mask_bias(mask):
    """[B, L] 1/0 key mask -> additive [B, 1, 1, L] bias. -1e4 is FP16-safe."""
    return ((mask - 1.0) * 1e4)[:, None, None, :]


def rotate_half_rope(x, cos, sin):
    """x: [B, heads, L, hd]; cos/sin: [L, hd] (halves duplicated)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


# --------------------------------------------------------------------------- DiT

def dit_rope_tables():
    """3D RoPE for the fixed 1x64x64 token grid, as ComfyUI's
    VideoRopePosition3DEmb (h/w extrapolation 4, t 1) + split-half rotation.
    Returns cos, sin: [4096, 128]."""
    dim_h = DIT_HEAD_DIM // 6 * 2  # 42
    dim_w = dim_h
    dim_t = DIT_HEAD_DIM - 2 * dim_h  # 44
    rng_s = torch.arange(0, dim_h, 2, dtype=torch.float64)[: dim_h // 2] / dim_h
    rng_t = torch.arange(0, dim_t, 2, dtype=torch.float64)[: dim_t // 2] / dim_t
    h_theta = 10000.0 * 4.0 ** (dim_h / (dim_h - 2))
    w_theta = 10000.0 * 4.0 ** (dim_w / (dim_w - 2))
    t_theta = 10000.0
    fh = torch.outer(torch.arange(TOKENS_H, dtype=torch.float64), 1.0 / h_theta**rng_s)
    fw = torch.outer(torch.arange(TOKENS_W, dtype=torch.float64), 1.0 / w_theta**rng_s)
    ft = torch.outer(torch.zeros(1, dtype=torch.float64), 1.0 / t_theta**rng_t)
    # per-token angles, ordered (t, h, w) along the 64 rotary pairs
    ang = torch.cat(
        [
            ft[:, None, None, :].expand(1, TOKENS_H, TOKENS_W, -1),
            fh[None, :, None, :].expand(1, TOKENS_H, TOKENS_W, -1),
            fw[None, None, :, :].expand(1, TOKENS_H, TOKENS_W, -1),
        ],
        dim=-1,
    ).reshape(TOKENS_H * TOKENS_W, DIT_HEAD_DIM // 2)
    ang = torch.cat((ang, ang), dim=-1)
    return ang.cos().float(), ang.sin().float()


class DitWeights:
    """Plain fp32 tensors of the DiT, keys without the diffusion_model prefix."""

    def __init__(self, sd: dict[str, torch.Tensor]):
        self.sd = sd

    def __getitem__(self, k):
        return self.sd[k]


class Linear(nn.Module):
    def __init__(self, w, b=None, scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(w * scale, requires_grad=False)
        self.bias = None if b is None else nn.Parameter(b * scale, requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


def _p(t):
    return nn.Parameter(t.clone(), requires_grad=False)


class TimeEmbedding(nn.Module):
    """timestamp [1] -> (emb [1,1,2048] (normed sinusoid), adaln_lora [1,1,6144])."""

    def __init__(self, sd):
        super().__init__()
        half = DIT_DIM // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs[None, :], persistent=False)
        self.linear_1 = Linear(sd["t_embedder.1.linear_1.weight"])
        self.linear_2 = Linear(sd["t_embedder.1.linear_2.weight"])
        self.norm_w = _p(sd["t_embedding_norm.weight"])

    def sinusoid(self, timestamp):
        arg = timestamp.reshape(1, 1) * self.freqs
        return torch.cat((torch.cos(arg), torch.sin(arg)), dim=-1).reshape(1, 1, DIT_DIM)

    def forward(self, timestamp):
        s = self.sinusoid(timestamp)
        adaln_lora = self.linear_2(F.silu(self.linear_1(s)))
        emb = rms_norm(s, self.norm_w, 1e-6)
        return emb, adaln_lora


class Modulation(nn.Module):
    def __init__(self, sd, prefix, chunks):
        super().__init__()
        self.l1 = Linear(sd[prefix + ".1.weight"])
        self.l2 = Linear(sd[prefix + ".2.weight"])
        self.chunks = chunks

    def forward(self, emb, adaln_lora):
        m = self.l2(self.l1(F.silu(emb))) + adaln_lora[..., : self.chunks * DIT_DIM]
        return m.chunk(self.chunks, dim=-1)


class DitAttention(nn.Module):
    def __init__(self, sd, prefix, out_scale):
        super().__init__()
        self.q = Linear(sd[prefix + ".q_proj.weight"])
        self.k = Linear(sd[prefix + ".k_proj.weight"])
        self.v = Linear(sd[prefix + ".v_proj.weight"])
        self.o = Linear(sd[prefix + ".output_proj.weight"], scale=out_scale)
        self.q_norm = _p(sd[prefix + ".q_norm.weight"])
        self.k_norm = _p(sd[prefix + ".k_norm.weight"])

    def forward(self, x, context=None, rope=None):
        ctx = x if context is None else context
        b, lq, lk = x.shape[0], x.shape[1], ctx.shape[1]
        q = self.q(x).reshape(b, lq, DIT_HEADS, DIT_HEAD_DIM)
        k = self.k(ctx).reshape(b, lk, DIT_HEADS, DIT_HEAD_DIM)
        v = self.v(ctx).reshape(b, lk, DIT_HEADS, DIT_HEAD_DIM).transpose(1, 2)
        q = rms_norm(q, self.q_norm, 1e-6).transpose(1, 2)
        k = rms_norm(k, self.k_norm, 1e-6).transpose(1, 2)
        if rope is not None:
            cos, sin = rope
            q = rotate_half_rope(q, cos, sin)
            k = rotate_half_rope(k, cos, sin)
        o = attention(q, k, v).transpose(1, 2).reshape(b, lq, DIT_DIM)
        return self.o(o)


class DitBlock(nn.Module):
    def __init__(self, sd, i, residual_scale):
        super().__init__()
        p = f"blocks.{i}"
        inv = 1.0 / residual_scale
        self.mod_sa = Modulation(sd, p + ".adaln_modulation_self_attn", 3)
        self.mod_ca = Modulation(sd, p + ".adaln_modulation_cross_attn", 3)
        self.mod_mlp = Modulation(sd, p + ".adaln_modulation_mlp", 3)
        self.self_attn = DitAttention(sd, p + ".self_attn", inv)
        self.cross_attn = DitAttention(sd, p + ".cross_attn", inv)
        self.mlp1 = Linear(sd[p + ".mlp.layer1.weight"])
        self.mlp2 = Linear(sd[p + ".mlp.layer2.weight"], scale=inv)

    def forward(self, x, emb, adaln_lora, context, rope):
        sh, sc, g = self.mod_sa(emb, adaln_lora)
        x = x + g * self.self_attn(layer_norm(x) * (1 + sc) + sh, rope=rope)
        sh, sc, g = self.mod_ca(emb, adaln_lora)
        x = x + g * self.cross_attn(layer_norm(x) * (1 + sc) + sh, context=context)
        sh, sc, g = self.mod_mlp(emb, adaln_lora)
        x = x + g * self.mlp2(F.gelu(self.mlp1(layer_norm(x) * (1 + sc) + sh)))
        return x


class DitPart1(nn.Module):
    """(sample [1,16,1,128,128], encoder_hidden_states [1,512,1024], timestamp [1])
    -> (hidden [1,4096,2048] (residual / residual_scale), emb [1,1,2048])."""

    def __init__(self, sd, residual_scale):
        super().__init__()
        self.time = TimeEmbedding(sd)
        # x_embedder: 68 = (16 latent + 1 padding-mask) * 2 * 2 features
        self.x_embedder = Linear(sd["x_embedder.proj.1.weight"], scale=1.0 / residual_scale)
        self.blocks = nn.ModuleList(DitBlock(sd, i, residual_scale) for i in range(SPLIT_BLOCK))
        cos, sin = dit_rope_tables()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def patchify(self, sample):
        # ComfyUI concatenates an all-zero padding mask channel, then
        # "b c (t r) (h m) (w n) -> b t h w (c r m n)" with r=1, m=n=2.
        x = sample.reshape(1, LATENT_C, 1, LATENT_H, LATENT_W)
        x = torch.cat((x, torch.zeros_like(x[:, :1])), dim=1)
        x = x.reshape(1, LATENT_C + 1, TOKENS_H, PATCH, TOKENS_W, PATCH)
        x = x.permute(0, 2, 4, 1, 3, 5)  # b h w c m n
        return x.reshape(1, TOKENS_H * TOKENS_W, (LATENT_C + 1) * PATCH * PATCH)

    def forward(self, sample, encoder_hidden_states, timestamp):
        emb, adaln_lora = self.time(timestamp)
        x = self.x_embedder(self.patchify(sample))
        rope = (self.cos, self.sin)
        for blk in self.blocks:
            x = blk(x, emb, adaln_lora, encoder_hidden_states, rope)
        return x, emb


class DitPart2(nn.Module):
    """(hidden, emb, context, timestamp) -> output [1,16,1,128,128].
    adaln_lora is recomputed from timestamp (as in the published graphs)."""

    def __init__(self, sd, residual_scale):
        super().__init__()
        self.time = TimeEmbedding(sd)
        self.blocks = nn.ModuleList(
            DitBlock(sd, i, residual_scale) for i in range(SPLIT_BLOCK, DIT_BLOCKS)
        )
        self.final_mod = Modulation(sd, "final_layer.adaln_modulation", 2)
        self.final_linear = Linear(sd["final_layer.linear.weight"])
        cos, sin = dit_rope_tables()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, hidden, emb, context, timestamp):
        _, adaln_lora = self.time(timestamp)
        x = hidden
        rope = (self.cos, self.sin)
        for blk in self.blocks:
            x = blk(x, emb, adaln_lora, context, rope)
        shift, scale = self.final_mod(emb, adaln_lora)
        x = self.final_linear(layer_norm(x) * (1 + scale) + shift)  # [1, 4096, 64]
        # "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)"
        x = x.reshape(1, TOKENS_H, TOKENS_W, PATCH, PATCH, LATENT_C)
        x = x.permute(0, 5, 1, 3, 2, 4)  # b C h p1 w p2
        return x.reshape(1, LATENT_C, 1, LATENT_H, LATENT_W)


# ------------------------------------------------------- text encoder (clip.bin)

QWEN_LAYERS, QWEN_DIM, QWEN_HEADS, QWEN_KV_HEADS, QWEN_HEAD_DIM = 28, 1024, 16, 8, 128
QWEN_ROPE_THETA = 1_000_000.0
QWEN_NORM_PRESCALE = 64.0  # keeps mean(x^2) of Qwen's massive activations in FP16

ADAPTER_DIM, ADAPTER_HEADS, ADAPTER_BLOCKS = 1024, 16, 6
ADAPTER_HEAD_DIM = ADAPTER_DIM // ADAPTER_HEADS  # 64


def rope_1d(seq, head_dim, theta):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    ang = torch.outer(torch.arange(seq, dtype=torch.float64), inv)
    ang = torch.cat((ang, ang), dim=-1)
    return ang.cos().float(), ang.sin().float()


class QwenLayer(nn.Module):
    def __init__(self, sd, i):
        super().__init__()
        p = f"model.layers.{i}"
        self.in_norm = _p(sd[p + ".input_layernorm.weight"])
        self.post_norm = _p(sd[p + ".post_attention_layernorm.weight"])
        self.q = Linear(sd[p + ".self_attn.q_proj.weight"])
        self.k = Linear(sd[p + ".self_attn.k_proj.weight"])
        self.v = Linear(sd[p + ".self_attn.v_proj.weight"])
        self.o = Linear(sd[p + ".self_attn.o_proj.weight"])
        self.q_norm = _p(sd[p + ".self_attn.q_norm.weight"])
        self.k_norm = _p(sd[p + ".self_attn.k_norm.weight"])
        self.gate = Linear(sd[p + ".mlp.gate_proj.weight"])
        self.up = Linear(sd[p + ".mlp.up_proj.weight"])
        self.down = Linear(sd[p + ".mlp.down_proj.weight"])

    def forward(self, x, cos, sin, bias):
        L = x.shape[1]
        h = rms_norm(x, self.in_norm, 1e-6, QWEN_NORM_PRESCALE)
        q = rms_norm(self.q(h).reshape(1, L, QWEN_HEADS, QWEN_HEAD_DIM), self.q_norm, 1e-6)
        k = rms_norm(self.k(h).reshape(1, L, QWEN_KV_HEADS, QWEN_HEAD_DIM), self.k_norm, 1e-6)
        v = self.v(h).reshape(1, L, QWEN_KV_HEADS, QWEN_HEAD_DIM)
        q = rotate_half_rope(q.transpose(1, 2), cos, sin)
        k = rotate_half_rope(k.transpose(1, 2), cos, sin)
        rep = QWEN_HEADS // QWEN_KV_HEADS
        k = k.repeat_interleave(rep, dim=1)
        v = v.transpose(1, 2).repeat_interleave(rep, dim=1)
        o = attention(q, k, v, bias).transpose(1, 2).reshape(1, L, QWEN_HEADS * QWEN_HEAD_DIM)
        x = x + self.o(o)
        h = rms_norm(x, self.post_norm, 1e-6, QWEN_NORM_PRESCALE)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class AdapterAttention(nn.Module):
    def __init__(self, sd, p):
        super().__init__()
        self.q = Linear(sd[p + ".q_proj.weight"])
        self.k = Linear(sd[p + ".k_proj.weight"])
        self.v = Linear(sd[p + ".v_proj.weight"])
        self.o = Linear(sd[p + ".o_proj.weight"])
        self.q_norm = _p(sd[p + ".q_norm.weight"])
        self.k_norm = _p(sd[p + ".k_norm.weight"])

    def forward(self, x, ctx, rope_q, rope_k, bias):
        lq, lk = x.shape[1], ctx.shape[1]
        q = rms_norm(self.q(x).reshape(1, lq, ADAPTER_HEADS, ADAPTER_HEAD_DIM), self.q_norm, 1e-6)
        k = rms_norm(self.k(ctx).reshape(1, lk, ADAPTER_HEADS, ADAPTER_HEAD_DIM), self.k_norm, 1e-6)
        v = self.v(ctx).reshape(1, lk, ADAPTER_HEADS, ADAPTER_HEAD_DIM).transpose(1, 2)
        q = rotate_half_rope(q.transpose(1, 2), *rope_q)
        k = rotate_half_rope(k.transpose(1, 2), *rope_k)
        return self.o(attention(q, k, v, bias).transpose(1, 2).reshape(1, lq, ADAPTER_DIM))


class AdapterBlock(nn.Module):
    def __init__(self, sd, i):
        super().__init__()
        p = f"llm_adapter.blocks.{i}"
        self.norm_sa = _p(sd[p + ".norm_self_attn.weight"])
        self.norm_ca = _p(sd[p + ".norm_cross_attn.weight"])
        self.norm_mlp = _p(sd[p + ".norm_mlp.weight"])
        self.self_attn = AdapterAttention(sd, p + ".self_attn")
        self.cross_attn = AdapterAttention(sd, p + ".cross_attn")
        self.mlp0 = Linear(sd[p + ".mlp.0.weight"], sd[p + ".mlp.0.bias"])
        self.mlp2 = Linear(sd[p + ".mlp.2.weight"], sd[p + ".mlp.2.bias"])

    def forward(self, x, ctx, rope, self_bias, cross_bias):
        x = x + self.self_attn(rms_norm(x, self.norm_sa, 1e-6), rms_norm(x, self.norm_sa, 1e-6),
                               rope, rope, self_bias)
        x = x + self.cross_attn(rms_norm(x, self.norm_ca, 1e-6), ctx, rope, rope, cross_bias)
        return x + self.mlp2(F.gelu(self.mlp0(rms_norm(x, self.norm_mlp, 1e-6))))


class AnimaClip(nn.Module):
    """Merged Qwen3-0.6B + LLM adapter, the app's clip.bin contract:
    (input_embedding [1,512,1024], t5_ids [1,512] int32, t5_mask [1,512],
     qwen_mask [1,512]) -> context [1,512,1024].

    input_embedding is the (prompt-weighted) Qwen token_emb lookup the app
    does on the CPU; pads sit at the end, so Qwen's causal mask alone keeps
    them out of the real tokens. qwen_mask hides them from the adapter's cross
    attention, t5_mask from its self attention, and padded T5 rows are zeroed,
    which is what ComfyUI's zero padding of the adapter output to 512 gives."""

    def __init__(self, te_sd, dit_sd):
        super().__init__()
        self.layers = nn.ModuleList(QwenLayer(te_sd, i) for i in range(QWEN_LAYERS))
        self.final_norm = _p(te_sd["model.norm.weight"])
        self.embed = _p(dit_sd["llm_adapter.embed.weight"])
        self.blocks = nn.ModuleList(AdapterBlock(dit_sd, i) for i in range(ADAPTER_BLOCKS))
        self.out_proj = Linear(dit_sd["llm_adapter.out_proj.weight"], dit_sd["llm_adapter.out_proj.bias"])
        self.out_norm = _p(dit_sd["llm_adapter.norm.weight"])
        qc, qs = rope_1d(SEQ_TEXT, QWEN_HEAD_DIM, QWEN_ROPE_THETA)
        ac, as_ = rope_1d(SEQ_TEXT, ADAPTER_HEAD_DIM, 10000.0)
        self.register_buffer("q_cos", qc, persistent=False)
        self.register_buffer("q_sin", qs, persistent=False)
        self.register_buffer("a_cos", ac, persistent=False)
        self.register_buffer("a_sin", as_, persistent=False)
        causal = torch.full((SEQ_TEXT, SEQ_TEXT), -1e4).triu(1)
        self.register_buffer("causal", causal[None, None], persistent=False)

    def qwen(self, input_embedding):
        x = input_embedding
        for layer in self.layers:
            x = layer(x, self.q_cos, self.q_sin, self.causal)
        return rms_norm(x, self.final_norm, 1e-6, QWEN_NORM_PRESCALE)

    def forward(self, input_embedding, t5_ids, t5_mask, qwen_mask):
        src = self.qwen(input_embedding)
        # int32 indices straight into Gather: no int64 Cast for the HTP.
        x = F.embedding(t5_ids, self.embed)
        rope = (self.a_cos, self.a_sin)
        self_bias, cross_bias = key_mask_bias(t5_mask), key_mask_bias(qwen_mask)
        for blk in self.blocks:
            x = blk(x, src, rope, self_bias, cross_bias)
        x = rms_norm(self.out_proj(x), self.out_norm, 1e-6)
        return x * t5_mask[..., None]
