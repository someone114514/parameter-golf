#!/usr/bin/env python3
import runpy
import sys
import types

import torch
import torch.nn.functional as F


def _flash_attn_func(q, k, v, causal=True, **_kwargs):
    qh = q.transpose(1, 2)
    kh = k.transpose(1, 2)
    vh = v.transpose(1, 2)
    if kh.size(1) != qh.size(1):
        repeat = qh.size(1) // kh.size(1)
        kh = kh.repeat_interleave(repeat, dim=1)
        vh = vh.repeat_interleave(repeat, dim=1)
    out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=causal)
    return out.transpose(1, 2).contiguous()


def _flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=0,
    max_seqlen_k=0,
    causal=True,
    **_kwargs,
):
    del max_seqlen_q, max_seqlen_k
    outs = []
    cuq = cu_seqlens_q.detach().cpu().tolist()
    cuk = cu_seqlens_k.detach().cpu().tolist()
    for i in range(len(cuq) - 1):
        qs, qe = int(cuq[i]), int(cuq[i + 1])
        ks, ke = int(cuk[i]), int(cuk[i + 1])
        qi = q[qs:qe].unsqueeze(0)
        ki = k[ks:ke].unsqueeze(0)
        vi = v[ks:ke].unsqueeze(0)
        outs.append(_flash_attn_func(qi, ki, vi, causal=causal).squeeze(0))
    return torch.cat(outs, dim=0) if outs else q.new_empty(q.shape)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_with_flash_stub.py <script.py> [args...]")
    mod = types.ModuleType("flash_attn_interface")
    mod.flash_attn_func = _flash_attn_func
    mod.flash_attn_varlen_func = _flash_attn_varlen_func
    sys.modules["flash_attn_interface"] = mod
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
