#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


TOKENIZER_NAME = "fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"
DATASET_NAME = "fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"


BASELINE_ENV = {
    "SEED": "42",
    "CASEOPS_ENABLED": "1",
    "QK_GAIN_INIT": "5.25",
    "EVAL_SEQ_LEN": "2560",
    "TTT_EVAL_SEQ_LEN": "2560",
    "PHASED_TTT_NUM_PHASES": "3",
    "TTT_LORA_RANK": "80",
    "TTT_LOCAL_LR_MULT": "0.75",
    "TTT_BETA2": "0.99",
    "TTT_WEIGHT_DECAY": "0.5",
    "EMBED_BITS": "7",
    "EMBED_CLIP_SIGMAS": "14.0",
    "MLP_CLIP_SIGMAS": "11.5",
    "MIN_LR": "0.1",
    "BETA2": "0.99",
    "WARMDOWN_FRAC": "0.85",
    "VAL_LOSS_EVERY": "0",
    "SMEAR_GATE_ENABLED": "1",
    "SPARSE_ATTN_GATE_ENABLED": "1",
    "SPARSE_ATTN_GATE_SCALE": "0.5",
    "GATED_ATTN_QUANT_GATE": "1",
    "LQER_ENABLED": "1",
    "LQER_RANK": "4",
    "LQER_TOP_K": "3",
    "LQER_FACTOR_BITS": "4",
    "LQER_ASYM_ENABLED": "1",
    "LQER_ASYM_GROUP": "64",
    "AWQ_LITE_ENABLED": "1",
    "AWQ_LITE_BITS": "8",
    "AWQ_LITE_GROUP_TOP_K": "1",
    "AWQ_LITE_GROUP_SIZE": "64",
    "ASYM_LOGIT_RESCALE": "1",
    "COMPRESSOR": "pergroup",
}


BASE_CANDIDATE = {
    "TTT_MASK": "no_qv",
    "TTT_Q_LORA": "0",
    "TTT_V_LORA": "0",
    "TTT_K_LORA": "1",
    "TTT_MLP_LORA": "1",
    "TTT_O_LORA": "1",
    "TTT_LM_HEAD_LORA": "1",
    "TTT_UPDATE_LOSS_MODE": "ce",
    "TTT_UPDATE_LOSS_CAP": "0",
    "TTT_LORA_A_LR_MULT": "1.0",
    "TTT_LORA_B_LR_MULT": "1.0",
    "TTT_LORA_A_WD_MULT": "1.0",
    "TTT_LORA_B_WD_MULT": "1.0",
}


CANDIDATES = {
    "qv_on": {
        "TTT_MASK": "all",
        "TTT_Q_LORA": "1",
        "TTT_V_LORA": "1",
    },
    "robust_hard5": {
        "TTT_UPDATE_LOSS_MODE": "hard_cap",
        "TTT_UPDATE_LOSS_CAP": "5.0",
    },
    "robust_hard6": {
        "TTT_UPDATE_LOSS_MODE": "hard_cap",
        "TTT_UPDATE_LOSS_CAP": "6.0",
    },
    "robust_soft5": {
        "TTT_UPDATE_LOSS_MODE": "soft_cap",
        "TTT_UPDATE_LOSS_CAP": "5.0",
        "TTT_UPDATE_LOSS_MIN_WEIGHT": "0.25",
    },
    "robust_soft6": {
        "TTT_UPDATE_LOSS_MODE": "soft_cap",
        "TTT_UPDATE_LOSS_CAP": "6.0",
        "TTT_UPDATE_LOSS_MIN_WEIGHT": "0.25",
    },
    "no_lm_head": {
        "TTT_LM_HEAD_LORA": "0",
    },
    "no_k": {
        "TTT_K_LORA": "0",
    },
    "no_o": {
        "TTT_O_LORA": "0",
    },
    "mlp_only": {
        "TTT_K_LORA": "0",
        "TTT_O_LORA": "0",
        "TTT_LM_HEAD_LORA": "0",
    },
    "ko_only": {
        "TTT_MLP_LORA": "0",
        "TTT_LM_HEAD_LORA": "0",
    },
    "mlp_lm_head": {
        "TTT_K_LORA": "0",
        "TTT_O_LORA": "0",
        "TTT_LM_HEAD_LORA": "1",
    },
    "a_half": {
        "TTT_LORA_A_LR_MULT": "0.5",
    },
    "a_quarter": {
        "TTT_LORA_A_LR_MULT": "0.25",
    },
    "a_frozen": {
        "TTT_LORA_A_LR_MULT": "0.0",
    },
    "a_wd2": {
        "TTT_LORA_A_WD_MULT": "2.0",
    },
}


def infer_caseops_paths(data_dir):
    roots = [
        data_dir,
        data_dir / "datasets",
        data_dir / "datasets" / "fineweb10B_sp8192_caseops",
        data_dir / "datasets" / "fineweb10B_sp8192_caseops" / "datasets",
    ]
    tokenizers = [r / "tokenizers" / TOKENIZER_NAME for r in roots]
    datasets = [r / "datasets" / DATASET_NAME for r in roots]
    tokenizer = next((p for p in tokenizers if p.exists()), None)
    dataset = next((p for p in datasets if p.exists()), None)
    if tokenizer is None:
        tried = "\n".join(str(p) for p in tokenizers)
        raise SystemExit(f"missing tokenizer; tried:\n{tried}")
    if dataset is None:
        tried = "\n".join(str(p) for p in datasets)
        raise SystemExit(f"missing dataset; tried:\n{tried}")
    return tokenizer, dataset


def run_one(args, name, overrides):
    env = os.environ.copy()
    lrzip_bin = Path.home() / ".local" / "lrzip" / "usr" / "bin"
    if lrzip_bin.exists():
        env["PATH"] = f"{lrzip_bin}:{env.get('PATH', '')}"
    tokenizer_path, data_path = infer_caseops_paths(args.data_dir)
    env.update(BASELINE_ENV)
    candidate_env = dict(BASE_CANDIDATE)
    candidate_env.update(overrides)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "RUN_ID": f"{args.run_prefix}_{name}",
            "ARTIFACT_DIR": str(args.artifact_dir),
            "DATA_DIR": str(args.data_dir),
            "DATA_PATH": str(data_path),
            "TOKENIZER_PATH": str(tokenizer_path),
            "TTT_EVAL_ONLY": "1",
            "TTT_ENABLED": "1",
            "TTT_LORA_RANK": str(args.rank),
            "TTT_LORA_ALPHA": str(args.alpha),
            "TTT_LORA_LR": str(args.lr),
            "TTT_LOCAL_LR_MULT": str(args.local_lr_mult),
            "TTT_WEIGHT_DECAY": str(args.weight_decay),
            "TTT_BETA2": str(args.beta2),
            "TTT_CHUNK_SIZE": str(args.chunk_size),
            "TTT_BATCH_SIZE": str(args.batch_size),
            "PHASED_TTT_NUM_PHASES": str(args.num_phases),
            "PHASED_TTT_PREFIX_DOCS": str(args.prefix_docs),
            "EVAL_SUBSET_START_DOC": str(args.start_doc),
            "EVAL_SUBSET_SCALE_PREFIX_DOCS": "1" if args.scale_prefix_docs else "0",
            "TTT_COMPILE_ENABLED": "1" if args.compile else "0",
            "TTT_SKIP_WARMUP": "0" if args.compile else "1",
            "TTT_EVAL_ONLY_DIAG_QUANTIZED": "1" if args.diag_quantized else "0",
            "TTT_EVAL_ONLY_DIAG_ONLY": "1" if args.diag_only else "0",
            "VAL_BATCH_TOKENS": str(args.val_batch_tokens),
        }
    )
    env.update(candidate_env)
    if args.subset_docs:
        env["EVAL_SUBSET_DOCS"] = str(args.subset_docs)
        env.pop("EVAL_SUBSET_TOKENS", None)
    else:
        env["EVAL_SUBSET_TOKENS"] = str(args.subset_tokens)
        env.pop("EVAL_SUBSET_DOCS", None)

    if args.flash_stub:
        cmd = [sys.executable, str(args.flash_stub), str(args.submission)]
    else:
        cmd = [sys.executable, str(args.submission)]
    shown = " ".join(f"{k}={v}" for k, v in sorted(candidate_env.items()) if v != BASE_CANDIDATE.get(k))
    print(f"\n===== official_subset {name} {shown} =====", flush=True)
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    log_path = args.artifact_dir / f"{args.run_prefix}_{name}.outer.log"
    result_line = None
    with log_path.open("w", encoding="utf-8") as f:
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            if "quantized_ttt_phased" in line:
                result_line = line.strip()
            elif args.diag_only and "diagnostic quantized eval-only" in line:
                result_line = line.strip()
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"{name} failed with exit code {rc}; see {log_path}")
    if result_line is None:
        raise SystemExit(f"{name} finished without quantized_ttt_phased line; see {log_path}")
    return result_line


def parse_bpb(line):
    m = re.search(r"val_bpb:([0-9.]+)", line)
    if not m:
        raise ValueError(f"Cannot parse val_bpb from: {line}")
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="records/track_10min_16mb/2026-04-30_LongCtx_NoQV_QK525_Prefix2750_Followup/train_gpt.py")
    ap.add_argument("--artifact-dir", default="records/track_10min_16mb/2026-04-30_LongCtx_NoQV_QK525_Prefix2750_Followup")
    ap.add_argument("--data-dir", default="/workspace/caseops_data")
    ap.add_argument("--flash-stub", default="")
    ap.add_argument("--run-prefix", default="official_qv_1m")
    ap.add_argument("--start-doc", type=int, default=0)
    ap.add_argument("--subset-tokens", type=int, default=1_000_000)
    ap.add_argument("--subset-docs", type=int, default=0)
    ap.add_argument("--prefix-docs", type=int, default=2500)
    ap.add_argument("--num-phases", type=int, default=3)
    ap.add_argument("--rank", type=int, default=80)
    ap.add_argument("--alpha", type=float, default=144.0)
    ap.add_argument("--lr", type=float, default=0.0001)
    ap.add_argument("--local-lr-mult", type=float, default=0.75)
    ap.add_argument("--weight-decay", type=float, default=0.5)
    ap.add_argument("--beta2", type=float, default=0.99)
    ap.add_argument("--chunk-size", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--diag-quantized", action="store_true")
    ap.add_argument("--diag-only", action="store_true")
    ap.add_argument("--val-batch-tokens", type=int, default=65536)
    ap.add_argument(
        "--candidates",
        default="qv_on",
        help="Comma-separated candidate names. Use list to print available candidates.",
    )
    ap.add_argument("--no-scale-prefix-docs", dest="scale_prefix_docs", action="store_false")
    ap.set_defaults(scale_prefix_docs=True)
    args = ap.parse_args()

    args.submission = Path(args.submission).resolve()
    args.artifact_dir = Path(args.artifact_dir).resolve()
    args.data_dir = Path(args.data_dir).resolve()
    args.flash_stub = Path(args.flash_stub).resolve() if args.flash_stub else None
    if args.candidates.strip() == "list":
        print("\n".join(sorted(CANDIDATES)))
        return
    required = [args.submission, args.artifact_dir / "final_model.int6.ptz", args.data_dir]
    if args.flash_stub is not None:
        required.append(args.flash_stub)
    for path in required:
        if not path.exists():
            raise SystemExit(f"missing required path: {path}")

    requested = [c.strip() for c in args.candidates.split(",") if c.strip()]
    unknown = [c for c in requested if c not in CANDIDATES]
    if unknown:
        raise SystemExit(f"unknown candidates: {', '.join(unknown)}")

    base_line = run_one(args, "base_no_qv", {})
    if args.diag_only:
        print("\n===== official_subset diagnostic summary =====")
        print(base_line)
        return
    base_bpb = parse_bpb(base_line)
    results = []
    for cand in requested:
        line = run_one(args, cand, CANDIDATES[cand])
        results.append((cand, line, parse_bpb(line)))
    print("\n===== official_subset summary =====")
    print(base_line)
    for cand, line, bpb in results:
        print(line)
        print(f"delta_bpb(base-{cand}): {base_bpb - bpb:+.8f}")


if __name__ == "__main__":
    main()
