#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def run_one(args, name, mask, q_lora, v_lora):
    env = os.environ.copy()
    lrzip_bin = Path.home() / ".local" / "lrzip" / "usr" / "bin"
    if lrzip_bin.exists():
        env["PATH"] = f"{lrzip_bin}:{env.get('PATH', '')}"
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "RUN_ID": f"{args.run_prefix}_{name}",
            "ARTIFACT_DIR": str(args.artifact_dir),
            "DATA_DIR": str(args.data_dir),
            "CASEOPS_ENABLED": "1",
            "TTT_EVAL_ONLY": "1",
            "TTT_ENABLED": "1",
            "TTT_MASK": mask,
            "TTT_Q_LORA": str(q_lora),
            "TTT_V_LORA": str(v_lora),
            "TTT_K_LORA": "1",
            "TTT_MLP_LORA": "1",
            "TTT_O_LORA": "1",
            "TTT_LORA_RANK": str(args.rank),
            "TTT_LORA_ALPHA": str(args.alpha),
            "TTT_LORA_LR": str(args.lr),
            "TTT_LOCAL_LR_MULT": str(args.local_lr_mult),
            "TTT_WEIGHT_DECAY": str(args.weight_decay),
            "TTT_BETA2": str(args.beta2),
            "TTT_CHUNK_SIZE": str(args.chunk_size),
            "TTT_BATCH_SIZE": str(args.batch_size),
            "PHASED_TTT_PREFIX_DOCS": str(args.prefix_docs),
            "PHASED_TTT_NUM_PHASES": str(args.num_phases),
            "EVAL_SUBSET_START_DOC": str(args.start_doc),
            "EVAL_SUBSET_SCALE_PREFIX_DOCS": "1" if args.scale_prefix_docs else "0",
            "TTT_COMPILE_ENABLED": "1" if args.compile else "0",
            "TTT_SKIP_WARMUP": "0" if args.compile else "1",
            "TTT_EVAL_ONLY_DIAG_QUANTIZED": "1" if args.diag_quantized else "0",
            "COMPRESSOR": "pergroup",
        }
    )
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
    print(f"\n===== official_subset {name} mask={mask} q={q_lora} v={v_lora} =====", flush=True)
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
    ap.add_argument("--data-dir", default="/tmp/caseops_preflight_again/data")
    ap.add_argument("--flash-stub", default="tools/run_with_flash_stub.py")
    ap.add_argument("--run-prefix", default="official_qv_1m")
    ap.add_argument("--start-doc", type=int, default=0)
    ap.add_argument("--subset-tokens", type=int, default=1_000_000)
    ap.add_argument("--subset-docs", type=int, default=0)
    ap.add_argument("--prefix-docs", type=int, default=2500)
    ap.add_argument("--num-phases", type=int, default=1)
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
    ap.add_argument("--no-scale-prefix-docs", dest="scale_prefix_docs", action="store_false")
    ap.set_defaults(scale_prefix_docs=True)
    args = ap.parse_args()

    args.submission = Path(args.submission).resolve()
    args.artifact_dir = Path(args.artifact_dir).resolve()
    args.data_dir = Path(args.data_dir).resolve()
    args.flash_stub = Path(args.flash_stub).resolve() if args.flash_stub else None
    required = [args.submission, args.artifact_dir / "final_model.int6.ptz", args.data_dir]
    if args.flash_stub is not None:
        required.append(args.flash_stub)
    for path in required:
        if not path.exists():
            raise SystemExit(f"missing required path: {path}")

    base_line = run_one(args, "base_no_qv", "no_qv", 0, 0)
    qv_line = run_one(args, "qv_on", "all", 1, 1)
    base_bpb = parse_bpb(base_line)
    qv_bpb = parse_bpb(qv_line)
    print("\n===== official_subset summary =====")
    print(base_line)
    print(qv_line)
    print(f"delta_bpb(base-qv): {base_bpb - qv_bpb:+.8f}")


if __name__ == "__main__":
    main()
