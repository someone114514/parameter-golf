import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


MODES = {
    "no_ttt": {
        "PHASED_TTT_PREFIX_DOCS": "0",
        "GLOBAL_TTT_LR": "0.001",
        "TTT_TAIL_POLICY": "no_update",
    },
    "base2060": {
        "PHASED_TTT_PREFIX_DOCS": "3000",
        "GLOBAL_TTT_LR": "0.001",
        "TTT_TAIL_POLICY": "none",
    },
    "p3500_lr0008": {
        "PHASED_TTT_PREFIX_DOCS": "3500",
        "GLOBAL_TTT_LR": "0.0008",
        "TTT_TAIL_POLICY": "none",
    },
    "p3500_batch300_150": {
        "PHASED_TTT_PREFIX_DOCS": "3500",
        "GLOBAL_TTT_LR": "0.0008",
        "TTT_TAIL_POLICY": "batch300_150",
    },
    "p3500_batch300_off": {
        "PHASED_TTT_PREFIX_DOCS": "3500",
        "GLOBAL_TTT_LR": "0.0008",
        "TTT_TAIL_POLICY": "batch300_off",
    },
    "p3500_doc512_256": {
        "PHASED_TTT_PREFIX_DOCS": "3500",
        "GLOBAL_TTT_LR": "0.0008",
        "TTT_TAIL_POLICY": "doc512_256",
    },
    "p3500_doc384_192": {
        "PHASED_TTT_PREFIX_DOCS": "3500",
        "GLOBAL_TTT_LR": "0.0008",
        "TTT_TAIL_POLICY": "doc384_192",
    },
}


def prepare_artifact_dir(model_path: Path, artifact_dir: Path) -> Path:
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target = artifact_dir / "final_model.int6.ptz"
    if target.exists():
        return artifact_dir
    if target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(model_path.resolve())
    except OSError:
        shutil.copy2(model_path, target)
    return artifact_dir


def root_relative(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def run_one(args, mode: str) -> int:
    if mode not in MODES:
        raise SystemExit(f"unknown mode {mode!r}; choices: {', '.join(MODES)}")
    artifact_dir = prepare_artifact_dir(root_relative(args.model), root_relative(args.artifact_dir))
    caseops_root = Path(args.caseops_root)
    data_dir = caseops_root / "datasets"
    data_path = data_dir / "datasets" / "fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
    tokenizer_path = data_dir / "tokenizers" / "fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"
    env = os.environ.copy()
    env.update(
        {
            "SEED": str(args.seed),
            "RUN_ID": f"{args.run_id_prefix}_{mode}_seed{args.seed}",
            "ARTIFACT_DIR": str(artifact_dir),
            "TTT_EVAL_ONLY": "1",
            "TTT_SEGMENT_LOG": "1",
            "TTT_COMPILE_ENABLED": str(args.compile),
            "TTT_SKIP_WARMUP": str(args.skip_warmup),
            "CASEOPS_ROOT": str(caseops_root),
            "CASEOPS_ENABLED": "1",
            "DATA_DIR": str(data_dir),
            "DATA_PATH": str(data_path),
            "TOKENIZER_PATH": str(tokenizer_path),
            "TTT_LOCAL_LR_MULT": "0.80",
            "MATRIX_LR": "0.028",
            "LQER_RANK": "2",
            "LQER_ASYM_GROUP": "32",
            "LQER_TOP_K": "4",
        }
    )
    env.update(MODES[mode])
    if args.batch_range:
        env["TTT_BATCH_RANGE"] = args.batch_range
    log_path = HERE / f"{env['RUN_ID']}.outer.log"
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.gpus}",
        str(HERE / "train_gpt.py"),
    ]
    print(f"===== running {mode} -> {log_path} =====", flush=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        return proc.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/prefix3500_global0008_seed42_final_model.int6.ptz")
    parser.add_argument("--artifact-dir", default="artifacts/ttt_segment_probe_model")
    parser.add_argument("--caseops-root", default="/workspace/caseops_data")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id-prefix", default="ttt_segment_probe")
    parser.add_argument("--batch-range", default="")
    parser.add_argument("--compile", default="0")
    parser.add_argument("--skip-warmup", default="1")
    parser.add_argument(
        "--modes",
        default="no_ttt,base2060,p3500_lr0008,p3500_batch300_150,p3500_batch300_off,p3500_doc512_256",
        help=f"comma-separated modes from: {', '.join(MODES)}",
    )
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in modes:
        rc = run_one(args, mode)
        if rc != 0:
            raise SystemExit(f"{mode} failed with exit code {rc}")


if __name__ == "__main__":
    main()
