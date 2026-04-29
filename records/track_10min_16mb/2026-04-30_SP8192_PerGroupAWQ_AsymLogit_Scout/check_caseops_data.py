#!/usr/bin/env python3
import argparse
import glob
import importlib.util
import os
from pathlib import Path

import numpy as np
import sentencepiece as spm


def import_train(script_dir):
    spec = importlib.util.spec_from_file_location("train_gpt_caseops_check", script_dir / "train_gpt.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--caseops-root",
        default="data/datasets/fineweb10B_sp8192_caseops",
        help="Root created by download_caseops_data.py",
    )
    ap.add_argument("--sample-train-tokens", type=int, default=16)
    ap.add_argument(
        "--import-train",
        action="store_true",
        help="Also import train_gpt.py and use its loaders. Requires flash_attn_3.",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir
    while repo_root != repo_root.parent and not (repo_root / ".git").exists():
        repo_root = repo_root.parent
    caseops_root = Path(args.caseops_root)
    if not caseops_root.is_absolute():
        caseops_root = repo_root / caseops_root

    data_path = caseops_root / "datasets" / "datasets" / "fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
    tok_path = caseops_root / "datasets" / "tokenizers" / "fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"

    os.environ.update(
        {
            "CASEOPS_ENABLED": "1",
            "VOCAB_SIZE": "8192",
            "DATA_DIR": str(caseops_root / "datasets"),
            "DATA_PATH": str(data_path),
            "TOKENIZER_PATH": str(tok_path),
            "VAL_LOSS_EVERY": "0",
        }
    )

    train_glob = str(data_path / "fineweb_train_*.bin")
    val_glob = str(data_path / "fineweb_val_*.bin")
    byte_glob = str(data_path / "fineweb_val_bytes_*.bin")
    train_files = sorted(glob.glob(train_glob))
    val_files = sorted(p for p in glob.glob(val_glob) if "_bytes_" not in Path(p).name)
    byte_files = sorted(glob.glob(byte_glob))

    if not train_files:
        raise FileNotFoundError(f"no train shards: {train_glob}")
    if not val_files:
        raise FileNotFoundError(f"no val shards: {val_glob}")
    if not byte_files:
        raise FileNotFoundError(f"no val byte sidecars: {byte_glob}")
    if not tok_path.exists():
        raise FileNotFoundError(f"missing tokenizer: {tok_path}")

    def load_data_shard(file):
        header_bytes = 256 * np.dtype("<i4").itemsize
        token_bytes = np.dtype("<u2").itemsize
        file = Path(file)
        header = np.fromfile(file, dtype="<i4", count=256)
        if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
            raise ValueError(f"unexpected shard header for {file}")
        n = int(header[2])
        expected = header_bytes + n * token_bytes
        if file.stat().st_size != expected:
            raise ValueError(f"shard size mismatch for {file}: expected {expected}")
        return np.fromfile(file, dtype="<u2", count=n, offset=header_bytes)

    train = load_data_shard(train_files[0])
    val_full = np.concatenate([load_data_shard(f) for f in val_files])
    usable = (val_full.size - 1) // 2048 * 2048
    val = val_full[: usable + 1]
    val_bytes_full = np.concatenate([load_data_shard(f) for f in byte_files])
    if val_bytes_full.size < val.size:
        raise ValueError(f"byte sidecar too short: {val_bytes_full.size} < {val.size}")
    val_bytes = val_bytes_full[: val.size]
    sp = spm.SentencePieceProcessor(model_file=str(tok_path))

    if int(sp.vocab_size()) != 8192:
        raise ValueError(f"tokenizer vocab {sp.vocab_size()} != VOCAB_SIZE 8192")
    if val.size != val_bytes.size:
        raise ValueError(f"val tokens {val.size} != val bytes {val_bytes.size}")
    if int(val_bytes.sum()) <= 0:
        raise ValueError("val byte sidecar sum is zero")

    if args.import_train:
        mod = import_train(script_dir)
        h = mod.Hyperparameters()
        mod.set_logging_hparams(None)
        train_t = mod.load_data_shard(Path(train_files[0]))
        val_t = mod.load_validation_tokens(h.val_files, h.eval_seq_len)
        bytes_t = mod.load_validation_byte_sidecar(h.val_bytes_files, h.eval_seq_len, val_t.numel())
        if train_t.numel() != train.size or val_t.numel() != val.size or bytes_t.numel() != val_bytes.size:
            raise ValueError("train_gpt.py loader disagrees with standalone checker")

    print("caseops_check:ok")
    print(f"caseops_root={caseops_root}")
    print(f"train_shards={len(train_files)} first_train_tokens={train.size}")
    print(f"val_shards={len(val_files)} val_tokens={val.size}")
    print(f"byte_sidecars={len(byte_files)} val_bytes_sum={int(val_bytes.sum())}")
    print(f"tokenizer_vocab={sp.vocab_size()} sample_train={train[:args.sample_train_tokens].tolist()}")
    if args.import_train:
        print("train_gpt_loader=ok")


if __name__ == "__main__":
    main()
