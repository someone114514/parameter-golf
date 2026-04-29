#!/usr/bin/env python3
from huggingface_hub import snapshot_download


def main():
    snapshot_download(
        repo_id="romeerp/parameter-golf-caseops-v1",
        repo_type="dataset",
        local_dir="data/datasets/fineweb10B_sp8192_caseops",
        allow_patterns=[
            "datasets/manifest.json",
            "datasets/tokenizers/*",
            "datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin",
            "datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin",
            "datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_*.bin",
        ],
    )


if __name__ == "__main__":
    main()
