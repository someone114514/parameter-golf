# AggroTTT + Legal N-Gram Overlay

This record is an aggressive hybrid built on the merged `2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072` stack, with two evaluation-time additions:

1. `score-first` legal TTT adapted from `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
2. online prefix-only n-gram agreement overlay adapted from open PR `#1145`

The goal is not to revive old illegal cache scoring. The overlay keeps the stronger parts of the old n-gram family while staying in a single left-to-right pass:

- token n-gram expert
- within-word continuation expert
- word-start expert
- agreement-based boost on top of the model's normalized distribution

## Current Status

This directory is an implementation scaffold intended for cloud runs. The checked-in metadata is intentionally marked as pending until fresh runs are produced.

## Main Changes

### Base

- 11-layer 512d banked GPT
- XSA on all layers
- BigramHash `3072 x 112`
- AR self-generated Full GPTQ calibration
- selective `±1` pruning to fit the artifact budget
- split-LR defaults for later layers

### Added score-first TTT

- `TTT_ENABLED=1` by default
- score each chunk before any update touches it
- adapt only later full-parameter blocks plus `final_norm`
- reset SGD state per chunk by default
- probe and rollback chunks that get worse after adaptation
- cosine chunk LR decay

### Added legal aggressive online n-gram overlay

- `ONLINE_NGRAM_ENABLED=1` by default
- strict prefix-only state
- full-vocabulary renormalization through logit boosting
- no two-pass rescore
- no target-conditioned gating

## Local Smoke Test

This record exposes a very cheap smoke path that does not require dataset access:

```bash
conda activate parameter-golf
SMOKE_TEST=1 python records/track_10min_16mb/2026-04-05_AggroTTT_LegalNgram/train_gpt.py
```

If you prefer shell env style:

```bash
conda activate parameter-golf
SMOKE_TEST=1 python records/track_10min_16mb/2026-04-05_AggroTTT_LegalNgram/train_gpt.py
```

## Cloud Training

Base aggressive run:

```bash
conda activate parameter-golf
cd /path/to/parameter-golf

SEED=42 \
TTT_ENABLED=1 \
ONLINE_NGRAM_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-04-05_AggroTTT_LegalNgram/train_gpt.py
```

Suggested ablations:

```bash
# base only
TTT_ENABLED=0 ONLINE_NGRAM_ENABLED=0 SEED=42 torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-04-05_AggroTTT_LegalNgram/train_gpt.py

# TTT only
TTT_ENABLED=1 ONLINE_NGRAM_ENABLED=0 SEED=42 torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-04-05_AggroTTT_LegalNgram/train_gpt.py

# overlay only
TTT_ENABLED=0 ONLINE_NGRAM_ENABLED=1 SEED=42 torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-04-05_AggroTTT_LegalNgram/train_gpt.py
```

3-seed sweep:

```bash
for SEED in 42 314 999; do
  TTT_ENABLED=1 \
  ONLINE_NGRAM_ENABLED=1 \
  SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-05_AggroTTT_LegalNgram/train_gpt.py
done
```

## Recommended Knobs

- `TTT_CHUNK_TOKENS=131072`
- `TTT_EPOCHS=1`
- `TTT_FREEZE_BLOCKS=6`
- `TTT_LR=0.0005`
- `TTT_BATCH_SEQS=8`
- `TTT_MOMENTUM=0.0`
- `TTT_ROLLBACK_REL_TOL=0.01`
- `SPLIT_LR_ENABLED=1`
- `SPLIT_LR_LAYER=6`
- `SPLIT_LR_LATE_MULT=1.2`
- `TOKEN_ORDER=16`
- `TOKEN_BOOST=2.625`
- `WORD_ORDER=4`
- `AGREE_ADD_BOOST=0.5`

These are the defaults currently encoded in the record.
