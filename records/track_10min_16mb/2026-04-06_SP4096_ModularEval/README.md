# Modular Record: `#1333` Baseline + Throughput Toggle + Stronger Causal SLOT

This record folder is a modularized implementation track built on the `#1333` stack:

- `SP4096`
- `Depth Recurrence`
- `Parallel Residuals`
- `MuonEq-R`
- `QK-Gain 5.0`
- `Full GPTQ int6 + Brotli`
- `Causal SLOT`

Default behavior matches the `#1333` style submission path:

- `EVAL_MODE=slot`
- `SLOT_ENABLED=1`
- `SLOT_PARAM_MODE=baseline`
- `COMPILE_ENABLED=0`

The point of this folder is not to silently change the base. It exposes controlled switches so experiments stay comparable.

## Main Switches

### Base / Eval

- `EVAL_MODE=sliding|slot`
- `SLOT_ENABLED=0|1`
- `SLIDING_WINDOW_ENABLED=0|1`

### Throughput

- `THROUGHPUT_MODE=off|compile`
- `COMPILE_ENABLED=0|1`

### SLOT

- `SLOT_PARAM_MODE=baseline|per_layer|split_branch`
- `SLOT_TARGET_LAYERS=9,10`
- `SLOT_LR=0.008`
- `SLOT_STEPS=16`
- `SLOT_BATCH_SEQS=32`
- `SLOT_SCHEDULE=constant|cosine`
- `SLOT_EARLY_STOP=0|1`

## Compliance Notes

- No pre-quant TTT
- No two-pass rescoring
- Model weights are frozen during eval
- Causal SLOT only optimizes delta parameters on context-only positions
- First window is scored with zero delta before any SLOT optimization
- Final scoring uses the standard full-vocab softmax

## Smoke Test

```bash
SMOKE_TEST=1 python records/track_10min_16mb/2026-04-06_SP4096_ModularEval/train_gpt.py
```

## Reproduction Templates

### `#1333`-style baseline

```bash
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python data/cached_challenge_fineweb.py --variant sp4096 --skip-manifest
SEED=42 \
RECUR_LAYERS=4,5 \
RECUR_START_STEP=3000 \
PARALLEL_START_LAYER=7 \
EVAL_MODE=slot \
SLOT_ENABLED=1 \
SLOT_PARAM_MODE=baseline \
SLOT_LR=0.008 \
SLOT_STEPS=16 \
torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-04-06_SP4096_ModularEval/train_gpt.py
```

### Per-layer SLOT experiment

```bash
SEED=42 \
EVAL_MODE=slot \
SLOT_ENABLED=1 \
SLOT_PARAM_MODE=per_layer \
SLOT_TARGET_LAYERS=9,10 \
SLOT_SCHEDULE=cosine \
torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-04-06_SP4096_ModularEval/train_gpt.py
```

### Compile-only throughput experiment

```bash
SEED=42 \
THROUGHPUT_MODE=compile \
COMPILE_ENABLED=1 \
EVAL_MODE=slot \
SLOT_PARAM_MODE=baseline \
torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-04-06_SP4096_ModularEval/train_gpt.py
```
