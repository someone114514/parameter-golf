# Record Fork: Original `#1333` + Optional Legal N-Gram Eval

This folder starts from the original `#1333` code and keeps the original causal SLOT path intact. It also adds a separate `EVAL_MODE=ngram` final evaluator using a prefix-only legal n-gram overlay.

The reported metrics below are the original `#1333` baseline numbers and do **not** claim any measured n-gram result yet.

## Original `#1333` 3-Seed Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | Sliding BPB | **Causal SLOT BPB** | SLOT gain | Artifact |
|------|-------------|---------------------|-----------|----------|
| 42   | 1.0893      | **1.0762**          | -0.0131   | 15,999,461 |
| 314  | 1.0897      | **1.0766**          | -0.0131   | 15,997,932 |
| 999  | 1.0897      | **1.0770**          | -0.0127   | 15,994,941 |
| **Mean** | | **1.0766** | **-0.0130** | |

Merged SOTA (PR #1019): **1.1147 BPB**. Delta: **-0.0381 BPB**.

## Current Modes

### Baseline `#1333`
- `EVAL_MODE=slot SLOT_ENABLED=1`
- Original causal SLOT evaluator from `#1333`

### N-Gram Experiment
- `EVAL_MODE=ngram ONLINE_NGRAM_ENABLED=1 SLOT_ENABLED=0`
- Uses `online_best_agree_eval.py` and `online_ngram_state.c`
- Prefix-only token / within-word / word-start hints
- One-token logit tilt plus full-vocab renormalization
- No two-pass rescoring

## Baseline `#1333` Techniques

1. **4096-Vocab + MLP 4x + WD 0.090**
2. **Depth Recurrence (layers 4,5)**
3. **Parallel Residuals (from layer 7)**
4. **MuonEq-R**
5. **QK-Gain 5.0**
6. **Full GPTQ int6 + Brotli + LZMA Compressed Wrapper**
7. **Optional Causal SLOT** from original `#1333`
8. **Optional Legal N-Gram** from this fork

## Reproduction

Original `#1333` SLOT:
```bash
pip install brotli
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp4096 --skip-manifest
SEED=42 RECUR_LAYERS=4,5 RECUR_START_STEP=3000 PARALLEL_START_LAYER=7 \
SLOT_ENABLED=1 SLOT_LR=0.008 SLOT_STEPS=16 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

N-gram experiment:
```bash
pip install brotli
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp4096 --skip-manifest
SEED=42 RECUR_LAYERS=4,5 RECUR_START_STEP=3000 PARALLEL_START_LAYER=7 \
EVAL_MODE=ngram ONLINE_NGRAM_ENABLED=1 SLOT_ENABLED=0 \
WORD_ORDER=4 NGRAM_WORD_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Compliance Notes For The N-Gram Path

- Strict prefix-only hint extraction
- Full-vocab softmax retained after logit tilting
- Single-pass left-to-right evaluation
- No target-conditioned gating
- No two-pass rescoring

## Credits

PR #1218 @clarkkev, PR #1285 @dexhunter, PR #1204 @msisovic, PR #1289 @MatoTeziTanka, PR #1260 @dexhunter, PR #1019 @abaybektursun, PR #1287 @dentity007, PR #1217 @bigbag, PR #493 @parinzee, PR #1306 @resouer (causal SLOT), PR #1176 @bigbag (SLOT concept), PR #1145 @g-w1 (legal n-gram line)
