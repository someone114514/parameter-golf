# Scylla1405 Modular Eval Record

This record starts from the `#1405` Scylla GPTQ base and turns it into a stricter, modular experiment package.

The default path is intentionally conservative:

- `EVAL_MODE=sliding`
- `OGD_ENABLED=0`
- `TTT_ENABLED=0`
- `ONLINE_NGRAM_ENABLED=0`

That keeps the checked-in default behavior aligned with the compliance story. Research features are opt-in.

## Base Lineage

- Base lineage: PR `#1405` (`2026-04-06_Scylla_GPTQ_BH3072`)
- Tokenizer path defaults to the bundled Scylla artifacts in this folder
- Defaults now match the intended Scylla base recipe:
  - `VOCAB_SIZE=998`
  - `BIGRAM_VOCAB_SIZE=3072`
  - `BIGRAM_DIM=112`
  - `WARMDOWN_ITERS=4000`
  - `EVAL_STRIDE=64`

## Modular Eval Modes

- `EVAL_MODE=sliding`
  - Clean base evaluation only
- `EVAL_MODE=ogd`
  - Enables the bundled OGD path only if `OGD_ENABLED=1`
- `EVAL_MODE=ttt`
  - Enables score-first TTT only if `TTT_ENABLED=1`
- `EVAL_MODE=ngram`
  - Enables online n-gram agreement only if `ONLINE_NGRAM_ENABLED=1`
- `EVAL_MODE=ttt_ngram`
  - Research mode for sequential TTT then n-gram evaluation

The code no longer runs TTT or OGD by default.

## N-Gram Notes

The online n-gram evaluator is bundled from the earlier legal online overlay work, but adapted for Scylla compatibility:

- default fast path uses token and within-word experts
- `NGRAM_WORD_ENABLED=0` by default
- word-start expert is only allowed when a compatible SentencePiece tokenizer is available
- `candidate.meta.npz` is used for boundary and leading-space metadata on the Scylla path

Recommended fast research path:

```bash
EVAL_MODE=ngram \
ONLINE_NGRAM_ENABLED=1 \
NGRAM_WORD_ENABLED=0 \
ONLINE_NGRAM_BATCH_SEQS=32 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Smoke Test

```bash
conda activate parameter-golf
SMOKE_TEST=1 python records/track_10min_16mb/2026-04-06_Scylla_GPTQ_BH3072_ModularEval/train_gpt.py
```

This only checks import, model construction, and record entrypoint wiring.

## Example Runs

Clean base:

```bash
TOKENIZER_PATH=records/track_10min_16mb/2026-04-06_Scylla_GPTQ_BH3072_ModularEval/candidate.vocab \
TOKENIZER_META_PATH=records/track_10min_16mb/2026-04-06_Scylla_GPTQ_BH3072_ModularEval/candidate.meta.npz \
DATA_PATH=./data/datasets/fineweb10B_scylla \
EVAL_MODE=sliding \
torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-04-06_Scylla_GPTQ_BH3072_ModularEval/train_gpt.py
```

Fast n-gram:

```bash
TOKENIZER_PATH=records/track_10min_16mb/2026-04-06_Scylla_GPTQ_BH3072_ModularEval/candidate.vocab \
TOKENIZER_META_PATH=records/track_10min_16mb/2026-04-06_Scylla_GPTQ_BH3072_ModularEval/candidate.meta.npz \
DATA_PATH=./data/datasets/fineweb10B_scylla \
EVAL_MODE=ngram \
ONLINE_NGRAM_ENABLED=1 \
NGRAM_WORD_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-04-06_Scylla_GPTQ_BH3072_ModularEval/train_gpt.py
```

TTT research:

```bash
TOKENIZER_PATH=records/track_10min_16mb/2026-04-06_Scylla_GPTQ_BH3072_ModularEval/candidate.vocab \
TOKENIZER_META_PATH=records/track_10min_16mb/2026-04-06_Scylla_GPTQ_BH3072_ModularEval/candidate.meta.npz \
DATA_PATH=./data/datasets/fineweb10B_scylla \
EVAL_MODE=ttt \
TTT_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-04-06_Scylla_GPTQ_BH3072_ModularEval/train_gpt.py
```
