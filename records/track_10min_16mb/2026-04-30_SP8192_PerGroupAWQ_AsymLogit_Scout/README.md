# SP8192 PerGroup AWQ + AsymLogit Scout

This is an experimental scaffold for the next strict 8xH100 seed-42 run.
It is not a record submission yet: no result is claimed in this folder.

## Why this folder exists

The current clean frontier moved from the older SmearGate BOS-fix stack toward
the per-group lrzip CaseOps line:

- PR #1950: strict/compliance reproduction of PR #1934, 3-seed mean around
  `1.06003` BPB.
- PR #1945: AWQ-lite plus asymmetric logit rescale, strict 3-seed mean around
  `1.05943` BPB after the seed-42 rerun.

The immediate goal is to generate a new local artifact (`final_model.pt` and
`final_model.int6.ptz`) for post-training/quantization experiments. The scout
run should be validated on seed 42 before spending more 8-GPU time.

## Included code

- `train_gpt.py`: PR #1945 training script, which already contains AWQ-lite
  mixed GPTQ and `ASYM_LOGIT_RESCALE`.
- `run_seed42_v21_strict.sh`: strict reproduction-style V21 run. This is the
  first script to run because it has direct PR #1945 evidence.
- `run_seed42_1950_awq_asym.sh`: composition experiment: PR #1950-style clip,
  `EMBED_WD=0.06`, and `GPTQ_RESERVE_SECONDS=5.5`, plus AWQ-lite/asym-logit.
- `download_caseops_data.py`: downloads the public CaseOps dataset from
  `romeerp/parameter-golf-caseops-v1`.

## Setup

```bash
apt-get update && apt-get install -y lrzip
pip install -r records/track_10min_16mb/2026-04-30_SP8192_PerGroupAWQ_AsymLogit_Scout/requirements.txt
pip install --no-deps flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
python3 records/track_10min_16mb/2026-04-30_SP8192_PerGroupAWQ_AsymLogit_Scout/download_caseops_data.py
python3 records/track_10min_16mb/2026-04-30_SP8192_PerGroupAWQ_AsymLogit_Scout/check_caseops_data.py
```

On the training server, after FlashAttention 3 is installed, run the stricter
loader check too:

```bash
python3 records/track_10min_16mb/2026-04-30_SP8192_PerGroupAWQ_AsymLogit_Scout/check_caseops_data.py --import-train
```

The local standalone check has been verified against the public CaseOps layout:
`train_shards=1`, `first_train_tokens=100000000`, `val_tokens=47851521`,
`byte_sidecars=1`, `tokenizer_vocab=8192`.

The default data root is:

```text
data/datasets/fineweb10B_sp8192_caseops
```

Override with `CASEOPS_ROOT=/path/to/fineweb10B_sp8192_caseops` if needed.

## Recommended first run

Run the evidence-backed strict V21 config first:

```bash
bash records/track_10min_16mb/2026-04-30_SP8192_PerGroupAWQ_AsymLogit_Scout/run_seed42_v21_strict.sh
```

Expected decision criteria:

- If seed 42 is more than a couple seconds over budget, stop and adjust only
  `GPTQ_RESERVE_SECONDS` or logging/diagnostic overhead. Near-boundary runs
  are expected for this lineage and preserve useful training steps.
- If seed 42 post-TTT is not at least competitive with PR #1945's strict seed-42
  rerun, do not run three seeds.
- If it succeeds, pull back `final_model.pt`, `final_model.int6.ptz`, and
  `run.log` for local quantization experiments.

## Second run, only if needed

The #1950-composed script is a cleaner hypothesis but has less direct evidence.
It defaults to `GPTQ_RESERVE_SECONDS=4.0` to avoid over-reserving training time;
override it from the shell if the first run overshoots too much:

```bash
bash records/track_10min_16mb/2026-04-30_SP8192_PerGroupAWQ_AsymLogit_Scout/run_seed42_1950_awq_asym.sh
# or:
GPTQ_RESERVE_SECONDS=5.5 bash records/track_10min_16mb/2026-04-30_SP8192_PerGroupAWQ_AsymLogit_Scout/run_seed42_1950_awq_asym.sh
```

Use it only after the V21 strict run is understood.

## After a successful run

Bring back:

- `runs/.../final_model.pt`
- `runs/.../final_model.int6.ptz`
- `runs/.../run.log`
- this `train_gpt.py`

Then run local heldout probes against train shards only. Do not tune on
validation. A post-training candidate should clear both local heldout slices
before another 8-GPU run.
