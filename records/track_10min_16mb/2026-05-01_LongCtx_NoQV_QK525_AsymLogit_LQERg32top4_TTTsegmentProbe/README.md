# TTT Segment Probe for PR #2060

This is an eval-only probe folder for the PR #2060 stack:

`LongCtx No-QV QK5.25 + AsymLogit + LQER g32/top4 + TTT-local 0.80`

The training, quantization, tokenizer, dataset loader, document splitter, and TTT
batch queue are inherited from #2060. The probe adds only:

- global segment BPB logging for the real TTT queue,
- a no-update/no-TTT segment baseline on the exact same queue,
- fixed tail-gate policies for late short-document local TTT updates.

The purpose is to answer whether late short-document batches are intrinsically
hard or are being made worse by TTT updates.

## Queue Invariance

Both no-TTT and TTT probes use the same code path:

- `ValidationData(h, device)`
- `_find_docs(val_data.val_tokens)`
- `_select_ttt_doc_entries(docs, h)`
- `_build_ttt_global_batches(doc_entries, h, ascending=False)`

No alternate loader, shuffle, token-order traversal, or custom document ordering
is used. Full validation keeps `VAL_DOC_FRACTION=1.0`, so document selection is
deterministic.

## Probe Modes

`TTT_SEGMENT_LOG=1` logs all-rank segment lines:

```text
ttseg: policy:<policy> bucket:<bucket> batches:<n> tokens:<n> bytes:<n> val_loss:<x> val_bpb:<x>
```

Buckets are based on the original #2060 TTT batch index:

```text
b700p, b600_699, b500_599, b400_499, b300_399,
b200_299, b100_199, b050_099, b000_049
```

Tail policies:

- `none`: exact existing TTT behavior.
- `no_update`: score with zero LoRA and do no local update. Use with `PHASED_TTT_PREFIX_DOCS=0` for the no-TTT segment baseline.
- `batch300_150`: after global TTT is complete, local LR is `0.25x` for `batch_num <= 300` and `0x` for `batch_num <= 150`.
- `batch300_off`: after global TTT is complete, local update is off for `batch_num <= 300`.
- `batch200_off`: after global TTT is complete, local update is off for `batch_num <= 200`.
- `doc512_256`: after global TTT is complete, local LR is `0.25x` for batches with max doc length `<=512` and `0x` for `<=256`.
- `doc384_192`: same idea with `384/192`.

Policies use only fixed batch/doc-length structure, not observed validation loss.

## Suggested A100 Eval-Only Suite

Place the pulled model at:

```text
artifacts/prefix3500_global0008_seed42_final_model.int6.ptz
```

Then run:

```text
python3 records/track_10min_16mb/2026-05-01_LongCtx_NoQV_QK525_AsymLogit_LQERg32top4_TTTsegmentProbe/run_segment_probe_suite.py --model artifacts/prefix3500_global0008_seed42_final_model.int6.ptz --caseops-root /workspace/caseops_data --gpus 8 --modes no_ttt,base2060,p3500_lr0008,p3500_batch300_150,p3500_batch300_off,p3500_doc512_256
```

Use the same pulled model for all modes. This isolates TTT policy effects from
training/quantization noise.

## Data

Use the included Python downloader:

```text
python3 records/track_10min_16mb/2026-05-01_LongCtx_NoQV_QK525_AsymLogit_LQERg32top4_TTTsegmentProbe/download_caseops_data.py --local-dir /workspace/caseops_data
```

The script downloads the same CaseOps/SP8192 dataset layout expected by the
training code.
