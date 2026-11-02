# The environment the matched-step run used

Recorded for `prereg/matched-step-result.md`. It is deliberately explicit about what was
captured and what was not, because the run's own machine no longer exists.

## What it ran on

| | |
|---|---|
| host | RunPod RTX 4090, 12 vCPU, SECURE cloud, `$0.89/hr` |
| image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| host CUDA | 13.2 |
| Python | 3.14.10, installed with `uv python install 3.14` into `/workspace/v314` |
| torch | 2.11.0+cu128 |
| villa | `4b3c72882`, `vesuvius` installed editable from `villa/vesuvius` |
| imagecodecs | 2026.10.10 |

`vesuvius` requires Python >=3.14,<3.15, which the image's 3.12.3 does not satisfy, so the venv
is not optional. Three packages had to be added on top of the `vesuvius` install before
`ink_detection.inference.infer` would run end to end: `tifffile` (imported at module level),
`einops` and `timm` (imported lazily when the network is built), and `imagecodecs` (only reached
when the first output TIFF is written with LZW, so it fails *after* a full inference).

## What was not captured, and what that costs

**The exact versions of `tifffile`, `einops` and `timm` were not recorded.** The pod was stopped
when the run finished, and a stopped RunPod pod stays bound to its machine; by the time the
environment record was being assembled another user held that GPU, so it could not be restarted
to run `uv pip freeze`. It was terminated rather than left billing for its disk.

What that does and does not undermine:

- It does **not** affect the reported numbers. Those are in `results/matched_step.csv` and
  `results/matched_step_coverage_matched.csv`, and the scoring that produced them runs anywhere
  from `scripts/matched_step.py`.
- It does **not** leave the environment unbounded. The environment control in
  `matched-step-result.md` re-ran a checkpoint whose map was produced in a different environment
  entirely and agreed to IoU 0.9994, with no pixel differing by more than one uint8 step. Whatever
  those three pins were worth, it is smaller than that.
- It **does** mean a bit-exact re-run of the inference cannot be guaranteed from this file alone.
  Anyone reproducing it should expect agreement at the level the control measured, not equality.

The lesson, recorded so it is not repeated: capture `uv pip freeze` into the results directory as
the first step of a run, not the last. An environment that exists only on rented hardware is not
a record.

## Reproducing

```
scripts/padview.py               # zero-copy padded view of a render, as the atlas used
scripts/matched_step_run.sh      # download, pad, infer s43_020k, fetch published maps, score
scripts/matched_step_score.py    # per-surface scoring via inkrepro.metrics
scripts/matched_step_covmatch.py # the exploratory coverage-matched re-score
scripts/matched_step.py          # the readout tables, from the two CSVs
```

`matched_step_run.sh` is transcribed from the script that ran, pod paths and all. It is kept
verbatim rather than tidied, because a cleaned-up version would be a different script from the
one that produced the numbers.
