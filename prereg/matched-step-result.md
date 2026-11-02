# Result: how much of the disagreement is the seed, and how much is the training step?

Filed 2026-11-02, after running what `prereg/matched-step.md` registered on 2026-11-02 and
nothing else. The pre-registration committed to publishing whichever way this landed, including
the case where it undermines the headline. It did not undermine it, but it did not confirm it in
the way the registration expected either.

Reproduce with:

```
python -I scripts/matched_step.py results/matched_step.csv \
    results/matched_step_coverage_matched.csv prereg/matched_step_sample.csv
```

## What was run

`s43_020k` (`hybrid_3d2d-seed43/step-020000`) on the 40 frozen surfaces of
`prereg/matched_step_sample.csv`, forward and reverse, at the atlas's own settings, scored over
each surface's published `valid_mask.tif` with this repository's `inkrepro.metrics`.

Two controls ran before any of it was analysed.

**The environment control.** The atlas ran on a different machine and a different Python. This run
used Python 3.14.10 and torch 2.11.0+cu128, which is not the atlas's environment, and it pairs a
new map against old ones, so any environment-induced difference would land entirely on the new map.
`s42_020k` was therefore re-run here, where a published map of the same checkpoint already exists.
The two agree to **IoU 0.9994**, with a maximum difference of one uint8 step (1/255) and 0.005 per
cent of pixels on the other side of 0.5. The environment difference is ~17x smaller than the
smallest threshold the readout turns on, so the pairing is sound.

That control also caught a real defect. `infer.json` records that the atlas inferred on a
zero-copy **padded view** of each render (1200x6540 to 1216x6592, `scripts/padview.py` in the
atlas repository). The first attempt here skipped the padding and produced maps at the unpadded
size. Pairing those against the padded published maps would have put a pipeline artefact into the
result. The shapes disagreeing is what exposed it.

**The scorer control.** The sample's first surface carries its published mixed-pair IoU,
`0.13649655597278823`. The scoring path used here reproduces that value to the last digit before
any new number was computed.

Neither abort condition fired: the new maps are not degenerate (152 distinct values), and their
ink coverage sits 2.7 to 3.7 per cent below the range the published checkpoints span, against an
abort bound of a factor of two.

## The registered result

Primary readout, forward direction, median paired difference of matched-step IoU minus mixed-pair
IoU, two-sided sign test over the 40 pairs:

| direction | matched-step | mixed-pair | median difference | 95% CI | sign test |
|---|---|---|---|---|---|
| forward (primary) | 0.1241 | 0.1540 | **-0.0285** | -0.0406, -0.0149 | 34/40 negative, p = 8.4e-06 |
| reverse (secondary) | 0.1165 | 0.1484 | **-0.0335** | -0.0441, -0.0156 | 32/40 negative, p = 1.8e-04 |

This is the case the registration called the surprise: **matching the training step made the two
checkpoints agree less, not more.** The registration said that if this happened it would be
reported as the surprising result it is and not explained away. It is reported here first, before
any analysis that softens it.

## Why it happens, and why that matters more than the number

IoU between two binarised maps cannot exceed the smaller ink coverage divided by the larger. The
three runs call very different amounts of ink at a 0.5 threshold:

| | s42_020k | s43_020k | s43_060k |
|---|---|---|---|
| median `frac_ink`, forward | 0.0495 | 0.0245 | 0.0842 |
| median `frac_ink`, reverse | 0.0479 | 0.0272 | 0.0858 |

`s43_020k` calls about half as much ink as the reference and `s43_060k` about 1.7x as much, so in
the forward direction the matched-step pair starts with a median per-surface ceiling of 0.579
against the mixed pair's 0.613. That 0.034 gap is the same size as the -0.0285 effect.

**This explanation does not survive the reverse direction**, and saying so is the point of
reporting it. In reverse the ceiling slightly *favours* the matched-step pair (0.622 against
0.602) and yet the effect there is larger, -0.0335. The simple bound is therefore not the whole
mechanism; what coverage does to this measure is not captured by the ceiling alone.

## Coverage-matched re-scoring (exploratory, not registered)

`s42_020k` appears in both pairs, so both were re-scored with the other run thresholded to call
exactly as much ink as `s42_020k` does on that surface. This removes the ceiling difference and
leaves only whether two runs agree about *where* the ink is.

| direction | matched-step | mixed-pair | median difference | 95% CI | sign test |
|---|---|---|---|---|---|
| forward | 0.1589 | 0.1493 | **-0.0005** | -0.0127, +0.0090 | 20/20, p = 1.0 |
| reverse | 0.1349 | 0.1490 | **-0.0091** | -0.0221, +0.0049 | 16/24, p = 0.27 |

Once the two runs are made to call the same amount of ink, the matched-step pair and the mixed
pair agree equally well. Forward is as clean a null as the test can produce.

This analysis was chosen after seeing the registered result, so it carries none of the
registration's authority and no registered threshold is applied to it. It is reported alongside
the registered number, never instead of it.

## What this does and does not license

**It does** answer the objection the registration was written to answer. The published figure
cannot be dismissed as an artefact of comparing an undertrained checkpoint against a
better-trained one: remove the step mismatch and the disagreement does not shrink.

**It does not** show that the training step is irrelevant. It shows something narrower and, for
anyone reading an ink-agreement number, more useful: between these checkpoints the step changes
chiefly *how much* ink is called, not *where*, and a fixed-threshold IoU charges that difference
as spatial disagreement. The coverage confound this repository already documents at Spearman
rho = +0.829 is not a caveat attached to the measure. On this sample it is the dominant term.

**The honest summary of the headline** is that the published 16.0 per cent is a statement about
two checkpoints that disagree about ink coverage as much as they disagree about ink location, and
that is true whether or not their training steps match.

## Limitations

- 40 surfaces of 335, drawn across coverage quartiles. The registered sample is not a favourable
  subset (its mixed-pair median is 0.1540 against the population's 0.1598), but it is a sample.
- One new checkpoint, at one step, for one seed pair. Nothing here speaks to other steps.
- The coverage-matched threshold is chosen per surface from the reference run's own coverage, so
  it inherits whatever that run gets wrong about how much ink a surface carries.
- Inference ran at villa `4b3c728` on Python 3.14.10 / torch 2.11.0+cu128, not the atlas's
  environment. The control above bounds what that is worth, on one surface.
- The full environment, including three dependency versions that were *not* captured before the
  machine was released, is in `prereg/matched-step-environment.md`. The run log is
  `results/matched_step_run.log`.
