# Pre-registration: how much of the disagreement is the seed, and how much is the training step?

Filed 2026-11-02, before any inference was run. The sample is frozen in
`prereg/matched_step_sample.csv`, committed in the same change as this file.

## The problem with the headline number

This repository reports that two `ink_9um` checkpoints agree on a median 16.0 per cent of the ink
they call. The two checkpoints are `hybrid_3d2d-seed42/step-020000` and
`hybrid_3d2d-seed43/step-060000`. They differ in **two** ways at once: the training seed and the
training step. The published number therefore cannot say whether what it measures is seed variance,
an undertrained checkpoint compared against a better-trained one, or both. That is the first
objection a reader should raise, and the repository currently cannot answer it.

`scrollprize/ink_9um` publishes both seeds at 010000, 020000, 030000, 040000, 050000, 060000 and
075000, so a matched-step pair is available. `s42_020k` already exists for all 335 atlas surfaces,
so only **one** new checkpoint has to be run: `hybrid_3d2d-seed43/step-020000` (`s43_020k`).

## What is being run

`s43_020k` inference on the 40 surfaces in `prereg/matched_step_sample.csv`, forward and reverse,
at the atlas's own settings (`vesuvius.ink_detection.inference.infer` at villa `4b3c728`,
`--overlap 0.5 --blend-mode hann --direction both`), on the published
`surface-volumes.tar` renders, scored over each surface's published `valid_mask.tif`.

**Sample.** Ten surfaces from each quartile of ink coverage, drawn with a fixed seed (20261102),
so the sample spans the coverage confound rather than clustering where most surfaces sit. It
covers coverage 0.027 to 0.214 across six scrolls, and its mixed-pair median IoU is 0.1540 against
the full-population 0.1598, so it is not a favourable subset.

## The comparison

| pair | what differs |
|---|---|
| `s42_020k` vs `s43_020k` | seed only (new) |
| `s42_020k` vs `s43_060k` | seed and training step (already published) |

Both are measured on the same 40 surfaces, so the comparison is paired.

## Pre-registered readout, decided before the numbers exist

**Primary:** the median paired difference, matched-step IoU minus mixed-pair IoU, forward
direction, with a two-sided sign test over the 40 pairs.

- **median difference <= +0.010** -- the training step contributes little. The published 16.0 per
  cent is essentially a seed effect, and the headline stands as a statement about seed variance.
- **median difference >= +0.050** -- the step difference is carrying a large part of the
  disagreement. The published number then understates matched-step reproducibility and must be
  relabelled as "seed and step", not "seed".
- **between +0.010 and +0.050** -- partial. Report the number and the split; do not round the
  interpretation to either end.
- **A negative median difference** would mean the matched-step pair agrees *less* than the mixed
  pair, which no account of the measure predicts. If that happens it is reported as the surprising
  result it is, not explained away.

**Secondary, reported but not decisive:** the same difference in the reverse direction; and whether
the coverage confound (Spearman rho between coverage and IoU) holds for the matched-step pair, which
tests whether the confound is a property of the measure rather than of these two checkpoints.

**Abort conditions.** If `s43_020k` output is degenerate -- constant, empty, or with ink coverage
outside the range spanned by the existing checkpoints by more than a factor of two -- the run is
reported as failed rather than analysed, because that indicates a broken inference rather than a
finding.

**Commitment.** The result is published in this repository whichever way it lands, including the
case where it undermines the headline number.
