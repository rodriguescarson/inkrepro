# inkrepro

How much of an ink prediction survives changing the checkpoint?

An ink model turns a flattened surface into a probability map, and that map gets looked at, screened,
and sometimes published. What is almost never reported next to it is how much of it would still be
there if the same model had been trained with a different random seed, or stopped at a different
step. This tool measures that.

It needs no GPU and runs no inference. The eligible-scroll atlas already publishes, for all 340
surfaces across the eight eligible scrolls, the output of three ink checkpoints run at identical
settings. Comparing them is a download, a correlation, and a delete.

## What it reports

For a pair of predictions over the valid region of one surface:

- **`iou`** — of every pixel that at least one run called ink, the fraction both called ink. This is
  the number to read.
- `dice`, `pearson_r` — reported because they are what gets quoted. `pearson_r` is computed over a
  region that is mostly not ink, so it reads higher than the ink itself agrees.
- `frac_ink_a`, `frac_ink_b`, `mean_abs_diff`, `n_valid`.

Every pair is labelled by what actually differs between the two runs:

| `comparison` | meaning |
|---|---|
| `checkpoint` | different checkpoint, same reading direction. The one that answers the question. |
| `direction` | *same* checkpoint, layer order reversed. This reads the other face of the sheet, so the two are not meant to agree, and these rows must not be averaged in with the others. |
| `checkpoint+direction` | both differ. Reported for completeness, interpret with care. |

## Usage

```bash
pip install -e .

# two prediction TIFFs you already have
inkrepro pair a.tif b.tif --mask valid_mask.tif

# every published prediction for one surface, downloaded and compared
inkrepro surface PHerc0813 z4704_w080

# the whole atlas, one surface at a time, downloading and deleting as it goes
inkrepro population manifest.csv --out results/population.csv --resume
```

```bash
# render what the two checkpoints disagree about
inkrepro figure a.tif b.tif --mask valid_mask.tif --out fig.png \
  --label-a "seed42 step-020000" --label-b "seed43 step-060000"
```

`--resume` skips surfaces already present in the output, so an interrupted run continues rather than
starting over. One surface that fails to download does not stop the run; it is counted and reported
at the end.

## The figures

![checkpoint disagreement on an atlas surface](figures/p0813_checkpoints.png)

Black is ink both checkpoints called, red and blue are ink only one of them called, pale grey is
valid surface neither called ink. The detail window is chosen as the area with the *most ink*, not
the most disagreement — picking the window where the two runs differ most would be choosing the
evidence to fit the claim.

The ink/no-ink decision is always made at full resolution and only then downsampled for display.
Downsampling the probabilities first and thresholding afterwards answers a different question: a
blurred 0.4 and 0.6 average to 0.5 and appear as ink that neither run actually called.

## The masking rule

Rendered surface volumes are zero outside the traced sheet, and the ink models write zero there too.
Those pixels are most of the canvas on many surfaces, and including them inflates every agreement
score with pixels on which the two models were never asked anything. `inkrepro` always measures over
a valid region.

Pass `--mask valid_mask.tif` when you have it. That mask is authoritative and is used on its own,
*not* intersected with "both runs wrote something" — a pixel inside the valid region that a model
scored exactly 0.0 is a real prediction of "no ink", and dropping it would bias the comparison
toward the pixels both models already felt strongly about.

Without a mask the fallback is "both runs wrote a nonzero value", which is all a bare pair of TIFFs
makes available. It does discard genuine zeros. It is a convenience, not an equivalent, and the CLI
says so when it uses it.

## The evaluator is validated before it is used

A metric that has not been checked against known inputs cannot tell a real disagreement from its own
arithmetic. `tests/test_metrics.py` covers:

- **A/A** — a prediction against itself scores exactly 1.0, before and after uint8 round-tripping.
- **Planted perturbations** — agreement falls monotonically as noise of increasing known size is
  added, and a hand-counted label flip lands on the hand-computed IoU.
- **The masking rule** — two runs that anticorrelate on every real pixel must not score as agreeing
  because they both wrote zero across the empty 80% of the canvas.
- **Degenerate cases** — disjoint ink scores 0, no ink anywhere scores 1 rather than dividing by
  zero, mismatched shapes raise instead of silently cropping.
- **A float32 precision trap** — `np.full((8, 8), 0.1, np.float32).std()` is `7.45e-09`, not zero.
  An `== 0` guard against constant input therefore never fires, and the constant region reaches
  `corrcoef` and comes back as a divide-by-almost-zero rather than a clean NaN. The statistics are
  computed in float64 against a tolerance far below one uint8 step. This was a real bug in this
  code, caught by its own tests.

Run them with `pytest`. `RuntimeWarning` is an error in this project, so a silently degenerate
computation fails the suite rather than printing a warning nobody reads.

## What it found on the eligible-scroll atlas

Run over all 340 surfaces in `rodriguescarson/eligible-scroll-atlas-renders`, comparing the two
independently seeded 9.362 um checkpoints (`hybrid_3d2d-seed42/step-020000` and
`hybrid_3d2d-seed43/step-060000`) at the settings the atlas itself used:

![population distribution](figures/population.png)

| | median IoU |
|---|---|
| independent seeds, 335 atlas surfaces, 670 comparisons | **0.160** (IQR 0.127-0.195) |
| positive control: seed42 vs a soup derived from seed42 | 0.351 (2.20x) |
| curated on-sheet control w043 (PHerc0139), independent seeds | **0.409** |

Two checkpoints that differ only in training seed agree on about a sixth of the ink they call. The
median holds across all eight scrolls, from 0.132 (PHerc0268) to 0.191 (PHerc0813).

The positive control is the check that the measure means anything: `soup42_early3` is derived from
the seed42 run, so it is not an independent opinion, and it scores 2.2x higher. A metric that could
not tell those apart would not be worth reporting.

The curated control is the one that matters for interpretation. w043 is a hand-made on-sheet surface
from the labelled corpus, and the *same two checkpoints* reach 0.409 on it -- higher than 669 of the
670 atlas comparisons. So the low atlas numbers are not simply what these models always do. On a
surface that is on the papyrus they agree more than twice as well.

**Five of the 340 surfaces are missing, and they are exactly the five that passed the ink screen**
(`maps_held=True` in the manifest: PHerc0358 z11280_w020, PHerc0813 z5888_w020, z7696_w020,
z12496_w060, z13088_w040). Their maps are held back, so they returned 404. Since those are the
surfaces most likely to carry real ink, and real ink plausibly reproduces better, excluding them
may bias the population figure downward. The number is 335 surfaces, not 340, and not a claim about
the candidates.

Reproduce with:

```bash
inkrepro population manifest.csv --out results/population.csv --resume
python scripts/plot_population.py results/population.csv figures/population.png
```

## Limitations

- Agreement between checkpoints is not accuracy. Two models can agree on something that is not ink,
  and the published 9.362 um checkpoints share training data, so these numbers are an upper bound on
  how independent they really are.
- `soup42_early3` is derived from the seed42 run, so a `s42_020k` vs `soup42_early3` row is not two
  independent opinions.
- The available checkpoint pairs differ in training step as well as seed, so a single
  cross-checkpoint number mixes the two. Separating them needs matched-step pairs, which exist for
  `step-010000` on both seeds.
- `direction` rows answer a different question and are labelled, not excluded. Filter on the
  `comparison` column before computing any summary.

## Licence

MIT.
