# Corrections

This file records claims this repository made and then withdrew, with the evidence that
forced each change. The history is kept because a number that was quoted and later moved
is more useful to a reader than a number that was quietly edited.

## 2026-10-24 — the "curated on-sheet control" was training data (withdrawn)

**What was claimed.** Between 19 and 24 October the README, `results/SUMMARY.txt` and
`scripts/plot_population.py` reported a curated control: surface w043 of PHerc0139 scored
IoU 0.4093 between the two independently seeded checkpoints, "higher than 669 of the 670
atlas comparisons", and concluded that "on a surface that is on the papyrus they agree more
than twice as well". A figure, `figures/w043_checkpoints.png`, illustrated it.

**Why it is wrong.** w043 is a training segment for both checkpoints. The ink_9um label
dataset lists `pherc0139-w043` against source volume `20260112000000-w043_2026011217`, and
that is the exact segment the control inferred on. Both seed42 and seed43 train on the same
label set, so both had already seen that surface's ink. Two checkpoints agreeing on a
surface they were both trained on measures memorisation, not sheet placement, and no claim
about being "on the papyrus" follows from it.

Source: `https://huggingface.co/buckets/scrollprize/datasets/tree/ink_9um`, README table
"Aligned public 2.399 µm volumes", row `pherc0139-w043`. The model card at
`scrollprize/ink_9um` states the same count ("segments from PHerc. 0139 (9)").

**What replaces it.** Four PHerc0139 windings that appear in neither training list — w051,
w055, w058, w059. The trained-on PHerc0139 public segments are w029, w030, w035, w039,
w040, w041, w043, w044 and w045; the four replacements are in none of them. See the README
section "A held-out on-sheet control".

**The old number is not recoverable.** The full-resolution w043 maps were produced on a pod
that no longer exists, and only lossy `-ds8.jpg` previews remain locally. Those must never
reach a correlation, so 0.4093 cannot be re-derived or re-checked. It should be treated as
withdrawn rather than as a datum.

**Not published.** The claim reached this public repository but was never submitted to the
Progress Prize or posted to the Vesuvius forum. The drafts that carried it were corrected
before filing.

## 2026-10-24 — "the ink models write zero off-sheet" (corrected)

The masking section asserted that "the ink models write zero" outside the traced sheet.
Measured on PHerc0813 z4704_w080: the published `valid_mask.tif` covers 7,906,886 px, while
the union of the six maps' nonzero pixels covers 12,097,632 px, 1.530x as many. The
published mask is a strict subset of that union, and there are zero pixels inside the mask
where every run wrote 0. The models therefore write nonzero *outside* the mask, and the
no-mask fallback's error is that it adds off-sheet pixels, not that it discards genuine
zeros. The README now says this, and `src/inkrepro/metrics.py` carries the measurement.
