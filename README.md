# disaster-emotion-dbn

**From Weibo posts to spatiotemporal emotion–behavior dynamics: a three-stage
pipeline for four natural-hazard types in China (earthquake, typhoon,
landslide, debris flow).**

This repository releases the code of a complete research pipeline:

```
Stage 1  Weibo emotion classification (chinese-roberta-wwm-ext, SMP2020-EWECT)
             │  labels every post with 6 emotion probabilities (EMO_probs)
             ▼
Stage 2  Emotion-gated harm-span extraction (RoBERTa + emotion gate)
             │  per-post harm-description probability (raw_prob) + span text
             ▼
Stage 3  Dynamic Bayesian network over a space–time grid (Stan, ADVI)
             rings x 12-h bins around each event; AR(1) emotion/behavior
             latents with contemporaneous (θ₁) and lagged (θ₂) coupling;
             nested typhoon structure; ring-level moderators (population / GDP)
```

The two NLP stages produce, for every post, an **emotion channel**
(P(angry)+P(fear)+P(sad)) and a **behavior channel** (probability that the
post describes an actual harmful consequence). Stage 3 models how both
channels evolve in time and distance around each hazard event.

## Repository layout

```
├── stage1-sentiment/          # 6-class emotion classifier
│   ├── train.py               #   fine-tune on SMP2020-EWECT
│   ├── predict.py             #   annotate post CSVs with EMO / EMO_probs
│   ├── get_confusion_and_probs.py  #   confusion matrix + probability diagnostics
│   ├── count2020.py           #   training-set class distribution
│   └── eval_metrics.json      #   final validation metrics
├── stage2-span-extraction/    # emotion-gated harm-span extractor
│   ├── config.py              #   all hyperparameters and paths
│   ├── model.py               #   RoBERTaGateSpanExtractor
│   ├── data_utils.py          #   span-token alignment, dataset, masking augmentation
│   ├── train.py               #   joint loss (span start/end + weighted binary)
│   ├── predict.py             #   emotion-calibrated harm probability + span decoding
│   ├── eval_joint_span.py     #   joint strict EM + SQuAD-style overlap evaluation
│   ├── augment_data.py        #   positive-span oversampling
│   ├── geocount.py            #   geographic heat map of negative posts
│   └── eval_joint_results.json    # final evaluation numbers
└── stage3-dbn/                # spatiotemporal DBN (R + Stan)
    ├── config/hazards.R       #   single source of truth: hazard configs, fit settings, paths
    ├── stan/
    │   ├── dbn_core.stan      #   unified core model (4 hazards, conditional parameter blocks)
    │   └── dbn_moderator.stan #   core + ring-level moderator (population / GDP)
    ├── R/                     #   loader, gridding, fitting, summaries, figures, moderators,
    │                          #   typhoon dedup + deterministic CSV parser, consistency eval
    ├── runs/                  #   end-to-end run scripts (rebuild from raw, final refits, paper outputs)
    ├── tests/                 #   unit/smoke tests + typhoon nested-model subset test
    ├── docs/parameter_justification.md   # rationale for every constant and prior
    └── renv.lock              # R dependency lockfile
```

## Stage 1 — emotion classification

`chinese-roberta-wwm-ext` fine-tuned for 3 epochs on SMP2020-EWECT
(`usual_train.txt`, 6 classes: fear / neutral / sad / surprise / angry /
happy). The trained classifier annotates hazard-related Weibo CSVs with
`EMO` (argmax label) and `EMO_probs` (full probability vector).

Validation results (SMP2020 eval split, `eval_metrics.json`):
accuracy **0.772**, macro-F1 **0.737**.

```bash
python stage1-sentiment/train.py       # training (best macro-F1 checkpoint kept)
python stage1-sentiment/predict.py     # annotate a CSV of posts
```

Requires the [SMP2020-EWECT](https://github.com/smp2020/ect) dataset under
`smp2020_data/` and the `hfl/chinese-roberta-wwm-ext` weights.

## Stage 2 — emotion-gated harm-span extraction

`RoBERTaGateSpanExtractor`: the 6-dim emotion vector from Stage 1 is mapped
to a sigmoid gate that modulates token representations before three heads
(span start, span end, binary "contains harm description"). Training uses
negative-sample and happy-penalty loss weighting; prediction calibrates the
harm probability with the emotion distribution.

Joint evaluation on the held-out validation split
(`eval_joint_results.json`): binary harm detection F1 **0.983**; joint
strict exact-match (harm head must fire *and* boundaries must match)
P/R/F1 = 0.769 / 0.795 / **0.782**; SQuAD-style character-overlap F1
micro **0.571**, macro 0.606.

```bash
python stage2-span-extraction/augment_data.py --input <train.csv> --repeat 3
python stage2-span-extraction/train.py
python stage2-span-extraction/predict.py
python stage2-span-extraction/eval_joint_span.py
```

The behavior-channel probability (`raw_prob`/`calibrated_prob`) is the
input consumed by Stage 3.

## Stage 3 — spatiotemporal DBN

Posts of each event are aggregated into a grid of **K distance rings x 20
time bins** (-72 h to +168 h, 12-h bins); per cell the model observes the
negative-emotion share (logit scale) and the mean harm probability (Beta
likelihood, precision scaled by post count). Latent per-cell states follow
AR(1) dynamics with structured prior means (time/space natural splines,
event covariates, earthquake jump term), event-level random intercepts,
typhoon-level nesting (track points within a typhoon), and
emotion→behavior coupling θ₁ (contemporaneous) and θ₂ (lagged).
Inference: meanfield ADVI via cmdstanr (30k iterations, tol_rel_obj 0.01).

Key design points (all documented in `stage3-dbn/docs/parameter_justification.md`):

- **Conditional parameter blocks**: parameters that do not apply to a
  hazard (jump term, nesting) have size 0 and do not exist in the fit.
- **Deterministic typhoon dedup**: a post is attributed to the spatially
  nearest track point within its typhoon; overlaps across typhoons are
  legitimate observations for both. The line parser never silently
  truncates (unlike `fread` on unescaped quotes).
- **Post-count weighting**: a cell with n posts gets emotion SE scaled by
  1/√n and Beta precision φ·n (a 1-post cell no longer equals a
  1321-post cell).
- **Ring-level moderators**: population density / GDP per ring enter via
  `dbn_moderator.stan` (`gamma_mod` main effect, `delta_mod` moderation of
  the emotion→behavior coupling).

```bash
# tests (unit -> packing -> compile -> quick smoke fit)
Rscript stage3-dbn/tests/test_refactor.R
Rscript stage3-dbn/tests/test_typhoon_nested.R

# end-to-end reproduction
Rscript stage3-dbn/runs/rerun_typhoon_ruleB.R    # typhoon dedup (rule B) + baseline + moderators
Rscript stage3-dbn/runs/extract_all_modrings.R   # ring moderator matrices (pop/GDP, all hazards)
Rscript stage3-dbn/runs/rerun_all_from_raw.R     # rebuild EQ/LS/DF grids + baselines + 8 moderator fits
Rscript stage3-dbn/runs/rerun_final_30k.R        # final locked-setting rerun + evaluation + tables/figures
Rscript stage3-dbn/runs/run_slope_sensitivity.R  # earthquake random-slope sensitivity analysis

# paper figures
Rscript stage3-dbn/runs/export_figure2_data.R    # export plotting CSVs (+ FigS1/S2 supplementary maps)
Rscript stage3-dbn/runs/make_fig3_split_S1S2.R   # R-native supplementary figures
python stage3-dbn/runs/make_figures_v3.py        # unified Python figure rendering
```

R dependencies are pinned in `stage3-dbn/renv.lock` (tidyverse, cmdstanr,
data.table, terra/sf/exactextractr, ...); [CmdStan](https://mc-stan.org/cmdstan/)
is required.

## Data and third-party inputs (not redistributed)

| Input | Role | Source |
|---|---|---|
| SMP2020-EWECT `usual_train.txt` / `usual_eval_labeled.txt` | Stage 1 training/validation | SMP2020 challenge |
| `hfl/chinese-roberta-wwm-ext` | backbone for both NLP stages | Hugging Face |
| Annotated span training CSV (`text`, `span`, `EMO_probs`, `NONE`) | Stage 2 training | manual annotation |
| Per-event post CSVs (`lon, lat, created_at, time_gap, text, ..., EMO_probs, raw_prob`) | Stage 3 raw input | collected Weibo data |
| `event_features.csv` | event covariates + centers (mag/depth, volume/max_rain, typhoon level) | compiled per event |
| 2020 China 1-km population raster / 2020 GDP raster | ring-level moderators & map figures | WorldPop / gridded GDP |

## Path configuration

The scripts were extracted from the original research environment and keep
its absolute paths (e.g. `/Users/rogerbqw/Downloads/...` in
`stage1-sentiment/*.py` constants and `stage3-dbn/config/hazards.R::paths`
plus the `source(...)` lines of run scripts). Adjust these to your own
layout before running — they are deliberately left as-is so the released
code matches the code that produced the paper results byte for byte.

## Reproducibility

- All seeds fixed (PyTorch 42 / ADVI 123 / figure simulations 123).
- `renv.lock` pins the R environment; `requirements.txt` lists Python
  dependencies per stage.
- Stage 3 tests assert structural invariants (grid dimensions, lag
  pointers, RMSE >= MAE, absence of zombie parameters).

## License

No license has been chosen yet; all rights reserved until one is added.
