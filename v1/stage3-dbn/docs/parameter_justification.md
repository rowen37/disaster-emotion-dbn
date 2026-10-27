# Parameters and Constants (design principle)

**Principle: every parameter and constant must be justifiable.** This
document gives, item by item, the origin and rationale of every constant,
prior, and threshold in `stage3-dbn/`. Update this file before changing
any value.

## 1. Data binning constants (`config/hazards.R`)

| Constant | Value | Origin |
|---|---|---|
| TIME_MIN / TIME_MAX | -72 / +168 h | Collection window defined in the paper's Methods |
| BIN_HOURS | 12 | Paper Methods (20 time bins) |
| JUMP_START | 7 | **Derived**: (0-(-72))/12+1 = 7, the first post-event bin [0,+12h). The legacy `t>=8` was off by one |
| K / ring_km / max_dist_km | 10 rings x 5 km x 50 km; typhoon 20 x 10 km x 200 km | Paper Methods and Table S5 |
| df_dist / df_time | 6/8 (typhoon spatial 5) | Spline degrees of freedom in paper Table S5 |

## 2. Stan model priors (`stan/dbn_core.stan`)

| Parameter | Prior | Rationale |
|---|---|---|
| alpha_e/b | N(0, 1.5) | ±2σ = ±3 on the logit baseline corresponds to p ≈ (0.05, 0.95): covers plausible baselines without favoring either direction |
| psi_e/b (time spline coefficients) | N(0, 2) | Allows ±4 logit of smooth temporal variation; wider priors would confound the spline with the jump/intercept |
| lambda_e/b (space spline coefficients) | N(0, 2) | Same, spatial dimension |
| beta_e/b (autoregression) | N(0, 0.8) | Weak stationarity constraint: |β|<1 lies within about 1.25σ; does not presuppose persistence |
| theta1/theta2 (coupling) | N(0, 0.5) | Expected effect magnitude well below 1; symmetric prior does not presuppose direction |
| gamma_*_cov (covariates) | N(0, 1) | Covariates are standardized, so the prior says "a 1-SD covariate difference maps to at most ~1 logit" |
| gamma_mod | N(0, 1) | Same |
| delta_mod (moderation) | N(0, 0.5) | Same scale as theta: moderation expected smaller than the main effect |
| jump_e/b | N(0, 1.5) | Step magnitude on the baseline scale (capped at ±3 logit) |
| tau_w_e/b (track-point/event SD) | half-Cauchy(0, 1) | Standard weakly-informative prior for hierarchical SDs; heavy tail allows large heterogeneity |
| tau_v_e/b (typhoon SD) | half-Cauchy(0, 1) | Same |
| sigma_obs_e / sigma_proc_e | half-Cauchy(0, 1) | Weakly-informative noise scales; separating the two is a model fix (the legacy model shared one σ) |
| phi_b_beta (Beta precision base) | half-Cauchy(0, 5) | Precision can reach tens; heavy tail avoids truncation |

**Behavior latent variance fixed at 1.0** (`b_true ~ normal(mu_b, 1.0)`):
the scale of b is not simultaneously identifiable with theta1/beta_b
(scale confounding); fixing the variance at 1 is the standard
identification constraint, as stated in the paper's Methods.

## 3. Numerical constants

| Constant | Location | Rationale |
|---|---|---|
| 1e-6 | Stan: clamp after inv_logit | The Beta distribution has open support (0,1); the likelihood is undefined at the endpoints |
| epsilon = 0.001 | R: clamp before logit | logit of exactly 0/1 is ±∞; 0.001 is about ±6.9 on the logit scale, beyond any reasonable observation |
| B_obs missing placeholder 0.5 / E_obs placeholder 0 | R: packing | Placeholders never enter the likelihood (observation terms are skipped when E_miss/B_miss = 1); neutral values only satisfy Stan's data types |
| n_posts missing floor 1 | R: packing | Protects the weighting denominator; a cell with missing N has no posts and never enters the observation likelihood anyway |

## 4. Conditional parameter blocks (design principle in action)

With `has_jump=0`, `jump_raw` has size 0 (the parameter **does not exist**
in the fit); with `has_group=0`, `z_v_*_raw` and `tau_v_*_arr` have size
0. Transformed quantities `jump_e`/`tau_v_e` still appear in draws as
deterministic constants 0 - the zombie checks in the tests look at the
free-parameter names (`jump_raw` etc.) instead.

## 5. Fitting settings (`config/hazards.R`: fit_settings)

| Setting | Value | Rationale |
|---|---|---|
| algorithm | meanfield ADVI | NUTS is expensive at this size (2N latent variables); meanfield is known to understate posterior variance, so important conclusions need a NUTS cross-check (to do) |
| iter | 30000 | Final cap; with the tol=0.01 plateau criterion (at 20k most fits had not reached the plateau) |
| tol_rel_obj | 0.01 | Plateau criterion: relative ELBO improvement < 1%; runs distinguish "stopped by convergence" from "hit the iteration cap" |
| eta | 0.1 | Step size; 0.1 is stable on this model (0.3 once diverged on earthquake) |
| output_samples | 1000 | Standard number of draws for posterior summaries |
| seed | 123 | Reproducibility |

## 6. Data-processing rules (`R/typhoon_dedup.R`)

| Rule | Rationale |
|---|---|
| Post unique key = created_at + text + lon + lat | Strictest composite key available without exported Weibo IDs |
| Within one typhoon, attribute to the spatially nearest track point | Closest-approach principle: a post describes the situation at the moment nearest to that track point |
| No dedup across typhoons | A city hit by successive typhoons belongs to two distinct events (different time windows) |
| Parse drops: front/anchor/structure failures | Rows where the structured line parser cannot locate column boundaries (unescaped quotes in the source); dropped and counted, never silently |
| Parse drops: rows with empty raw_prob | The source data lacks a behavior probability (not scored); the model needs paired emotion+behavior observations |
| No 200-km filtering inside the parser | Distance filtering is done once by `grid_one_event`'s max_dist_km (single responsibility) |

## 7. Known interpretive caveats

- **Legacy fread truncation**: 401-1.csv holds 41,918 posts while the
  legacy RDS recorded only 579 (fread silently truncated at a broken-quote
  row). The legacy typhoon results rested on this truncated input; the
  deterministic parser in this repository makes every dropped row
  accountable (see the drop_report).
- **Meaning of phi_b_beta**: with use_weights=1 it is the per-post
  precision base; cell precision = phi x n. It is not directly comparable
  to the phi of the legacy equal-weight version.
