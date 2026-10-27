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
| JUMP_START | 7 | **Derived**: (0-(-72))/12+1 = 7, the first post-event bin [0,+12h); a naive `t>=8` would activate the jump one bin late |
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
| gamma_*_cov (covariates) | N(0, 0.5) | Standardized covariates receive a regularizing prior after the explicit VIF/condition-number audit |
| gamma_mod | N(0, 1) | Same |
| delta_mod (moderation) | N(0, 0.5) | Same scale as theta: moderation expected smaller than the main effect |
| jump_e/b | N(0, 1.5) | Step magnitude on the baseline scale (capped at ±3 logit) |
| tau_w_e/b (track-point/event SD) | half-Cauchy(0, 1) | Standard weakly-informative prior for hierarchical SDs; heavy tail allows large heterogeneity |
| tau_v_e/b (typhoon SD) | half-Cauchy(0, 1) | Same |
| sigma_obs_e / sigma_proc_e | half-Cauchy(0, 1) | Weakly-informative noise scales; separating the two avoids confounding observation noise with process dynamics |
| phi_b_beta (Beta precision base) | half-Cauchy(0, 5) | Precision can reach tens; heavy tail avoids truncation |

### Optimized landslide/debris-flow main model

The optimized main model uses `N(0, 0.5)` for standardized disaster intensity
and population/GDP terms.  This is a regularizing prior chosen because the explicit VIF/condition-number
audit detects remaining correlation, especially between event-level mean
population and GDP.

| Addition | Prior / construction | Rationale |
|---|---|---|
| rho_e / rho_b | N(0, 0.3) | Previous-time mean residual in adjacent radial rings; tighter than own-ring persistence to avoid spatial/temporal aliasing |
| population/GDP | log1p ring values; event mean (between) + ring-centered deviation (within); separate z scores | Separates comparisons between event locations from gradients within one event |
| posting presence | Bernoulli-logit hurdle with time, space, event, intensity, population/GDP and latent residual loadings | Empty cells are modeled rather than treated as ignorable missingness |
| positive post count | shifted NB2 for N-1 with the same confounder families and separate coefficients | Gives the positive component exact support 1,2,... and permits overdispersion |
| omega_occ / omega_count | N(0, 0.5) | Regularizes the MNAR dependence of observation/count on latent residual emotion and harmful-text states |
| phi_count | half-Cauchy(0, 5) | Weakly informative positive dispersion prior for the shifted NB2 count component |

The structured baselines contain temporal and spatial splines, event effects,
disaster intensity, and the four population/GDP components.  `beta_e`,
`beta_b`, `theta1`, and `theta2` operate on deviations from those baselines,
not on unadjusted latent levels.  Spatial propagation uses only adjacent rings
at the previous 12-hour bin; a same-time spatial lag is deliberately avoided
because it would create a simultaneous autoregressive system requiring a
different joint normalization.

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
| n_posts floor 1 | R: packing | The post count is shared by the emotion and behavior channels. The floor is only a Stan placeholder for a missing cell; its observation likelihood is skipped |

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
| Behavior probability input | Current analysis is explicitly locked to the same uncalibrated `behavior_prob` for all four hazards. There is no fallback. Thresholded labels and span presence never enter the DBN |
| Invalid behavior probability | Stop instead of silently dropping it; Stage 2 is required to provide one finite value in [0,1] per post |
| Invalid/missing EMO_probs | Keep the post for the behavior channel but mark its emotion value missing; never reinterpret missing emotion as zero |
| No 200-km filtering inside the parser | Distance filtering is done once by `grid_one_event`'s max_dist_km (single responsibility) |

## 7. Known interpretive caveats

- **fread truncation hazard**: data.table::fread can silently truncate at
  a broken-quote row (one 41,918-post file was once read as 579 rows by
  fread). The deterministic parser makes every dropped row accountable
  (see the drop_report).
- **Meaning of phi_b_beta**: with use_weights=1 it is the per-post
  precision base; cell precision = phi x N. It is not directly comparable
  to the phi of an equal-weight fit.
- **Calibration**: the current analysis does not calibrate. A future switch
  requires one independently fitted calibrator, all four hazards regenerated
  with `behavior_prob_calibrated`, and an explicit configuration change. Raw
  and calibrated scores are never mixed or selected per file.
