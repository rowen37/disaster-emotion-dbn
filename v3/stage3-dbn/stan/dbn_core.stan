// =====================================================================
// Stage 3 - unified core DBN model.
// Design principle: parameters that do not apply to a hazard do not exist
// in the fit (conditional parameter blocks via zero-length arrays):
//   has_jump = 1 -> jump_raw has size 2 (earthquake step term), else 0
//   has_group = 1 -> nested random-intercept parameters exist (typhoon
//   only), else they do not
//
// Nesting (typhoon):
//   intercept = global alpha + typhoon-level v_g + track-point-level w_j
//   -- track points of one typhoon share v_g instead of being treated as
//      independent events (fixes pseudo-replication)
//   -- v_g ~ N(0, tau_v), w_j ~ N(0, tau_w), non-centered parameterization
//
// Rationale for every constant and prior: docs/parameter_justification.md
// =====================================================================
data {
  int<lower=1> N;
  int<lower=1> J;                        // number of track points / events
  int<lower=1> K;
  int<lower=1> T;

  array[N] int<lower=1, upper=J> event;
  array[N] int<lower=1, upper=K> dist_bin;
  array[N] int<lower=1, upper=T> time_bin;

  array[N] int<lower=0, upper=1> E_miss;
  array[N] int<lower=0, upper=1> B_miss;
  vector[N] E_obs;                       // logit cell mean (0 placeholder when missing, see docs)
  vector<lower=0, upper=1>[N] B_obs;     // raw model-score cell mean (0.5 placeholder when missing)
  vector<lower=1>[N] n_posts;           // posts in the cell, shared by both channels
  array[N] int<lower=0, upper=1> post_present;
  array[N] int<lower=0> post_count;

  int<lower=0, upper=1> use_weights;

  int<lower=1> M;
  matrix[K, M] B_spline;
  int<lower=1> P;
  matrix[T, P] T_spline;

  array[N] int<lower=0> lag_idx;
  array[N, 2] int<lower=0, upper=N> spatial_lag_idx;
  array[N] int<lower=0, upper=2> spatial_lag_n;

  int<lower=0, upper=1> has_jump;
  int<lower=1, upper=T> jump_start;      // derived from config (= 7), see docs

  int<lower=0, upper=1> has_group;       // 1 = enable typhoon-level nesting
  int<lower=1> G;                        // set to 1 when has_group=0 (placeholder, no parameters)
  array[J] int<lower=1, upper=G> group_of_event;

  int<lower=0> Gc;                       // number of event covariates (0 = none)
  matrix[J, Gc] ev_cov;
  array[J * Gc] int<lower=0, upper=1> ev_cov_miss;

  int<lower=1> Cc;                       // pop/GDP between/within components
  matrix[N, Cc] context_cov;
  real<lower=0> covariate_prior_sd;

  int<lower=0, upper=1> has_slope;       // 1 = event-level random spatial slope (sensitivity analysis)
  vector[K] dist_norm;                   // centered normalized distance ([-1,1], 0 = middle ring)

  int<lower=0, upper=1> has_coupling;    // 0 = no-coupling control (theta1/theta2 absent from the fit)
}
parameters {
  real alpha_e;
  real alpha_b;

  vector[P] psi_e;
  vector[P] psi_b;

  real beta_e;
  real beta_b;
  real rho_e;                            // previous-time adjacent-ring propagation
  real rho_b;
  vector[2 * has_coupling] theta_arr;    // [theta1, theta2]; absent when has_coupling=0

  vector[M] lambda_e;
  vector[M] lambda_b;

  vector[J] z_w_e;                       // track-point-level deviation (non-centered)
  vector[J] z_w_b;
  real<lower=0> tau_w_e;
  real<lower=0> tau_w_b;

  vector[has_group * G] z_v_e_raw;           // typhoon-level deviation; length 0 when has_group=0 (absent)
  vector[has_group * G] z_v_b_raw;
  vector<lower=0>[has_group] tau_v_e_arr;
  vector<lower=0>[has_group] tau_v_b_arr;

  real<lower=0> sigma_obs_e;
  real<lower=0> sigma_proc_e;
  real<lower=0> phi_b_beta;

  vector[N] e_true;
  vector[N] b_true;

  vector[Gc] gamma_e_cov;
  vector[Gc] gamma_b_cov;
  vector[Cc] gamma_e_context;
  vector[Cc] gamma_b_context;

  // Hurdle observation process: cell presence plus positive post count.
  real alpha_occ;
  real alpha_count;
  vector[P] psi_occ;
  vector[P] psi_count;
  vector[M] lambda_occ;
  vector[M] lambda_count;
  vector[J] z_w_occ;
  vector[J] z_w_count;
  real<lower=0> tau_w_occ;
  real<lower=0> tau_w_count;
  vector[Gc] gamma_occ_cov;
  vector[Gc] gamma_count_cov;
  vector[Cc] gamma_occ_context;
  vector[Cc] gamma_count_context;
  vector[2] omega_occ;                   // dependence on emotion/behavior residual states
  vector[2] omega_count;
  real<lower=0> phi_count;

  array[2 * has_jump] real jump_raw;     // earthquake step term; length 0 when has_jump=0 (absent)

  vector[has_slope * J] z_slope_e_raw;   // event-level spatial slope deviation; absent when has_slope=0
  vector[has_slope * J] z_slope_b_raw;
  vector<lower=0>[has_slope] tau_slope_e_arr;
  vector<lower=0>[has_slope] tau_slope_b_arr;
}
transformed parameters {
  // typhoon-level intercepts: identically 0 when has_group=0 (guaranteed by
  // the conditional block, no constant magic)
  vector[G] v_e = rep_vector(0, G);
  vector[G] v_b = rep_vector(0, G);
  real tau_v_e = 0;
  real tau_v_b = 0;
  if (has_group == 1) {
    for (g in 1:G) {
      v_e[g] = tau_v_e_arr[1] * z_v_e_raw[g];
      v_b[g] = tau_v_b_arr[1] * z_v_b_raw[g];
    }
    tau_v_e = tau_v_e_arr[1];
    tau_v_b = tau_v_b_arr[1];
  }

  real jump_e = 0;
  real jump_b = 0;
  if (has_jump == 1) {
    jump_e = jump_raw[1];
    jump_b = jump_raw[2];
  }

  real theta1 = 0;
  real theta2 = 0;
  if (has_coupling == 1) {
    theta1 = theta_arr[1];
    theta2 = theta_arr[2];
  }

  real tau_slope_e = 0;
  real tau_slope_b = 0;
  if (has_slope == 1) {
    tau_slope_e = tau_slope_e_arr[1];
    tau_slope_b = tau_slope_b_arr[1];
  }

  vector[T] time_e_cent;
  vector[T] time_b_cent;
  vector[T] time_occ_cent;
  vector[T] time_count_cent;
  {
    vector[T] time_e_raw;
    vector[T] time_b_raw;
    vector[T] time_occ_raw;
    vector[T] time_count_raw;
    for (t in 1:T) {
      time_e_raw[t] = dot_product(psi_e, T_spline[t]');
      time_b_raw[t] = dot_product(psi_b, T_spline[t]');
      time_occ_raw[t] = dot_product(psi_occ, T_spline[t]');
      time_count_raw[t] = dot_product(psi_count, T_spline[t]');
    }
    time_e_cent = time_e_raw - mean(time_e_raw);
    time_b_cent = time_b_raw - mean(time_b_raw);
    time_occ_cent = time_occ_raw - mean(time_occ_raw);
    time_count_cent = time_count_raw - mean(time_count_raw);
  }
}
model {
  // Baselines are local model-block variables, so thousands of redundant
  // baseline values are not written to every posterior draw.
  vector[N] base_e;
  vector[N] base_b;
  vector[N] base_occ;
  vector[N] base_count;

  // ---------------- regularized priors ----------------
  alpha_e ~ normal(0, 1.5);
  alpha_b ~ normal(0, 1.5);
  psi_e ~ normal(0, 2);
  psi_b ~ normal(0, 2);
  beta_e ~ normal(0, 0.8);
  beta_b ~ normal(0, 0.8);
  rho_e ~ normal(0, 0.3);
  rho_b ~ normal(0, 0.3);
  if (has_coupling == 1) theta_arr ~ normal(0, 0.5);
  lambda_e ~ normal(0, 2);
  lambda_b ~ normal(0, 2);
  z_w_e ~ normal(0, 1);
  z_w_b ~ normal(0, 1);
  tau_w_e ~ cauchy(0, 1);
  tau_w_b ~ cauchy(0, 1);
  sigma_obs_e ~ cauchy(0, 1);
  sigma_proc_e ~ cauchy(0, 1);
  phi_b_beta ~ cauchy(0, 5);
  gamma_e_cov ~ normal(0, covariate_prior_sd);
  gamma_b_cov ~ normal(0, covariate_prior_sd);
  gamma_e_context ~ normal(0, covariate_prior_sd);
  gamma_b_context ~ normal(0, covariate_prior_sd);

  alpha_occ ~ normal(0, 2.5);
  alpha_count ~ normal(0, 2.5);
  psi_occ ~ normal(0, 1);
  psi_count ~ normal(0, 1);
  lambda_occ ~ normal(0, 1);
  lambda_count ~ normal(0, 1);
  z_w_occ ~ normal(0, 1);
  z_w_count ~ normal(0, 1);
  tau_w_occ ~ cauchy(0, 1);
  tau_w_count ~ cauchy(0, 1);
  gamma_occ_cov ~ normal(0, covariate_prior_sd);
  gamma_count_cov ~ normal(0, covariate_prior_sd);
  gamma_occ_context ~ normal(0, covariate_prior_sd);
  gamma_count_context ~ normal(0, covariate_prior_sd);
  omega_occ ~ normal(0, 0.5);
  omega_count ~ normal(0, 0.5);
  phi_count ~ cauchy(0, 5);

  if (has_group == 1) {
    z_v_e_raw ~ normal(0, 1);
    z_v_b_raw ~ normal(0, 1);
    tau_v_e_arr ~ cauchy(0, 1);
    tau_v_b_arr ~ cauchy(0, 1);
  }
  if (has_jump == 1) jump_raw ~ normal(0, 1.5);
  if (has_slope == 1) {
    z_slope_e_raw ~ normal(0, 1);
    z_slope_b_raw ~ normal(0, 1);
    tau_slope_e_arr ~ cauchy(0, 1);
    tau_slope_b_arr ~ cauchy(0, 1);
  }

  // Structured baselines contain every measured confounder.  Dynamic
  // parameters below operate only on deviations from these baselines.
  for (i in 1:N) {
    int j = event[i];
    int k = dist_bin[i];
    int t = time_bin[i];
    real cov_e = 0;
    real cov_b = 0;
    real cov_occ = 0;
    real cov_count = 0;
    real slope_e_j = 0;
    real slope_b_j = 0;
    real jump_on = has_jump ? (t >= jump_start) : 0.0;

    for (g in 1:Gc) {
      if (ev_cov_miss[(g - 1) * J + j] == 0) {
        cov_e += gamma_e_cov[g] * ev_cov[j, g];
        cov_b += gamma_b_cov[g] * ev_cov[j, g];
        cov_occ += gamma_occ_cov[g] * ev_cov[j, g];
        cov_count += gamma_count_cov[g] * ev_cov[j, g];
      }
    }
    if (has_slope == 1) {
      slope_e_j = tau_slope_e * z_slope_e_raw[j];
      slope_b_j = tau_slope_b * z_slope_b_raw[j];
    }

    base_e[i] = alpha_e + time_e_cent[t]
      + dot_product(lambda_e, B_spline[k]')
      + v_e[group_of_event[j]] + tau_w_e * z_w_e[j] + cov_e
      + dot_product(gamma_e_context, context_cov[i]')
      + slope_e_j * dist_norm[k] + jump_e * jump_on;
    base_b[i] = alpha_b + time_b_cent[t]
      + dot_product(lambda_b, B_spline[k]')
      + v_b[group_of_event[j]] + tau_w_b * z_w_b[j] + cov_b
      + dot_product(gamma_b_context, context_cov[i]')
      + slope_b_j * dist_norm[k] + jump_b * jump_on;
    base_occ[i] = alpha_occ + time_occ_cent[t]
      + dot_product(lambda_occ, B_spline[k]')
      + tau_w_occ * z_w_occ[j] + cov_occ
      + dot_product(gamma_occ_context, context_cov[i]');
    base_count[i] = alpha_count + time_count_cent[t]
      + dot_product(lambda_count, B_spline[k]')
      + tau_w_count * z_w_count[j] + cov_count
      + dot_product(gamma_count_context, context_cov[i]');
  }

  for (i in 1:N) {
    real e_lag_resid = 0;
    real b_lag_resid = 0;
    real spatial_e_resid = 0;
    real spatial_b_resid = 0;
    real e_resid;
    real b_resid;
    real mu_e;
    real mu_b;
    real eta_occ;
    real eta_count;

    if (lag_idx[i] > 0) {
      e_lag_resid = e_true[lag_idx[i]] - base_e[lag_idx[i]];
      b_lag_resid = b_true[lag_idx[i]] - base_b[lag_idx[i]];
    }
    if (spatial_lag_n[i] > 0) {
      for (q in 1:spatial_lag_n[i]) {
        int s = spatial_lag_idx[i, q];
        spatial_e_resid += e_true[s] - base_e[s];
        spatial_b_resid += b_true[s] - base_b[s];
      }
      spatial_e_resid /= spatial_lag_n[i];
      spatial_b_resid /= spatial_lag_n[i];
    }

    mu_e = base_e[i] + beta_e * e_lag_resid + rho_e * spatial_e_resid;
    e_true[i] ~ normal(mu_e, sigma_proc_e);
    e_resid = e_true[i] - base_e[i];

    mu_b = base_b[i] + beta_b * b_lag_resid
      + theta1 * e_resid + theta2 * e_lag_resid
      + rho_b * spatial_b_resid;
    b_true[i] ~ normal(mu_b, 1.0);
    b_resid = b_true[i] - base_b[i];

    if (E_miss[i] == 0) {
      real se_scale = use_weights == 1 ? sqrt(n_posts[i]) : 1.0;
      E_obs[i] ~ normal(e_true[i], sigma_obs_e / se_scale);
    }
    if (B_miss[i] == 0) {
      real phi_eff = use_weights == 1 ? phi_b_beta * n_posts[i] : phi_b_beta;
      real p_b = inv_logit(b_true[i]);
      p_b = fmax(1e-6, fmin(1 - 1e-6, p_b));
      B_obs[i] ~ beta(p_b * phi_eff, (1 - p_b) * phi_eff);
    }

    // MNAR hurdle: empty cells still inform the latent emotion/behavior states
    // through their probability of being observed; positive counts use a
    // shifted negative-binomial model so support is exactly 1,2,... .
    eta_occ = base_occ[i] + omega_occ[1] * e_resid + omega_occ[2] * b_resid;
    post_present[i] ~ bernoulli_logit(eta_occ);
    if (post_present[i] == 1) {
      eta_count = base_count[i]
        + omega_count[1] * e_resid + omega_count[2] * b_resid;
      post_count[i] - 1 ~ neg_binomial_2_log(eta_count, phi_count);
    }
  }
}
