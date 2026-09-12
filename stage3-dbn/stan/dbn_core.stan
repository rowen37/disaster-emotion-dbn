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
  vector<lower=0, upper=1>[N] B_obs;     // probability cell mean (0.5 placeholder when missing)
  vector<lower=1>[N] n_posts;

  int<lower=0, upper=1> use_weights;

  int<lower=1> M;
  matrix[K, M] B_spline;
  int<lower=1> P;
  matrix[T, P] T_spline;

  array[N] int<lower=0> lag_idx;

  int<lower=0, upper=1> has_jump;
  int<lower=1, upper=T> jump_start;      // derived from config (= 7), see docs

  int<lower=0, upper=1> has_group;       // 1 = enable typhoon-level nesting
  int<lower=1> G;                        // set to 1 when has_group=0 (placeholder, no parameters)
  array[J] int<lower=1, upper=G> group_of_event;

  int<lower=0> Gc;                       // number of event covariates (0 = none)
  matrix[J, Gc] ev_cov;
  array[J * Gc] int<lower=0, upper=1> ev_cov_miss;

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
  {
    vector[T] time_e_raw;
    vector[T] time_b_raw;
    for (t in 1:T) {
      time_e_raw[t] = dot_product(psi_e, T_spline[t]');
      time_b_raw[t] = dot_product(psi_b, T_spline[t]');
    }
    time_e_cent = time_e_raw - mean(time_e_raw);
    time_b_cent = time_b_raw - mean(time_b_raw);
  }
}
model {
  // ---------------- priors (per-line rationale in docs/parameter_justification.md) ----------------
  alpha_e ~ normal(0, 1.5);              // logit baseline +-3 covers p in (0.05, 0.95)
  alpha_b ~ normal(0, 1.5);
  psi_e ~ normal(0, 2);                  // spline coefficients: allows +-4 logit smooth variation
  psi_b ~ normal(0, 2);
  beta_e ~ normal(0, 0.8);               // weak stationarity: |beta| < 1 within 2.5 sigma
  beta_b ~ normal(0, 0.8);
  if (has_coupling == 1) {
    theta_arr ~ normal(0, 0.5);          // coupling coefficients: effect magnitude expected < 1
  }
  lambda_e ~ normal(0, 2);
  lambda_b ~ normal(0, 2);
  z_w_e ~ normal(0, 1);                  // non-centered track-point effects
  z_w_b ~ normal(0, 1);
  tau_w_e ~ cauchy(0, 1);                // standard weakly-informative prior for hierarchical SDs (half-Cauchy)
  tau_w_b ~ cauchy(0, 1);
  sigma_obs_e ~ cauchy(0, 1);
  sigma_proc_e ~ cauchy(0, 1);
  phi_b_beta ~ cauchy(0, 5);             // precision scale can reach tens; heavy tail avoids truncation
  gamma_e_cov ~ normal(0, 1);            // standardized covariates: effect per 1 SD
  gamma_b_cov ~ normal(0, 1);
  if (has_group == 1) {
    z_v_e_raw ~ normal(0, 1);            // non-centered typhoon effects
    z_v_b_raw ~ normal(0, 1);
    tau_v_e_arr ~ cauchy(0, 1);
    tau_v_b_arr ~ cauchy(0, 1);
  }
  if (has_jump == 1) {
    jump_raw ~ normal(0, 1.5);           // step magnitude on the baseline scale
  }
  if (has_slope == 1) {
    z_slope_e_raw ~ normal(0, 1);        // event-level spatial slope (non-centered)
    z_slope_b_raw ~ normal(0, 1);
    tau_slope_e_arr ~ cauchy(0, 1);      // same family as the other hierarchical SDs
    tau_slope_b_arr ~ cauchy(0, 1);
  }

  for (i in 1:N) {
    int j = event[i];
    int k = dist_bin[i];
    int t = time_bin[i];

    real f_e = dot_product(lambda_e, B_spline[k]');
    real f_b = dot_product(lambda_b, B_spline[k]');

    real cov_e = 0;
    real cov_b = 0;
    for (g in 1:Gc) {
      if (ev_cov_miss[(g - 1) * J + j] == 0) {
        cov_e += gamma_e_cov[g] * ev_cov[j, g];
        cov_b += gamma_b_cov[g] * ev_cov[j, g];
      }
    }

    real ar_e = 0;
    real ar_b = 0;
    real lag_e = 0;
    if (lag_idx[i] > 0) {
      ar_e  = beta_e * e_true[lag_idx[i]];
      ar_b  = beta_b * b_true[lag_idx[i]];
      lag_e = e_true[lag_idx[i]];
    }

    real jump_on = has_jump ? (t >= jump_start) : 0.0;
    real slope_e_j = 0;
    real slope_b_j = 0;
    if (has_slope == 1) {
      slope_e_j = tau_slope_e * z_slope_e_raw[j];
      slope_b_j = tau_slope_b * z_slope_b_raw[j];
    }
    real mu_e = alpha_e + time_e_cent[t] + f_e
                + v_e[group_of_event[j]] + tau_w_e * z_w_e[j] + cov_e
                + slope_e_j * dist_norm[k]
                + ar_e + jump_e * jump_on;
    real mu_b = alpha_b + time_b_cent[t] + f_b
                + v_b[group_of_event[j]] + tau_w_b * z_w_b[j] + cov_b
                + slope_b_j * dist_norm[k]
                + ar_b + theta1 * e_true[i]
                + (lag_idx[i] > 0 ? theta2 * lag_e : 0.0)
                + jump_b * jump_on;

    real se_scale = use_weights == 1 ? sqrt(n_posts[i]) : 1.0;
    if (E_miss[i] == 0) {
      E_obs[i] ~ normal(e_true[i], sigma_obs_e / se_scale);
    }
    e_true[i] ~ normal(mu_e, sigma_proc_e);

    if (B_miss[i] == 0) {
      real phi_eff = use_weights == 1 ? phi_b_beta * n_posts[i] : phi_b_beta;
      real p_b = inv_logit(b_true[i]);
      p_b = fmax(1e-6, fmin(1 - 1e-6, p_b));  // Beta support is the open interval (0,1); clamp against numerical overflow
      B_obs[i] ~ beta(p_b * phi_eff, (1 - p_b) * phi_eff);
    }
    b_true[i] ~ normal(mu_b, 1.0);       // behavior latent variance fixed at 1: scale is not separately identified, see docs
  }
}
