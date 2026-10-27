// Supplemental context model built on the current original_no_five DBN.
// Population density and GDP are fitted jointly.  Each standardized context
// variable has an E main effect, a B main effect, and interactions with both
// synchronous (theta1) and lagged (theta2) emotion-to-behavior coupling.

data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> K;
  int<lower=1> T;

  array[N] int<lower=1, upper=J> event;
  array[N] int<lower=1, upper=K> dist_bin;
  array[N] int<lower=1, upper=T> time_bin;

  array[N] int<lower=0, upper=1> E_miss;
  array[N] int<lower=0, upper=1> B_miss;
  vector[N] E_obs;
  vector<lower=0, upper=1>[N] B_obs;
  vector<lower=1>[N] n_posts;
  int<lower=0, upper=1> use_weights;

  int<lower=1> M;
  matrix[K, M] B_spline;
  int<lower=1> P;
  matrix[T, P] T_spline;
  array[N] int<lower=0> lag_idx;

  int<lower=0, upper=1> has_jump;
  int<lower=1, upper=T> jump_start;

  int<lower=0, upper=1> has_group;
  int<lower=1> G;
  array[J] int<lower=1, upper=G> group_of_event;

  int<lower=0> Gc;
  matrix[J, Gc] ev_cov;
  array[J * Gc] int<lower=0, upper=1> ev_cov_miss;

  matrix[J, K] pop_ring;
  matrix[J, K] gdp_ring;
}

parameters {
  real alpha_e;
  real alpha_b;
  vector[P] psi_e;
  vector[P] psi_b;
  real beta_e;
  real beta_b;
  real theta1;
  real theta2;
  vector[M] lambda_e;
  vector[M] lambda_b;

  vector[J] z_w_e;
  vector[J] z_w_b;
  real<lower=0> tau_w_e;
  real<lower=0> tau_w_b;

  vector[has_group * G] z_v_e_raw;
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

  // [1] population density, [2] GDP.  Both inputs are log1p transformed
  // and standardized within a hazard over all event-ring combinations.
  vector[2] gamma_e_context;
  vector[2] gamma_b_context;
  vector[2] delta1_context;
  vector[2] delta2_context;

  array[2 * has_jump] real jump_raw;
}

transformed parameters {
  vector[G] v_e = rep_vector(0, G);
  vector[G] v_b = rep_vector(0, G);
  if (has_group == 1) {
    for (g in 1:G) {
      v_e[g] = tau_v_e_arr[1] * z_v_e_raw[g];
      v_b[g] = tau_v_b_arr[1] * z_v_b_raw[g];
    }
  }

  real jump_e = 0;
  real jump_b = 0;
  if (has_jump == 1) {
    jump_e = jump_raw[1];
    jump_b = jump_raw[2];
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
  alpha_e ~ normal(0, 1.5);
  alpha_b ~ normal(0, 1.5);
  psi_e ~ normal(0, 2);
  psi_b ~ normal(0, 2);
  beta_e ~ normal(0, 0.8);
  beta_b ~ normal(0, 0.8);
  theta1 ~ normal(0, 0.5);
  theta2 ~ normal(0, 0.5);
  lambda_e ~ normal(0, 2);
  lambda_b ~ normal(0, 2);
  z_w_e ~ normal(0, 1);
  z_w_b ~ normal(0, 1);
  tau_w_e ~ cauchy(0, 1);
  tau_w_b ~ cauchy(0, 1);
  sigma_obs_e ~ cauchy(0, 1);
  sigma_proc_e ~ cauchy(0, 1);
  phi_b_beta ~ cauchy(0, 5);
  gamma_e_cov ~ normal(0, 1);
  gamma_b_cov ~ normal(0, 1);

  gamma_e_context ~ normal(0, 1);
  gamma_b_context ~ normal(0, 1);
  delta1_context ~ normal(0, 0.5);
  delta2_context ~ normal(0, 0.5);

  if (has_group == 1) {
    z_v_e_raw ~ normal(0, 1);
    z_v_b_raw ~ normal(0, 1);
    tau_v_e_arr ~ cauchy(0, 1);
    tau_v_b_arr ~ cauchy(0, 1);
  }
  if (has_jump == 1) jump_raw ~ normal(0, 1.5);

  for (i in 1:N) {
    int j = event[i];
    int k = dist_bin[i];
    int t = time_bin[i];
    real f_e = dot_product(lambda_e, B_spline[k]');
    real f_b = dot_product(lambda_b, B_spline[k]');
    real cov_e = 0;
    real cov_b = 0;
    real ar_e = 0;
    real ar_b = 0;
    real lag_e = 0;
    real jump_on = has_jump ? (t >= jump_start) : 0.0;
    vector[2] m;
    m[1] = pop_ring[j, k];
    m[2] = gdp_ring[j, k];

    for (g in 1:Gc) {
      if (ev_cov_miss[(g - 1) * J + j] == 0) {
        cov_e += gamma_e_cov[g] * ev_cov[j, g];
        cov_b += gamma_b_cov[g] * ev_cov[j, g];
      }
    }
    if (lag_idx[i] > 0) {
      ar_e = beta_e * e_true[lag_idx[i]];
      ar_b = beta_b * b_true[lag_idx[i]];
      lag_e = e_true[lag_idx[i]];
    }

    {
      real mu_e = alpha_e + time_e_cent[t] + f_e
                  + v_e[group_of_event[j]] + tau_w_e * z_w_e[j] + cov_e
                  + dot_product(gamma_e_context, m)
                  + ar_e + jump_e * jump_on;
      real theta1_local = theta1 + dot_product(delta1_context, m);
      real theta2_local = theta2 + dot_product(delta2_context, m);
      real mu_b = alpha_b + time_b_cent[t] + f_b
                  + v_b[group_of_event[j]] + tau_w_b * z_w_b[j] + cov_b
                  + dot_product(gamma_b_context, m)
                  + ar_b + theta1_local * e_true[i]
                  + (lag_idx[i] > 0 ? theta2_local * lag_e : 0.0)
                  + jump_b * jump_on;
      real se_scale = use_weights == 1 ? sqrt(n_posts[i]) : 1.0;

      if (E_miss[i] == 0) {
        E_obs[i] ~ normal(e_true[i], sigma_obs_e / se_scale);
      }
      e_true[i] ~ normal(mu_e, sigma_proc_e);

      if (B_miss[i] == 0) {
        real phi_eff = use_weights == 1 ? phi_b_beta * n_posts[i] : phi_b_beta;
        real p_b = inv_logit(b_true[i]);
        p_b = fmax(1e-6, fmin(1 - 1e-6, p_b));
        B_obs[i] ~ beta(p_b * phi_eff, (1 - p_b) * phi_eff);
      }
      b_true[i] ~ normal(mu_b, 1.0);
    }
  }
}
