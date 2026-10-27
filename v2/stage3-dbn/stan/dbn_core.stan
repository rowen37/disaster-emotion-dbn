// =====================================================================
// 统一核心模型 —— 设计原则：不适用的参数在拟合中不存在
//
// 条件参数块（零长度数组实现）：
//   has_jump = 1 时 jump_raw 尺寸为 2（地震专属阶跃项），否则为 0（参数不存在）
//   has_group = 1 时嵌套随机截距参数存在（台风专属），否则不存在
//
// 嵌套结构（台风）：
//   截距 = 全局 α + 台风级 v_g + 路径点级 w_j
//   —— 同一台风的路径点共享 v_g，不再被当作独立事件（修复伪复制）
//   —— v_g ~ N(0, tau_v)，w_j ~ N(0, tau_w)，非中心化参数化
//
// 所有常数与先验的理由见 docs/parameter_justification.md
// =====================================================================
data {
  int<lower=1> N;
  int<lower=1> J;                        // 路径点/事件数
  int<lower=1> K;
  int<lower=1> T;

  array[N] int<lower=1, upper=J> event;
  array[N] int<lower=1, upper=K> dist_bin;
  array[N] int<lower=1, upper=T> time_bin;

  array[N] int<lower=0, upper=1> E_miss;
  array[N] int<lower=0, upper=1> B_miss;
  vector[N] E_obs;                       // logit 格子均值（缺失处置 0，见文档）
  vector<lower=0, upper=1>[N] B_obs;     // 概率格子均值（缺失处置 0.5，见文档）
  vector<lower=1>[N] n_posts;

  int<lower=0, upper=1> use_weights;

  int<lower=1> M;
  matrix[K, M] B_spline;
  int<lower=1> P;
  matrix[T, P] T_spline;

  array[N] int<lower=0> lag_idx;

  int<lower=0, upper=1> has_jump;
  int<lower=1, upper=T> jump_start;      // 由 config 推导（=7），见文档

  int<lower=0, upper=1> has_group;       // 1 = 启用台风级嵌套
  int<lower=1> G;                        // has_group=0 时置 1（占位，不产生参数）
  array[J] int<lower=1, upper=G> group_of_event;

  int<lower=0> Gc;                       // 事件协变量个数（0 = 无协变量）
  matrix[J, Gc] ev_cov;
  array[J * Gc] int<lower=0, upper=1> ev_cov_miss;

  int<lower=0, upper=1> has_slope;       // 1 = 启用事件级空间随机斜率（敏感性分析）
  vector[K] dist_norm;                   // 归一化中心化距离（[-1,1]，0=中点环）

  int<lower=0, upper=1> has_coupling;    // 0 = 无耦合对照（θ₁/θ₂ 不存在，导师要求）
}
parameters {
  real alpha_e;
  real alpha_b;

  vector[P] psi_e;
  vector[P] psi_b;

  real beta_e;
  real beta_b;
  vector[2 * has_coupling] theta_arr;    // [θ₁, θ₂]；has_coupling=0 时不存在

  vector[M] lambda_e;
  vector[M] lambda_b;

  vector[J] z_w_e;                       // 路径点级偏差（非中心化）
  vector[J] z_w_b;
  real<lower=0> tau_w_e;
  real<lower=0> tau_w_b;

  vector[has_group * G] z_v_e_raw;           // 台风级偏差；has_group=0 时长度 0（不存在）
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

  array[2 * has_jump] real jump_raw;     // 地震阶跃项；has_jump=0 时长度 0（不存在）

  vector[has_slope * J] z_slope_e_raw;   // 事件级空间斜率偏差；has_slope=0 时不存在
  vector[has_slope * J] z_slope_b_raw;
  vector<lower=0>[has_slope] tau_slope_e_arr;
  vector<lower=0>[has_slope] tau_slope_b_arr;
}
transformed parameters {
  // 台风级截距：has_group=0 时恒为 0（无常量魔法，由条件块保证）
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
  // ---------------- 先验（逐条理由见 docs/parameter_justification.md）----------------
  alpha_e ~ normal(0, 1.5);              // logit 基线 ±3 覆盖 p∈(0.05,0.95)
  alpha_b ~ normal(0, 1.5);
  psi_e ~ normal(0, 2);                  // 样条系数：允许 ±4 logit 的平滑起伏
  psi_b ~ normal(0, 2);
  beta_e ~ normal(0, 0.8);               // 平稳性弱约束 |β|<1 的 2.5σ 内
  beta_b ~ normal(0, 0.8);
  if (has_coupling == 1) {
    theta_arr ~ normal(0, 0.5);          // 耦合系数：效应量级预期 <1
  }
  lambda_e ~ normal(0, 2);
  lambda_b ~ normal(0, 2);
  z_w_e ~ normal(0, 1);                  // 非中心化路径点效应
  z_w_b ~ normal(0, 1);
  tau_w_e ~ cauchy(0, 1);                // 层次SD标准弱信息先验（half-Cauchy）
  tau_w_b ~ cauchy(0, 1);
  sigma_obs_e ~ cauchy(0, 1);
  sigma_proc_e ~ cauchy(0, 1);
  phi_b_beta ~ cauchy(0, 5);             // 精度尺度数量级可到数十
  gamma_e_cov ~ normal(0, 1);            // 标准化协变量：单位SD对应的效应
  gamma_b_cov ~ normal(0, 1);
  if (has_group == 1) {
    z_v_e_raw ~ normal(0, 1);            // 非中心化台风效应
    z_v_b_raw ~ normal(0, 1);
    tau_v_e_arr ~ cauchy(0, 1);
    tau_v_b_arr ~ cauchy(0, 1);
  }
  if (has_jump == 1) {
    jump_raw ~ normal(0, 1.5);           // 阶跃幅度与基线同尺度
  }
  if (has_slope == 1) {
    z_slope_e_raw ~ normal(0, 1);        // 事件级空间斜率（非中心化）
    z_slope_b_raw ~ normal(0, 1);
    tau_slope_e_arr ~ cauchy(0, 1);      // 与其他层次SD同族
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
      p_b = fmax(1e-6, fmin(1 - 1e-6, p_b));  // Beta 支撑集 (0,1) 开集，防数值越界
      B_obs[i] ~ beta(p_b * phi_eff, (1 - p_b) * phi_eff);
    }
    b_true[i] ~ normal(mu_b, 1.0);       // 行为潜方差固定1：尺度不可分，见文档
  }
}
