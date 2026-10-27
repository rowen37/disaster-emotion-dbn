# =====================================================================
# Stage 3 - DBN pipeline: RDS -> stan_data packing + unified fit entry.
# Unified single entry point for all hazards and variants:
#   - init lists are generated from the parameter blocks that actually
#     exist (an init referencing non-existent phi_e/phi_b silently never
#     takes effect)
#   - lag_idx is always rebuilt on the fly
#   - event-covariate standardization is unified (standardize only with
#     >= 2 valid values; otherwise disable the covariate)
# =====================================================================

epsilon <- 0.001

# ---- event-level covariates: first valid value per event -> missing flags -> standardize ----
prep_event_covs <- function(dat, cov_names) {
  G <- length(cov_names)
  ev <- dat %>%
    dplyr::distinct(event_id, .keep_all = TRUE) %>%
    dplyr::arrange(event_id) %>%
    dplyr::select(event_id, dplyr::all_of(cov_names))

  val <- as.matrix(ev[cov_names])
  storage.mode(val) <- "numeric"
  miss <- is.na(val) | !is.finite(val)
  val[miss] <- 0

  # per covariate: standardize only with >= 2 valid values and SD > 0, else disable
  for (g in seq_len(G)) {
    ok <- !miss[, g]
    if (sum(ok) >= 2 && stats::sd(val[ok, g]) > 1e-8) {
      val[ok, g] <- (val[ok, g] - mean(val[ok, g])) / stats::sd(val[ok, g])
    } else {
      miss[, g] <- TRUE
      val[, g] <- 0
    }
  }
  val[!is.finite(val)] <- 0

  list(
    event_id = as.character(ev$event_id),
    val = val,
    miss = matrix(as.integer(miss), nrow = nrow(val)),
    G = G
  )
}

validate_aggregated_data <- function(dat, cfg) {
  required <- c(
    "event_id", "dist_bin", "time_bin", "E_obs", "B_obs",
    "N", CONTEXT_COVARIATES
  )
  missing <- setdiff(required, names(dat))
  if (length(missing) > 0) {
    stop(
      "Stage 3 聚合数据缺少字段: ", paste(missing, collapse = ", "),
      "。请用当前 preprocess_hazard()/dedup_typhoon() 重新生成，不能混用旧聚合 RDS。"
    )
  }
  if (nrow(dat) == 0) stop("Stage 3 聚合数据为空")
  if (anyDuplicated(dat[c("event_id", "dist_bin", "time_bin")])) {
    stop("Stage 3 聚合数据存在重复的 event_id × dist_bin × time_bin")
  }
  rows_per_event <- table(dat$event_id)
  if (any(rows_per_event != cfg$K * N_TIME_BINS)) {
    stop("每个事件必须恰好包含 K × T 个完整时空格")
  }
  if (any(!is.finite(dat$dist_bin)) || any(dat$dist_bin < 1 | dat$dist_bin > cfg$K) ||
      any(!is.finite(dat$time_bin)) ||
      any(dat$time_bin < 1 | dat$time_bin > N_TIME_BINS)) {
    stop("Stage 3 聚合数据的时空格编号超出配置范围")
  }

  if (any(!is.finite(dat$N)) || any(dat$N < 0 | dat$N != floor(dat$N))) {
    stop("Stage 3 聚合数据的总帖子数字段 N 非法")
  }

  e_present <- !is.na(dat$E_obs)
  b_present <- !is.na(dat$B_obs)
  if (any(dat$N[e_present | b_present] < 1)) {
    stop("存在观测非缺失但 N < 1 的时空格")
  }
  if (any(!is.finite(dat$E_obs[e_present])) ||
      any(dat$E_obs[e_present] < 0 | dat$E_obs[e_present] > 1)) {
    stop("E_obs 含非有限值或超出 [0,1]")
  }
  if (any(!is.finite(dat$B_obs[b_present])) ||
      any(dat$B_obs[b_present] < 0 | dat$B_obs[b_present] > 1)) {
    stop("B_obs 含非有限值或超出 [0,1]")
  }
  invisible(TRUE)
}

# Previous-time adjacent-ring indices.  The own-ring previous-time state stays
# in lag_idx (beta_e/beta_b); these indices contain only k-1 and k+1 so the
# spatial propagation coefficients cannot duplicate the temporal AR terms.
attach_spatial_lag_idx <- function(dat, cfg) {
  key <- paste(dat$event_id, dat$dist_bin, dat$time_bin, sep = "|")
  lookup <- stats::setNames(seq_len(nrow(dat)), key)
  idx <- matrix(0L, nrow = nrow(dat), ncol = 2)
  n_neighbor <- integer(nrow(dat))
  for (i in seq_len(nrow(dat))) {
    if (dat$time_bin[i] <= 1) next
    rings <- c(dat$dist_bin[i] - 1L, dat$dist_bin[i] + 1L)
    rings <- rings[rings >= 1L & rings <= cfg$K]
    keys <- paste(dat$event_id[i], rings, dat$time_bin[i] - 1L, sep = "|")
    found <- unname(lookup[keys])
    found <- as.integer(found[!is.na(found)])
    n_neighbor[i] <- length(found)
    if (length(found) > 0) idx[i, seq_along(found)] <- found
  }
  list(idx = idx, n = n_neighbor)
}

# ---- main packing function ----
build_stan_data <- function(dat, cfg, use_weights = USE_POST_COUNT_WEIGHTS,
                            variant = c("baseline", "moderator"), mod_ring = NULL) {
  variant <- match.arg(variant)
  if (length(use_weights) != 1 || !(use_weights %in% c(0, 1))) {
    stop("use_weights 必须是 0 或 1")
  }
  if (variant == "moderator" && is.null(mod_ring))
    stop("moderator 模型必须提供 mod_ring 矩阵 [J x K]")
  if (variant == "moderator" && !is.null(cfg$has_slope) && cfg$has_slope == 1) {
    stop("moderator 模型不包含随机空间斜率；请勿静默混用 slope 配置")
  }
  if (variant == "moderator" && !is.null(cfg$has_coupling) &&
      cfg$has_coupling == 0) {
    stop("moderator 模型不支持 no-coupling 配置")
  }

  validate_aggregated_data(dat, cfg)
  dat <- attach_lag_idx(dat)   # rebuilt here; ignore any version stored in the RDS
  spatial_lag <- attach_spatial_lag_idx(dat, cfg)

  dat <- dat %>%
    dplyr::mutate(
      E_obs_tr = dplyr::if_else(
        !is.na(E_obs),
        qlogis(pmin(pmax(E_obs, epsilon), 1 - epsilon)),
        0
      ),
      B_obs_cl = dplyr::if_else(!is.na(B_obs),
                                pmin(pmax(B_obs, epsilon), 1 - epsilon), 0.5),
      E_miss = as.integer(is.na(E_obs)),
      B_miss = as.integer(is.na(B_obs)),
      n_i = pmax(as.numeric(N), 1),
      post_present = as.integer(N > 0),
      post_count = as.integer(N)
    )

  dat$event_idx <- as.integer(as.factor(dat$event_id))
  J <- dplyr::n_distinct(dat$event_idx)
  K <- cfg$K
  T <- N_TIME_BINS
  if (variant == "moderator") {
    if (!is.matrix(mod_ring) || !all(dim(mod_ring) == c(J, K)) ||
        any(!is.finite(mod_ring))) {
      stop("mod_ring 必须是无缺失的 J x K 数值矩阵（当前应为 ", J, " x ", K, "）")
    }
  }

  dist_centers <- ring_centers(cfg)
  B_spline <- splines::ns(dist_centers, df = cfg$df_dist)
  T_spline <- splines::ns(1:T, df = cfg$df_time)

  evc <- prep_event_covs(dat, cfg$ev_covs)
  # events in dat follow as.factor lexicographic order; evc is sorted by event_id: align them
  stopifnot(setequal(evc$event_id, unique(dat$event_id)))
  idx <- match(levels(as.factor(dat$event_id)), evc$event_id)
  evc$event_id <- evc$event_id[idx]
  evc$val <- evc$val[idx, , drop = FALSE]
  evc$miss <- evc$miss[idx, , drop = FALSE]

  # nested grouping (typhoon: cfg$group_col points to the typhoon id column;
  # other hazards lack the column -> no nesting)
  group_configured <- !is.null(cfg$group_col)
  if (group_configured && !(cfg$group_col %in% names(dat))) {
    stop("该灾种配置了上级分组字段但聚合数据缺少: ", cfg$group_col)
  }
  use_group <- group_configured &&
    (length(unique(dat[[cfg$group_col]])) < dplyr::n_distinct(dat$event_id))
  if (group_configured && !use_group) {
    stop("上级分组未形成嵌套结构；请检查 ", cfg$group_col, " 与 event_id 的连接")
  }
  if (use_group) {
    event_levels <- levels(as.factor(dat$event_id))
    ev_groups <- dat %>%
      dplyr::transmute(
        event_id = as.character(event_id),
        group_id = as.character(.data[[cfg$group_col]])
      ) %>%
      dplyr::distinct()
    if (anyDuplicated(ev_groups$event_id) || any(is.na(ev_groups$group_id))) {
      stop("每个事件必须且只能连接到一个非缺失的上级分组")
    }
    ev_groups <- ev_groups[match(event_levels, ev_groups$event_id), , drop = FALSE]
    group_levels <- sort(unique(ev_groups$group_id))
    group_of_event <- match(ev_groups$group_id, group_levels)
    G <- max(group_of_event)
  } else {
    group_of_event <- rep(1L, J)
    G <- 1L
  }

  # event-level random spatial slope (sensitivity-analysis switch; dist_norm = centered
  # and normalized ring-center distance)
  has_slope <- if (!is.null(cfg$has_slope)) as.integer(cfg$has_slope) else 0L
  dc <- ring_centers(cfg)
  dist_norm <- (dc - mean(dc)) / (max(dc) - min(dc)) * 2   # [-1, 1]
  # no-coupling control switch (ablation): with 0, theta1/theta2 do not exist in the fit
  has_coupling <- if (!is.null(cfg$has_coupling)) as.integer(cfg$has_coupling) else 1L

  context_cov <- as.matrix(dat[CONTEXT_COVARIATES])
  storage.mode(context_cov) <- "numeric"
  if (any(!is.finite(context_cov))) stop("context covariates must be finite")

  stan_data <- list(
    N = nrow(dat), J = J, K = K, T = T,
    event = dat$event_idx,
    dist_bin = dat$dist_bin,
    time_bin = dat$time_bin,
    E_miss = dat$E_miss,
    B_miss = dat$B_miss,
    E_obs = dat$E_obs_tr,
    B_obs = dat$B_obs_cl,
    n_posts = dat$n_i,
    post_present = dat$post_present,
    post_count = dat$post_count,
    use_weights = use_weights,
    M = ncol(B_spline), B_spline = B_spline,
    P = ncol(T_spline), T_spline = T_spline,
    lag_idx = dat$lag_idx,
    spatial_lag_idx = spatial_lag$idx,
    spatial_lag_n = spatial_lag$n,
    has_jump = cfg$has_jump,
    jump_start = JUMP_START,
    has_group = as.integer(use_group),
    G = G,
    group_of_event = group_of_event,
    Gc = evc$G,
    ev_cov = evc$val,
    ev_cov_miss = as.vector(evc$miss),  # column-major flattening, matching Stan index (g-1)*J+j
    Cc = ncol(context_cov),
    context_cov = context_cov,
    covariate_prior_sd = COVARIATE_PRIOR_SD,
    has_slope = has_slope,
    dist_norm = dist_norm,
    has_coupling = has_coupling
  )

  if (variant == "moderator") stan_data$mod_ring <- mod_ring
  list(stan_data = stan_data, dat = dat, cfg = cfg)
}

# ---- init: built per conditional parameter block (absent parameters stay out of init) ----
make_init <- function(stan_data, variant = c("baseline", "moderator")) {
  variant <- match.arg(variant)
  function() {
    init <- list(
      alpha_e = -1.0, alpha_b = -2.0,
      beta_e = 0.02, beta_b = 0.01,
      psi_e = rep(0.1, stan_data$P), psi_b = rep(0.1, stan_data$P),
      lambda_e = rep(0.1, stan_data$M), lambda_b = rep(0.1, stan_data$M),
      z_w_e = rep(0, stan_data$J), z_w_b = rep(0, stan_data$J),
      tau_w_e = 0.1, tau_w_b = 0.1,
      sigma_obs_e = 0.8, sigma_proc_e = 0.8,
      phi_b_beta = 1.0,
      gamma_e_cov = rep(0, stan_data$Gc), gamma_b_cov = rep(0, stan_data$Gc)
    )
    if (stan_data$has_jump == 1) init$jump_raw <- c(0.1, 0.1)
    if (stan_data$has_group == 1) {
      init$z_v_e_raw <- rep(0, stan_data$G)
      init$z_v_b_raw <- rep(0, stan_data$G)
      init$tau_v_e_arr <- 1
      init$tau_v_b_arr <- 1
    }
    if (variant == "baseline" && stan_data$has_slope == 1) {
      init$z_slope_e_raw <- rep(0, stan_data$J)
      init$z_slope_b_raw <- rep(0, stan_data$J)
      init$tau_slope_e_arr <- 0.1
      init$tau_slope_b_arr <- 0.1
    }
    if (variant == "baseline" && stan_data$has_coupling == 1) {
      init$theta_arr <- c(0.05, 0.0)
    }
    if (variant == "baseline") {
      occupancy <- mean(stan_data$post_present)
      positive <- stan_data$post_count[stan_data$post_present == 1]
      init$rho_e <- 0.02
      init$rho_b <- 0.02
      init$gamma_e_context <- rep(0, stan_data$Cc)
      init$gamma_b_context <- rep(0, stan_data$Cc)
      init$alpha_occ <- qlogis(pmin(pmax(occupancy, 0.01), 0.99))
      init$alpha_count <- log(max(mean(positive - 1), 0.1))
      init$psi_occ <- rep(0, stan_data$P)
      init$psi_count <- rep(0, stan_data$P)
      init$lambda_occ <- rep(0, stan_data$M)
      init$lambda_count <- rep(0, stan_data$M)
      init$gamma_occ_cov <- rep(0, stan_data$Gc)
      init$gamma_count_cov <- rep(0, stan_data$Gc)
      init$gamma_occ_context <- rep(0, stan_data$Cc)
      init$gamma_count_context <- rep(0, stan_data$Cc)
      init$z_w_occ <- rep(0, stan_data$J)
      init$z_w_count <- rep(0, stan_data$J)
      init$tau_w_occ <- 0.2
      init$tau_w_count <- 0.2
      init$omega_occ <- c(0, 0)
      init$omega_count <- c(0, 0)
      init$phi_count <- 2
    }
    if (variant == "moderator") {
      init$theta1 <- 0.05
      init$theta2 <- 0.0
      init$gamma_mod <- 0.0
      init$delta_mod <- 0.0
    }
    # latent states start near baseline levels to avoid violent ADVI transients
    init$e_true <- rep(-1.0, stan_data$N)
    init$b_true <- rep(-2.0, stan_data$N)
    init
  }
}

# ---- unified fit entry point ----
fit_hazard <- function(hazard, variant = c("baseline", "moderator"),
                       use_weights = USE_POST_COUNT_WEIGHTS, mod_ring = NULL,
                       dat = NULL, settings = fit_settings,
                       output_dir = NULL, quick = FALSE) {
  variant <- match.arg(variant)
  cfg <- hazard_configs[[hazard]]
  if (is.null(cfg)) stop("未知灾种: ", hazard)
  if (is.null(dat)) {
    input_file <- file.path(paths$stage3_data, paste0(hazard, "_st_data.rds"))
    if (!file.exists(input_file)) {
      stop("缺少当前 Stage 3 聚合数据: ", input_file, "；请先运行聚合")
    }
    dat <- readRDS(input_file)
  }

  packed <- build_stan_data(dat, cfg, use_weights = use_weights,
                            variant = variant, mod_ring = mod_ring)
  stan_path <- if (variant == "baseline") paths$stan_core else paths$stan_mod
  mod <- cmdstanr::cmdstan_model(stan_path)
  s <- settings
  if (quick) {
    s$iter <- min(s$iter, 1500)
    s$output_samples <- min(s$output_samples, 500)
  }

  fit <- mod$variational(
    data = packed$stan_data,
    seed = s$seed,
    algorithm = "meanfield",
    iter = s$iter,
    grad_samples = s$grad_samples,
    elbo_samples = s$elbo_samples,
    output_samples = s$output_samples,
    init = make_init(packed$stan_data, variant),
    eta = s$eta,
    tol_rel_obj = s$tol_rel_obj
  )

  draws <- fit$draws(format = "df")
  if (!is.null(output_dir)) {
    dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
    # archive names must distinguish model configs (a slope fit once overwrote baseline draws)
    cfg2 <- if (!is.null(packed$cfg$has_slope) && packed$cfg$has_slope == 1) "_slope" else ""
    tag <- paste0(hazard, "_", variant, cfg2, "_w", use_weights)
    saveRDS(draws, file.path(output_dir, paste0(tag, "_draws.rds")))
  }
  list(draws = draws, fit = fit, packed = packed, cfg = cfg)
}
