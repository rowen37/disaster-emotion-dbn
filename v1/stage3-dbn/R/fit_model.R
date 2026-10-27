# =====================================================================
# Stage 3 - DBN pipeline: RDS -> stan_data packing + unified fit entry.
# Replaces the legacy per-hazard fit scripts. Fixes:
#   - init lists are generated from the parameter blocks that actually
#     exist (legacy init referenced non-existent phi_e/phi_b and never
#     took effect)
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
  miss <- is.na(val)
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

# ---- main packing function ----
build_stan_data <- function(dat, cfg, use_weights = 1,
                            variant = c("baseline", "moderator"), mod_ring = NULL) {
  variant <- match.arg(variant)
  if (variant == "moderator" && is.null(mod_ring))
    stop("moderator 模型必须提供 mod_ring 矩阵 [J x K]")

  dat <- attach_lag_idx(dat)   # rebuilt here; ignore any version stored in the RDS

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
      n_i = dplyr::if_else(is.na(N) | N < 1, 1, as.numeric(N))
    )

  dat$event_idx <- as.integer(as.factor(dat$event_id))
  J <- dplyr::n_distinct(dat$event_idx)
  K <- cfg$K
  T <- N_TIME_BINS

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
  use_group <- !is.null(cfg$group_col) &&
    (cfg$group_col %in% names(dat)) &&
    (length(unique(dat[[cfg$group_col]])) < dplyr::n_distinct(dat$event_id))
  if (use_group) {
    ev_groups <- dat %>%
      dplyr::mutate(ev_lvl = as.factor(event_id)) %>%
      dplyr::group_by(ev_lvl) %>%
      dplyr::summarise(grp = as.factor(dplyr::first(.data[[cfg$group_col]])),
                       .groups = "drop")
    group_of_event <- as.integer(ev_groups$grp)   # aligned with the event factor order
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
    use_weights = use_weights,
    M = ncol(B_spline), B_spline = B_spline,
    P = ncol(T_spline), T_spline = T_spline,
    lag_idx = dat$lag_idx,
    has_jump = cfg$has_jump,
    jump_start = JUMP_START,
    has_group = as.integer(use_group),
    G = G,
    group_of_event = group_of_event,
    Gc = evc$G,
    ev_cov = evc$val,
    ev_cov_miss = as.vector(evc$miss),  # column-major flattening, matching Stan index (g-1)*J+j
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
    if (stan_data$has_slope == 1) {
      init$z_slope_e_raw <- rep(0, stan_data$J)
      init$z_slope_b_raw <- rep(0, stan_data$J)
      init$tau_slope_e_arr <- 0.1
      init$tau_slope_b_arr <- 0.1
    }
    if (stan_data$has_coupling == 1) {
      init$theta_arr <- c(0.05, 0.0)
    }
    if (variant == "moderator") {
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
                       use_weights = 1, mod_ring = NULL,
                       dat = NULL, settings = fit_settings,
                       output_dir = NULL, quick = FALSE) {
  variant <- match.arg(variant)
  cfg <- hazard_configs[[hazard]]
  if (is.null(cfg)) stop("未知灾种: ", hazard)
  if (is.null(dat)) {
    dat <- readRDS(file.path(paths$legacy_rds, paste0(hazard, "_st_data.rds")))
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
