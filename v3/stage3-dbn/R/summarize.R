# =====================================================================
# Stage 3 - DBN pipeline: posterior parameter summaries.
# =====================================================================

core_params <- function() {
  c("alpha_e", "alpha_b", "beta_e", "beta_b", "theta1", "theta2",
    "rho_e", "rho_b",
    "tau_w_e", "tau_w_b", "sigma_obs_e", "sigma_proc_e", "phi_b_beta",
    "tau_v_e", "tau_v_b",
    "alpha_occ", "alpha_count", "tau_w_occ", "tau_w_count",
    "omega_occ[1]", "omega_occ[2]", "omega_count[1]", "omega_count[2]",
    "phi_count")   # tau_v_* exist only in the nested (typhoon) model
}

summarize_draws <- function(draws, hazard = NULL, variant = NULL) {
  params <- intersect(core_params(), names(draws))
  extra <- grep("^(gamma_e_cov|gamma_b_cov|gamma_e_context|gamma_b_context|gamma_occ_cov|gamma_count_cov|gamma_occ_context|gamma_count_context|gamma_mod|delta_mod|jump_e|jump_b|tau_slope)",
                names(draws), value = TRUE)
  params <- c(params, extra)

  summ <- draws %>%
    dplyr::select(dplyr::all_of(params)) %>%
    tidyr::pivot_longer(dplyr::everything(), names_to = "parameter") %>%
    dplyr::group_by(parameter) %>%
    dplyr::summarise(
      mean = mean(value), sd = stats::sd(value),
      Q2.5 = stats::quantile(value, 0.025),
      Q97.5 = stats::quantile(value, 0.975),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      sig = Q2.5 > 0 | Q97.5 < 0,
      disaster = hazard, variant = variant
    )
  summ
}

optimized_parameter_map <- function(cfg) {
  event_labels <- cfg$ev_covs
  context_labels <- CONTEXT_COVARIATES
  mapping <- c(
    beta_e = "emotion residual persistence",
    beta_b = "harmful-text residual persistence",
    theta1 = "same-bin residual emotion-harmful-text association",
    theta2 = "12-hour lagged residual emotion-harmful-text association",
    rho_e = "previous-time adjacent-ring emotion propagation",
    rho_b = "previous-time adjacent-ring harmful-text propagation",
    `omega_occ[1]` = "emotion residual loading in grid-presence model",
    `omega_occ[2]` = "harmful-text residual loading in grid-presence model",
    `omega_count[1]` = "emotion residual loading in positive-count model",
    `omega_count[2]` = "harmful-text residual loading in positive-count model",
    phi_count = "positive-count negative-binomial dispersion"
  )
  for (channel in c("e", "b", "occ", "count")) {
    for (g in seq_along(event_labels)) {
      key <- sprintf("gamma_%s_cov[%d]", channel, g)
      mapping[key] <- paste(channel, "standardized event covariate", event_labels[g])
    }
    for (g in seq_along(context_labels)) {
      key <- sprintf("gamma_%s_context[%d]", channel, g)
      mapping[key] <- paste(channel, context_labels[g])
    }
  }
  mapping
}

summarize_optimized_draws <- function(draws, hazard, cfg) {
  mapping <- optimized_parameter_map(cfg)
  params <- intersect(names(mapping), names(draws))
  draws %>%
    dplyr::select(dplyr::all_of(params)) %>%
    tidyr::pivot_longer(dplyr::everything(), names_to = "parameter") %>%
    dplyr::group_by(parameter) %>%
    dplyr::summarise(
      mean = mean(value),
      sd = stats::sd(value),
      Q2.5 = stats::quantile(value, 0.025),
      Q50 = stats::quantile(value, 0.5),
      Q97.5 = stats::quantile(value, 0.975),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      hazard = hazard,
      result = unname(mapping[parameter]),
      interval_excludes_zero = Q2.5 > 0 | Q97.5 < 0,
      .before = 1
    )
}

write_summary <- function(summ, file) {
  utils::write.csv(summ, file, row.names = FALSE)
  invisible(summ)
}
