# =====================================================================
# 后验参数汇总（取代各 *_posterior.R / parameter_save.R 的汇总部分）
# =====================================================================

core_params <- function() {
  c("alpha_e", "alpha_b", "beta_e", "beta_b", "theta1", "theta2",
    "tau_w_e", "tau_w_b", "sigma_obs_e", "sigma_proc_e", "phi_b_beta",
    "tau_v_e", "tau_v_b")   # tau_v_* 仅嵌套模型（台风）存在
}

summarize_draws <- function(draws, hazard = NULL, variant = NULL) {
  params <- intersect(core_params(), names(draws))
  extra <- grep("^(gamma_e_cov|gamma_b_cov|gamma_mod|delta_mod|jump_e|jump_b|tau_slope)",
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

write_summary <- function(summ, file) {
  utils::write.csv(summ, file, row.names = FALSE)
  invisible(summ)
}
