# =====================================================================
# Stage 3 - DBN pipeline: model-observation consistency evaluation.
# Notes on what changed vs. the legacy accuracy table:
#   1) the legacy table reported RMSE = MAE for typhoon (only possible if
#      all |residuals| are equal - a transcription error)
#   2) the comparison target is explicit: conditional marginal trajectory
#      (random effects at zero) vs. observed cell means. This is a fit
#      consistency metric, not predictive accuracy (there is no
#      independent test set)
#   3) asserts RMSE >= MAE (numerical check of sqrt(mean(r^2)) >= mean(|r|))
# =====================================================================

# One hazard: model curve vs. observed means over time (T bins) and space (K rings)
eval_hazard <- function(draws, dat, cfg) {
  # ---- temporal dimension ----
  tt <- temporal_trajectory(draws, cfg)
  obs_t <- obs_temporal(dat)
  cmp_t <- dplyr::left_join(
    data.frame(time_hour = tt$time_hour,
               E_fit = colMeans(tt$prob_e),
               B_fit = colMeans(tt$prob_b)),
    obs_t, by = "time_hour")

  metrics <- function(o, f, dim) {
    r <- o - f
    data.frame(
      dimension = dim,
      RMSE = sqrt(mean(r^2, na.rm = TRUE)),
      MAE = mean(abs(r), na.rm = TRUE),
      n = sum(!is.na(r)))
  }
  out <- rbind(
    metrics(cmp_t$mean_E, cmp_t$E_fit, "temporal_emotion"),
    metrics(cmp_t$mean_B, cmp_t$B_fit, "temporal_behavior"))

  # ---- spatial dimension ----
  sd_curve <- spatial_decay(draws, cfg)
  obs_e <- obs_spatial(dat, "E_obs"); obs_b <- obs_spatial(dat, "B_obs")
  e_fit <- plogis(colMeans(sd_curve$logit_e))   # probability scale
  b_fit <- plogis(colMeans(sd_curve$logit_b))
  out <- rbind(out,
    metrics(obs_e, e_fit, "spatial_emotion"),
    metrics(obs_b, b_fit, "spatial_behavior"))

  # ---- invariant check: RMSE >= MAE must hold ----
  stopifnot(all(out$RMSE + 1e-12 >= out$MAE))
  out
}

# fits: named list keyed by hazard (keys of hazard_configs); each item is
#        list(draws = <draws df or rds path>, dat = <st_data>)
eval_all <- function(fits) {
  do.call(rbind, lapply(names(fits), function(hz) {
    f <- fits[[hz]]
    d <- if (is.character(f$draws)) readRDS(f$draws) else f$draws
    res <- eval_hazard(d, f$dat, hazard_configs[[hz]])
    res$hazard <- hz
    res
  }))
}
