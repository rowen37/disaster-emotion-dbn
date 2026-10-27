# =====================================================================
# Stage 3 - DBN pipeline: figure outputs (temporal / spatial marginal
# effects). All figures are written to paths$fig_dir. The jump curve is
# cut at the configured JUMP_START (legacy code hard-coded t >= 8, one
# bin late).
# =====================================================================

# logit -> probability with second-order Delta bias correction
delta_correct <- function(logit_mat) {
  sigma2 <- apply(logit_mat, 2, stats::var)
  p_raw <- plogis(logit_mat)
  second_deriv <- p_raw * (1 - p_raw) * (1 - 2 * p_raw)
  sigma2_mat <- matrix(sigma2, nrow = nrow(logit_mat), ncol = ncol(logit_mat),
                       byrow = TRUE)
  p_raw + 0.5 * sigma2_mat * second_deriv
}

get_draws_mat <- function(draws, prefix) {
  as.matrix(draws[, grep(paste0("^", prefix, "\\["), names(draws)), drop = FALSE])
}

# ---- spatial decay: model marginal effect (logit + Delta-corrected probability) ----
spatial_decay <- function(draws, cfg) {
  dc <- ring_centers(cfg)
  B_spline <- splines::ns(dc, df = cfg$df_dist)
  le <- get_draws_mat(draws, "lambda_e") %*% t(B_spline)
  lb <- get_draws_mat(draws, "lambda_b") %*% t(B_spline)
  logit_e <- matrix(draws$alpha_e, nrow = nrow(le), ncol = length(dc)) + le
  logit_b <- matrix(draws$alpha_b, nrow = nrow(lb), ncol = length(dc)) + lb
  list(
    dist_km = dc,
    logit_e = logit_e, logit_b = logit_b,
    prob_e = delta_correct(logit_e), prob_b = delta_correct(logit_b)
  )
}

# ---- temporal trajectory: global spline + jump (cut at JUMP_START), probability scale ----
temporal_trajectory <- function(draws, cfg) {
  T_spline <- splines::ns(seq_len(N_TIME_BINS), df = cfg$df_time)
  # center each draw's spline across time (matches time_e_cent inside the Stan
  # model; centering across draws would wash out the temporal structure)
  cent <- function(m) sweep(m, 1, rowMeans(m))
  tc_e <- cent(get_draws_mat(draws, "psi_e") %*% t(T_spline))
  tc_b <- cent(get_draws_mat(draws, "psi_b") %*% t(T_spline))

  shock_e <- matrix(0, nrow = nrow(tc_e), ncol = N_TIME_BINS)
  shock_b <- shock_e
  if (cfg$has_jump == 1) {
    for (t in JUMP_START:N_TIME_BINS) {   # fix: legacy code hard-coded t >= 8
      shock_e[, t] <- draws$jump_e
      shock_b[, t] <- draws$jump_b
    }
  }
  logit_e <- draws$alpha_e + tc_e + shock_e
  logit_b <- draws$alpha_b + tc_b + shock_b
  list(
    time_hour = time_bin_centers(),
    logit_e = logit_e, logit_b = logit_b,
    prob_e = delta_correct(logit_e), prob_b = delta_correct(logit_b)
  )
}

# ---- observed cell means (temporal / spatial) ----
obs_temporal <- function(dat) {
  dat %>%
    dplyr::group_by(time_bin) %>%
    dplyr::summarise(
      mean_E = mean(E_obs, na.rm = TRUE),
      mean_B = mean(B_obs, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::mutate(time_hour = time_bin_centers())
}

obs_spatial <- function(dat, var = "E_obs") {
  dat %>%
    dplyr::filter(!is.na(.data[[var]])) %>%
    dplyr::group_by(dist_bin) %>%
    dplyr::summarise(mean_v = mean(.data[[var]], na.rm = TRUE), .groups = "drop") %>%
    dplyr::pull(mean_v)
}

# ---- figure: temporal trajectory (probability scale + observed points) ----
fig_temporal <- function(draws, dat, cfg, hazard, fig_dir = paths$fig_dir) {
  dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
  tt <- temporal_trajectory(draws, cfg)
  ob <- obs_temporal(dat)
  e_ci <- apply(tt$prob_e, 2, stats::quantile, c(0.025, 0.975))
  b_ci <- apply(tt$prob_b, 2, stats::quantile, c(0.025, 0.975))

  grDevices::pdf(file.path(fig_dir, paste0(hazard, "_temporal.pdf")),
                 width = 8, height = 5, pointsize = 14)
  par(mar = c(5, 5, 2.5, 2), las = 1)
  plot(tt$time_hour, colMeans(tt$prob_e), type = "n",
       xlim = c(TIME_MIN, TIME_MAX), ylim = c(0, max(0.6, max(e_ci))),
       xlab = "Time relative to event (h)", ylab = "Negative emotion probability", axes = FALSE)
  axis(1); axis(2); box()
  polygon(c(tt$time_hour, rev(tt$time_hour)), c(e_ci[1, ], rev(e_ci[2, ])),
          col = grDevices::rgb(1, 0, 0, 0.12), border = NA)
  lines(tt$time_hour, colMeans(tt$prob_e), col = "red", lwd = 2)
  points(ob$time_hour, ob$mean_E, pch = 16, cex = 0.9)
  graphics::abline(v = 0, col = "gray", lty = 2)
  legend("topright", c("Model (conditional marginal)", "Observed cell mean"),
         col = c("red", "black"), lty = c(1, NA), pch = c(NA, 16), lwd = 2, bty = "n")
  dev.off()

  grDevices::pdf(file.path(fig_dir, paste0(hazard, "_behavior_temporal.pdf")),
                 width = 8, height = 5, pointsize = 14)
  par(mar = c(5, 5, 2.5, 2), las = 1)
  plot(tt$time_hour, colMeans(tt$prob_b), type = "n",
       xlim = c(TIME_MIN, TIME_MAX), ylim = c(0, max(0.2, max(b_ci))),
       xlab = "Time relative to event (h)", ylab = "Behavior probability", axes = FALSE)
  axis(1); axis(2); box()
  polygon(c(tt$time_hour, rev(tt$time_hour)), c(b_ci[1, ], rev(b_ci[2, ])),
          col = grDevices::rgb(0, 0, 1, 0.12), border = NA)
  lines(tt$time_hour, colMeans(tt$prob_b), col = "blue", lwd = 2)
  points(ob$time_hour, ob$mean_B, pch = 16, cex = 0.9)
  graphics::abline(v = 0, col = "gray", lty = 2)
  legend("topright", c("Model (conditional marginal)", "Observed cell mean"),
         col = c("blue", "black"), lty = c(1, NA), pch = c(NA, 16), lwd = 2, bty = "n")
  dev.off()
  invisible(tt)
}

# ---- figure: spatial decay (logit scale + observed points) ----
fig_spatial <- function(draws, dat, cfg, hazard, fig_dir = paths$fig_dir) {
  dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
  sd <- spatial_decay(draws, cfg)
  logit_ci_e <- apply(sd$logit_e, 2, stats::quantile, c(0.025, 0.975))
  logit_ci_b <- apply(sd$logit_b, 2, stats::quantile, c(0.025, 0.975))
  obs_e <- qlogis(pmin(pmax(obs_spatial(dat, "E_obs"), 0.001), 0.999))
  obs_b <- qlogis(pmin(pmax(obs_spatial(dat, "B_obs"), 0.001), 0.999))

  grDevices::pdf(file.path(fig_dir, paste0(hazard, "_spatial.pdf")),
                 width = 8, height = 5, pointsize = 14)
  par(mar = c(5, 5, 2.5, 2), las = 1)
  plot(sd$dist_km, colMeans(sd$logit_e), type = "n",
       xlim = c(0, cfg$max_dist_km),
       ylim = range(c(logit_ci_e, logit_ci_b, obs_e, obs_b)),
       xlab = "Distance from event center (km)", ylab = "Baseline (logit)", axes = FALSE)
  axis(1); axis(2); box()
  polygon(c(sd$dist_km, rev(sd$dist_km)), c(logit_ci_e[1, ], rev(logit_ci_e[2, ])),
          col = grDevices::rgb(1, 0, 0, 0.10), border = NA)
  lines(sd$dist_km, colMeans(sd$logit_e), col = "red", lwd = 2)
  polygon(c(sd$dist_km, rev(sd$dist_km)), c(logit_ci_b[1, ], rev(logit_ci_b[2, ])),
          col = grDevices::rgb(0, 0, 1, 0.10), border = NA)
  lines(sd$dist_km, colMeans(sd$logit_b), col = "blue", lwd = 2)
  points(sd$dist_km, obs_e, pch = 16, cex = 0.9)
  points(sd$dist_km, obs_b, pch = 17, cex = 0.9)
  legend("topright", c("Emotion (model)", "Behavior (model)", "Emotion (obs)", "Behavior (obs)"),
         col = c("red", "blue", "black", "black"),
         lty = c(1, 1, NA, NA), pch = c(NA, NA, 16, 17), lwd = 2, bty = "n")
  dev.off()
  invisible(sd)
}
