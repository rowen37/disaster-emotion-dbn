# =====================================================================
# Stage 3 - run script: generate the paper tables and figures
# (Table 1 / Table S6 / Fig 1 / Fig 2 / Fig 3 / Fig 4).
# =====================================================================
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1) stop("请使用 Rscript runs/<脚本名>.R 运行")
script_file <- normalizePath(sub("^--file=", "", script_arg))
source(file.path(dirname(script_file), "..", "R", "00_init.R"))
fig_dir <- paths$fig_dir
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

hz_order <- c("earthquake", "typhoon", "landslide", "debris_flow")
hz_cn <- c(earthquake = "Earthquake", typhoon = "Typhoon",
           landslide = "Landslide", debris_flow = "Debris flow")
draws_file <- function(hz) file.path(paths$out_dir, paste0(hz, "_baseline_w1_draws.rds"))
dat_file <- function(hz) file.path(paths$legacy_rds, paste0(hz, "_st_data.rds"))

all_draws <- setNames(lapply(hz_order, function(h) readRDS(draws_file(h))), hz_order)
all_dat <- setNames(lapply(hz_order, function(h) readRDS(dat_file(h))), hz_order)

# ---------------- Table 1: key parameters (posterior mean [95% CrI]) ----------------
tbl1 <- function(d) {
  p <- c(theta1 = "theta1", theta2 = "theta2", beta_e = "beta_e",
         beta_b = "beta_b", phi = "phi_b_beta",
         sigma_obs = "sigma_obs_e", sigma_proc = "sigma_proc_e",
         alpha_e = "alpha_e", alpha_b = "alpha_b")
  sapply(p, function(n) if (n %in% names(d))
    sprintf("%.3f [%.3f, %.3f]", mean(d[[n]]),
            quantile(d[[n]], .025), quantile(d[[n]], .975)) else "—")
}
t1 <- sapply(hz_order, function(h) tbl1(all_draws[[h]]))
write.csv(cbind(parameter = rownames(t1), as.data.frame(t1)),
          file.path(paths$out_dir, "Table1_final.csv"))

# ---------------- Table S6: moderator effects ----------------
rows <- list()
for (h in hz_order) for (v in c("pop", "gdp")) {
  f <- file.path(paths$out_dir, sprintf("%s_mod_%s_summary.csv", h, v))
  s <- read.csv(f)
  for (p in c("gamma_mod", "delta_mod", "theta1")) {
    r <- s[s$parameter == p, ]
    rows[[length(rows) + 1]] <- data.frame(
      hazard = h, moderator = v, parameter = p,
      estimate = sprintf("%.4f [%.4f, %.4f]", r$mean, r$Q2.5, r$Q97.5),
      significant = r$sig)
  }
}
write.csv(do.call(rbind, rows), file.path(paths$out_dir, "TableS6_final.csv"))

# ---------------- Fig 1: theta1 posteriors + forest plot ----------------
cols <- c(earthquake = "#C0392B", typhoon = "#2471A3",
          landslide = "#1E8449", debris_flow = "#B7950B")

grDevices::pdf(file.path(fig_dir, "Fig1_theta1_posteriors.pdf"),
               width = 8, height = 5)
par(mfrow = c(1, 2), mar = c(4, 4, 2.5, 1), las = 1)
# (a) posterior densities
dens <- lapply(hz_order, function(h) density(all_draws[[h]]$theta1))
xr <- range(sapply(dens, function(z) range(z$x)))
yr <- range(sapply(dens, function(z) range(z$y)))
plot(NA, xlim = xr, ylim = yr * 1.1, xlab = expression(theta[1]),
     ylab = "Posterior density", main = "")
for (i in seq_along(hz_order)) {
  h <- hz_order[i]
  polygon(dens[[i]], col = adjustcolor(cols[h], 0.35), border = cols[h], lwd = 2)
}
legend("topright", legend = hz_cn[hz_order], fill = adjustcolor(cols[hz_order], 0.35),
       border = cols[hz_order], bty = "n", cex = 0.9)
title(main = "(a) Contemporaneous coupling", cex.main = 1, line = 0.5)
# (b) forest plot
plist <- c("theta1", "theta2", "beta_e", "beta_b")
lab <- c(expression(theta[1]), expression(theta[2]),
         expression(beta[e]), expression(beta[b]))
yp <- seq_along(plist)
xr2 <- range(sapply(hz_order, function(h)
  sapply(plist, function(p) quantile(all_draws[[h]][[p]], c(.025, .975)))))
plot(NA, xlim = xr2, ylim = c(0.5, length(plist) + 0.5),
  yaxt = "n", xlab = "Posterior mean [95% CrI]", ylab = "")
axis(2, at = yp, labels = lab, las = 1)
abline(v = 0, lty = 2, col = "gray60")
off <- c(-0.15, -0.05, 0.05, 0.15)
for (i in seq_along(hz_order)) {
  h <- hz_order[i]; y <- yp + off[i]
  for (k in seq_along(plist)) {
    q <- quantile(all_draws[[h]][[plist[k]]], c(.025, .975))
    m <- mean(all_draws[[h]][[plist[k]]])
    segments(q[1], y[k], q[2], y[k], col = cols[h], lwd = 2.2)
    points(m, y[k], pch = 16, col = cols[h], cex = 0.9)
  }
}
title(main = "(b) Key parameters by hazard", cex.main = 1, line = 0.5)
dev.off()

# ---------------- Fig 2 / Fig 3: temporal and spatial curves (four hazards) ----------------
for (h in hz_order) {
  fig_temporal(all_draws[[h]], all_dat[[h]], hazard_configs[[h]], h)
  fig_spatial(all_draws[[h]], all_dat[[h]], hazard_configs[[h]], h)
}

# ---------------- Fig 4: per-ring temporal trajectories (posterior structural simulation) ----------------
ring_traj <- function(d, cfg, show_rings) {
  T_spline <- splines::ns(seq_len(N_TIME_BINS), df = cfg$df_time)
  B_spline <- splines::ns(ring_centers(cfg), df = cfg$df_dist)
  dn <- (ring_centers(cfg) - mean(ring_centers(cfg))) /
    (max(ring_centers(cfg)) - min(ring_centers(cfg))) * 2
  ns <- 400
  idx <- sample(seq_len(nrow(d)), ns)
  lapply(show_rings, function(k) {
    sim_E <- sim_B <- matrix(NA_real_, ns, N_TIME_BINS)
    for (s in seq_len(ns)) {
      i <- idx[s]
      gv <- function(pat) as.numeric(as.matrix(d[i, grep(pat, names(d)), drop = FALSE]))
      tc_e <- (gv("^psi_e") %*% t(T_spline)); tc_b <- (gv("^psi_b") %*% t(T_spline))
      tc_e <- as.numeric(tc_e - mean(tc_e)); tc_b <- as.numeric(tc_b - mean(tc_b))
      f_e <- as.numeric(B_spline[k, ] %*% gv("^lambda_e"))
      f_b <- as.numeric(B_spline[k, ] %*% gv("^lambda_b"))
      je <- if (cfg$has_jump == 1) ifelse(seq_len(N_TIME_BINS) >= JUMP_START, d$jump_e[i], 0) else rep(0, N_TIME_BINS)
      jb <- if (cfg$has_jump == 1) ifelse(seq_len(N_TIME_BINS) >= JUMP_START, d$jump_b[i], 0) else rep(0, N_TIME_BINS)
      e <- b <- numeric(N_TIME_BINS)
      e[1] <- d$alpha_e[i] + tc_e[1] + f_e + je[1] + rnorm(1, 0, d$sigma_proc_e[i])
      b[1] <- d$alpha_b[i] + tc_b[1] + f_b + jb[1] + d$theta1[i] * e[1] + rnorm(1)
      for (t in 2:N_TIME_BINS) {
        e[t] <- d$alpha_e[i] + tc_e[t] + f_e + je[t] + d$beta_e[i] * e[t-1] +
          rnorm(1, 0, d$sigma_proc_e[i])
        b[t] <- d$alpha_b[i] + tc_b[t] + f_b + jb[t] + d$beta_b[i] * b[t-1] +
          d$theta1[i] * e[t] + d$theta2[i] * e[t-1] + rnorm(1)
      }
      sim_E[s, ] <- plogis(e); sim_B[s, ] <- plogis(b)
    }
    list(E = sim_E, B = sim_B)
  })
}

grDevices::pdf(file.path(fig_dir, "Fig4_ring_trajectories.pdf"),
               width = 10, height = 9)
par(mfrow = c(4, 2), mar = c(4, 4, 2, 1), las = 1,
    cex.axis = 0.8, mgp = c(2, 0.6, 0))
for (h in hz_order) {
  cfg <- hazard_configs[[h]]
  K <- cfg$K
  show <- if (K == 10) c(1, 3, 5, 7, 10) else c(1, 5, 10, 15, 20)
  tr <- ring_traj(all_draws[[h]], cfg, show)
  th <- time_bin_centers()
  rl <- paste0((show - 1) * cfg$ring_km, "-", show * cfg$ring_km, " km")
  for (ch in c("E", "B")) {
    plot(NA, xlim = c(TIME_MIN, TIME_MAX), ylim = c(0, 1),
         xlab = "Hours relative to event", ylab = if (ch == "E")
           "Negative emotion probability" else "Behavior probability",
         main = if (ch == "E") hz_cn[h] else "")
    abline(v = 0, lty = 2, col = "gray60")
    ring_cols <- grDevices::colorRampPalette(c(cols[h], "gray65"))(length(tr))
    for (r in seq_along(tr)) {
      m <- colMeans(tr[[r]][[ch]])
      q <- apply(tr[[r]][[ch]], 2, quantile, c(.025, .975))
      polygon(c(th, rev(th)), c(q[1, ], rev(q[2, ])),
              col = adjustcolor(ring_cols[r], 0.10), border = NA)
      lines(th, m, col = ring_cols[r], lwd = 2)
    }
    if (h == hz_order[1] && ch == "E")
      legend("topright", rl, col = ring_cols, lwd = 2, bty = "n", cex = 0.7)
  }
}
dev.off()

cat("论文图表已生成:\n")
print(list.files(fig_dir, pattern = "Fig|earthquake_|typhoon_|landslide_|debris_flow_"))
cat("表: Table1_final.csv, TableS6_final.csv, S17_replacement_*.csv\n")
