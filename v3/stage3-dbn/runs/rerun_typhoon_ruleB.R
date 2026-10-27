# =====================================================================
# Stage 3 - run script: rebuild the typhoon data under dedup rule B
# (dedup within typhoon only) and refit the typhoon baseline and both
# moderator models.
# =====================================================================
script_file <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]]))
source(file.path(dirname(dirname(script_file)), "R", "00_init.R"))
log_line <- function(...) cat(format(Sys.time(), "[%H:%M:%S]"), ..., "\n"); flush.console()

log_line("规则B 重建台风数据")
stB <- dedup_typhoon(typhoons = NULL)
dir.create(paths$stage3_data, showWarnings = FALSE, recursive = TRUE)
saveRDS(stB, file.path(paths$stage3_data, "typhoon_st_data.rds"), compress = "gzip")
log_line(sprintf("stB: %d 行 %d 路径点 %d 台风 帖子 %d",
    nrow(stB), length(unique(stB$event_id)), length(unique(stB$typhoon_id)),
    sum(stB$N, na.rm = TRUE)))

# rebuild mod_rings (the event set may change slightly)
for (v in c("pop", "gdp")) {
  rp <- if (v == "pop") paths$pop_raster else paths$gdp_raster
  m <- build_mod_ring("typhoon", rp, dat = stB)
  saveRDS(m, file.path(paths$stage3_data, sprintf("modring_%s_typhoon.rds", v)),
          compress = "gzip")
  log_line(v, "mod_ring:", nrow(m), "x", ncol(m))
}

log_line("台风基线（规则B）")
res <- fit_hazard("typhoon", variant = "baseline",
                  use_weights = USE_POST_COUNT_WEIGHTS, dat = stB,
                  output_dir = paths$out_dir)
summ <- summarize_draws(res$draws, hazard = "typhoon", variant = "baseline_full_w1")
write_summary(summ, file.path(paths$out_dir, "typhoon_baseline_summary.csv"))
for (p in c("theta1", "theta2", "beta_e", "beta_b", "tau_w_e")) {
  r <- summ[summ$parameter == p, ]
  log_line(sprintf("  %s = %.4f [%.4f, %.4f]", p, r$mean, r$Q2.5, r$Q97.5))
}

for (v in c("pop", "gdp")) {
  log_line("台风", v, "调节（规则B）")
  mr <- readRDS(file.path(paths$stage3_data, sprintf("modring_%s_typhoon.rds", v)))
  r2 <- fit_hazard("typhoon", variant = "moderator",
                   use_weights = USE_POST_COUNT_WEIGHTS,
                   dat = stB, mod_ring = mr, output_dir = paths$out_dir)
  s2 <- summarize_draws(r2$draws, hazard = "typhoon",
                        variant = paste0("moderator_", v, "_full_w1"))
  write_summary(s2, file.path(paths$out_dir, sprintf("typhoon_mod_%s_summary.csv", v)))
  for (p in c("gamma_mod", "delta_mod")) {
    r <- s2[s2$parameter == p, ]
    log_line(sprintf("  %s = %.4f [%.4f, %.4f] %s", p, r$mean, r$Q2.5, r$Q97.5,
                     ifelse(r$sig, "显著", "")))
  }
}
log_line("全部完成")
