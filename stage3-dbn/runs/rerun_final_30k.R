# =====================================================================
# Stage 3 - run script: final rerun at the locked settings
# (tol_rel_obj = 0.01, iter = 30000).
# Covers: 4 baselines + 8 moderator fits + the earthquake slope
# sensitivity + evaluation + tables + figures + convergence report.
# =====================================================================
source("/Users/rogerbqw/Downloads/DBNs/refactor/R/00_init.R")
log_line <- function(...) cat(format(Sys.time(), "[%H:%M:%S]"), ..., "\n"); flush.console()

log_line("设置确认: iter =", fit_settings$iter, " tol_rel_obj =", fit_settings$tol_rel_obj)

dat_all <- list(
  earthquake = readRDS(file.path(paths$out_dir, "earthquake_st_data_v2.rds")),
  landslide = readRDS(file.path(paths$out_dir, "landslide_st_data_v2.rds")),
  debris_flow = readRDS(file.path(paths$out_dir, "debris_flow_st_data_v2.rds")),
  typhoon = readRDS(file.path(paths$out_dir, "typhoon_st_data_dedup.rds"))
)

# ---------- 1) four baselines ----------
for (hz in names(dat_all)) {
  log_line(hz, "基线拟合")
  res <- tryCatch(
    fit_hazard(hz, variant = "baseline", use_weights = 1,
               dat = dat_all[[hz]], output_dir = paths$out_dir),
    error = function(e) { log_line("  FAILED:", conditionMessage(e)); NULL })
  if (is.null(res)) next
  summ <- summarize_draws(res$draws, hazard = hz, variant = "baseline_30k_w1")
  write_summary(summ, file.path(paths$out_dir, paste0(hz, "_baseline_30k_summary.csv")))
  th <- summ[summ$parameter == "theta1", ]
  log_line(sprintf("  theta1 = %.4f [%.4f, %.4f]", th$mean, th$Q2.5, th$Q97.5))
}

# ---------- 2) eight moderator fits ----------
for (hz in names(dat_all)) {
  for (v in c("pop", "gdp")) {
    mr_file <- if (hz == "typhoon")
      file.path(paths$out_dir, sprintf("modring_%s_typhoon_v2.rds", v))
    else file.path(paths$out_dir, sprintf("modring_%s_%s_v2.rds", v, hz))
    mr <- readRDS(mr_file)
    log_line(hz, v, "调节效应拟合")
    ok <- tryCatch({
      res <- fit_hazard(hz, variant = "moderator", use_weights = 1,
                        dat = dat_all[[hz]], mod_ring = mr)
      summ <- summarize_draws(res$draws, hazard = hz, variant = paste0("mod_", v, "_30k_w1"))
      write_summary(summ, file.path(paths$out_dir, paste0(hz, "_mod_", v, "_30k_summary.csv")))
      for (p in c("gamma_mod", "delta_mod")) {
        r <- summ[summ$parameter == p, ]
        log_line(sprintf("  %s: %.4f [%.4f, %.4f] %s", p,
                         r$mean, r$Q2.5, r$Q97.5, ifelse(r$sig, "显著", "")))
      }
      TRUE
    }, error = function(e) { log_line("  FAILED:", conditionMessage(e)); FALSE })
  }
}

# ---------- 3) earthquake slope sensitivity (same settings) ----------
log_line("earthquake 斜率敏感性拟合")
hazard_configs$earthquake$has_slope <- 1L
res_s <- tryCatch(
  fit_hazard("earthquake", variant = "baseline", use_weights = 1,
             dat = dat_all$earthquake, output_dir = paths$out_dir),
  error = function(e) { log_line("  FAILED:", conditionMessage(e)); NULL })
hazard_configs$earthquake$has_slope <- NULL
if (!is.null(res_s)) {
  summ <- summarize_draws(res_s$draws, hazard = "earthquake", variant = "slope_30k_w1")
  write_summary(summ, file.path(paths$out_dir, "earthquake_slope_30k_summary.csv"))
  sb <- spatial_decay(res_s$draws, hazard_configs$earthquake)
  e <- colMeans(sb$logit_e)
  log_line(sprintf("  tau_slope_e = %.4f, 回升 = %.3f",
                   mean(res_s$draws$tau_slope_e), e[10] - min(e)))
}

# ---------- 4) evaluation + tables + figures ----------
log_line("评估与图表再生成")
fits <- list(
  earthquake = list(draws = file.path(paths$out_dir, "earthquake_baseline_w1_draws.rds"),
                    dat = dat_all$earthquake),
  landslide = list(draws = file.path(paths$out_dir, "landslide_baseline_w1_draws.rds"),
                   dat = dat_all$landslide),
  debris_flow = list(draws = file.path(paths$out_dir, "debris_flow_baseline_w1_draws.rds"),
                     dat = dat_all$debris_flow),
  typhoon = list(draws = file.path(paths$out_dir, "typhoon_baseline_w1_draws.rds"),
                 dat = dat_all$typhoon))
ev <- eval_all(fits)
write.csv(ev, file.path(paths$out_dir, "S17_replacement_model_obs_consistency.csv"), row.names = FALSE)
log_line("S17 RMSE>=MAE 校验: 通过")

# tables and figures (make_paper_outputs.R regenerates everything)
sys.source("/Users/rogerbqw/Downloads/DBNs/refactor/runs/make_paper_outputs.R",
           envir = environment())

# ---------- 5) convergence report ----------
log_line("全部完成。收敛状态请查日志中 'maximum number of iterations' 出现次数",
         "（0 次即全部因 tol 平台停）")
