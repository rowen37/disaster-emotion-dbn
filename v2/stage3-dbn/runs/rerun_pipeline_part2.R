# 全量重跑的后半部分（另起进程）：modring 重建 + 调节 x8 + 无耦合 x4 + 敏感性
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1) stop("请使用 Rscript runs/<脚本名>.R 运行")
script_file <- normalizePath(sub("^--file=", "", script_arg))
source(file.path(dirname(script_file), "..", "R", "00_init.R"))
log_line <- function(...) cat(format(Sys.time(), "[%H:%M:%S]"), ..., "\n"); flush.console()

ST_DIR <- paths$legacy_rds
DRAW_DIR <- paths$out_dir
hzs <- c("earthquake", "landslide", "debris_flow", "typhoon")
st_all <- setNames(lapply(hzs, function(h) readRDS(file.path(ST_DIR, paste0(h, "_st_data.rds")))), hzs)

# ---------- modring 统一缓存（缺失则重建） ----------
modring_of <- function(hz, v) {
  rp <- if (v == "pop") paths$pop_raster else paths$gdp_raster
  cache <- file.path(DRAW_DIR, sprintf("modring_%s_%s.rds", v, hz))
  mr <- tryCatch(readRDS(cache), error = function(e) NULL)
  if (is.null(mr) || nrow(mr) != length(unique(st_all[[hz]]$event_id))) {
    log_line("rebuild modring", hz, v)
    mr <- build_mod_ring(hz, rp, dat = st_all[[hz]])
    saveRDS(mr, cache)
  }
  mr
}

# ---------- 调节效应 x8 ----------
mod_summ <- list()
for (v in c("pop", "gdp")) {
  for (hz in hzs) {
    mr <- modring_of(hz, v)
    log_line("moderator fit:", hz, v)
    res <- fit_hazard(hz, variant = "moderator", use_weights = 1,
                      dat = st_all[[hz]], mod_ring = mr)
    s <- summarize_draws(res$draws, hazard = hz, variant = paste0("mod_", v, "_full_w1"))
    s$moderator <- v
    mod_summ[[paste(hz, v)]] <- s
    rm(res); gc()
  }
}
write.csv(dplyr::bind_rows(mod_summ), file.path(DRAW_DIR, "moderation_summaries.csv"), row.names = FALSE)
log_line("moderation all done")

# ---------- 无耦合 x4 ----------
nc_rows <- list()
for (hz in hzs) {
  log_line("nocoupling fit:", hz)
  hazard_configs[[hz]]$has_coupling <- 0L
  res0 <- fit_hazard(hz, variant = "baseline", use_weights = 1, dat = st_all[[hz]])
  hazard_configs[[hz]]$has_coupling <- NULL
  saveRDS(res0$draws, file.path(DRAW_DIR, paste0(hz, "_nocoupling_w1_draws.rds")))
  d1 <- readRDS(file.path(DRAW_DIR, paste0(hz, "_baseline_w1_draws.rds")))
  sd1 <- spatial_decay(d1, hazard_configs[[hz]]); sd0 <- spatial_decay(res0$draws, hazard_configs[[hz]])
  nc_rows[[hz]] <- data.frame(
    hazard = hz,
    B_spat_RMSE_coupled = sqrt(mean((obs_spatial(st_all[[hz]], "B_obs") -
        plogis(colMeans(sd1$logit_b)))^2, na.rm = TRUE)),
    B_spat_RMSE_nocoup = sqrt(mean((obs_spatial(st_all[[hz]], "B_obs") -
        plogis(colMeans(sd0$logit_b)))^2, na.rm = TRUE)))
  rm(res0, d1); gc()
}
write.csv(dplyr::bind_rows(nc_rows), file.path(DRAW_DIR, "nocoupling_spatial.csv"), row.names = FALSE)

# ---------- 敏感性：地震斜率 + 地震不加权 ----------
log_line("earthquake slope sensitivity")
hazard_configs$earthquake$has_slope <- 1L
res_sl <- fit_hazard("earthquake", variant = "baseline", use_weights = 1,
                     dat = st_all$earthquake, output_dir = DRAW_DIR)
hazard_configs$earthquake$has_slope <- NULL
write.csv(summarize_draws(res_sl$draws, hazard = "earthquake", variant = "slope_full_w1"),
          file.path(DRAW_DIR, "earthquake_slope_summary.csv"), row.names = FALSE)
sl_curve <- spatial_decay(res_sl$draws, hazard_configs$earthquake)
write.csv(data.frame(ring_km = ring_centers(hazard_configs$earthquake),
                     prob_e_slope = plogis(colMeans(sl_curve$logit_e))),
          file.path(DRAW_DIR, "earthquake_slope_spatial.csv"), row.names = FALSE)

log_line("earthquake unweighted sensitivity")
res_w0 <- fit_hazard("earthquake", variant = "baseline", use_weights = 0, dat = st_all$earthquake)
write.csv(summarize_draws(res_w0$draws, hazard = "earthquake", variant = "baseline_full_w0"),
          file.path(DRAW_DIR, "earthquake_w0_summary.csv"), row.names = FALSE)
tt0 <- temporal_trajectory(res_w0$draws, hazard_configs$earthquake)
write.csv(data.frame(time_hour = tt0$time_hour, prob_e = colMeans(tt0$prob_e)),
          file.path(DRAW_DIR, "earthquake_w0_temporal.csv"), row.names = FALSE)

log_line("PART2_ALL_DONE")
