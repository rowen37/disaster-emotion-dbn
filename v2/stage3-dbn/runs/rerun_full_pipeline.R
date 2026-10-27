# =====================================================================
# 全量重跑：确定性解析器 + 台风去重 + 基线/调节/敏感性 + 论文数值导出
# 输入: paths$raw_input（逐事件 CSV 布局, 含 raw_prob）
# 输出: st_data -> paths$legacy_rds; draws/表格 -> paths$out_dir
# =====================================================================
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1) stop("请使用 Rscript runs/<脚本名>.R 运行")
script_file <- normalizePath(sub("^--file=", "", script_arg))
source(file.path(dirname(script_file), "..", "R", "00_init.R"))
log_line <- function(...) cat(format(Sys.time(), "[%H:%M:%S]"), ..., "\n"); flush.console()

ST_DIR <- paths$legacy_rds
DRAW_DIR <- paths$out_dir
dir.create(ST_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(DRAW_DIR, showWarnings = FALSE, recursive = TRUE)

feat <- readr::read_csv(paths$event_feat, show_col_types = FALSE,
                        col_types = readr::cols(event_id = readr::col_character()))

# ---------- 1) 重建 st_data（三灾种 + 台风去重） ----------
st_all <- list()
for (hz in c("earthquake", "landslide", "debris_flow")) {
  log_line("rebuild", hz)
  files <- list.files(file.path(paths$raw_input, hz), pattern = "[.]csv$", full.names = TRUE)
  st_list <- lapply(files, function(f) {
    g <- grid_one_event(read_hazard_csv(f), hazard_configs[[hz]])
    if (is.null(g)) return(NULL)
    g$event_id <- tools::file_path_sans_ext(basename(f))
    g
  })
  st <- dplyr::bind_rows(Filter(Negate(is.null), st_list)) |> attach_lag_idx()
  st <- st |> dplyr::mutate(event_id = as.character(event_id)) |>
    dplyr::left_join(feat, by = "event_id")
  st_all[[hz]] <- st
  saveRDS(st, file.path(ST_DIR, paste0(hz, "_st_data.rds")))
  log_line(hz, "st_data:", nrow(st), "rows,", length(unique(st$event_id)), "events, N sum =",
           sum(st$N, na.rm = TRUE), "cells observed:", sum(!is.na(st$E_obs)), "/", nrow(st))
}
log_line("rebuild typhoon (dedup)")
ty <- dedup_typhoon()
st_all$typhoon <- ty
saveRDS(ty, file.path(ST_DIR, "typhoon_st_data.rds"))
log_line("typhoon st_data:", nrow(ty), "rows,", length(unique(ty$event_id)), "events, N sum =",
         sum(ty$N, na.rm = TRUE))

# ---------- 2) 基线 x4 ----------
base_summ <- list(); traj <- list(); spat <- list(); evals <- list()
for (hz in names(st_all)) {
  log_line("baseline fit:", hz)
  res <- fit_hazard(hz, variant = "baseline", use_weights = 1,
                    dat = st_all[[hz]], output_dir = DRAW_DIR)
  base_summ[[hz]] <- summarize_draws(res$draws, hazard = hz, variant = "baseline_full_w1")
  tt <- temporal_trajectory(res$draws, hazard_configs[[hz]])
  traj[[hz]] <- data.frame(hazard = hz, time_hour = tt$time_hour,
                           prob_e = colMeans(tt$prob_e), prob_b = colMeans(tt$prob_b))
  sd_curve <- spatial_decay(res$draws, hazard_configs[[hz]])
  spat[[hz]] <- data.frame(hazard = hz, ring_km = ring_centers(hazard_configs[[hz]]),
                           prob_e = plogis(colMeans(sd_curve$logit_e)),
                           prob_b = plogis(colMeans(sd_curve$logit_b)))
  ev <- eval_hazard(res$draws, st_all[[hz]], hazard_configs[[hz]]); ev$hazard <- hz
  evals[[hz]] <- ev
  log_line(hz, "baseline done, theta1 =",
           round(base_summ[[hz]]$mean[base_summ[[hz]]$parameter == "theta1"], 4))
}
write.csv(dplyr::bind_rows(base_summ), file.path(DRAW_DIR, "baseline_summaries.csv"), row.names = FALSE)
write.csv(dplyr::bind_rows(traj), file.path(DRAW_DIR, "temporal_curves.csv"), row.names = FALSE)
write.csv(dplyr::bind_rows(spat), file.path(DRAW_DIR, "spatial_curves.csv"), row.names = FALSE)
write.csv(dplyr::bind_rows(evals), file.path(DRAW_DIR, "eval_rmse.csv"), row.names = FALSE)

# theta1 两两差（S16 重做）
hzs <- names(st_all); pd_rows <- list()
for (i in seq_along(hzs)) for (j in seq_along(hzs)) {
  if (i >= j) next
  d1 <- readRDS(file.path(DRAW_DIR, paste0(hzs[i], "_baseline_w1_draws.rds")))
  d2 <- readRDS(file.path(DRAW_DIR, paste0(hzs[j], "_baseline_w1_draws.rds")))
  dif <- d1$theta1 - d2$theta1
  pd_rows[[length(pd_rows)+1]] <- data.frame(
    comparison = paste0(hzs[i], " - ", hzs[j]),
    difference_mean = mean(dif), Q2.5 = quantile(dif, .025), Q97.5 = quantile(dif, .975))
}
write.csv(dplyr::bind_rows(pd_rows), file.path(DRAW_DIR, "theta1_pairwise.csv"), row.names = FALSE)

# ---------- 3) 调节效应 x8（modring 统一缓存于 out_dir，缺失则重建） ----------
mod_summ <- list()
for (v in c("pop", "gdp")) {
  rp <- if (v == "pop") paths$pop_raster else paths$gdp_raster
  for (hz in hzs) {
    mr_cache <- file.path(DRAW_DIR, sprintf("modring_%s_%s.rds", v, hz))
    mr <- tryCatch(readRDS(mr_cache), error = function(e) NULL)
    if (is.null(mr) || nrow(mr) != length(unique(st_all[[hz]]$event_id))) {
      log_line("rebuild modring", hz, v)
      mr <- build_mod_ring(hz, rp, dat = st_all[[hz]])
      saveRDS(mr, mr_cache)
    }
    log_line("moderator fit:", hz, v)
    res <- fit_hazard(hz, variant = "moderator", use_weights = 1,
                      dat = st_all[[hz]], mod_ring = mr)
    s <- summarize_draws(res$draws, hazard = hz, variant = paste0("mod_", v, "_full_w1"))
    s$moderator <- v
    mod_summ[[paste(hz, v)]] <- s
  }
}
write.csv(dplyr::bind_rows(mod_summ), file.path(DRAW_DIR, "moderation_summaries.csv"), row.names = FALSE)

# ---------- 4) 无耦合 x4 + S13 ----------
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
}
write.csv(dplyr::bind_rows(nc_rows), file.path(DRAW_DIR, "nocoupling_spatial.csv"), row.names = FALSE)

# ---------- 5) 敏感性：地震事件斜率 + 地震不加权 ----------
log_line("earthquake slope sensitivity")
hazard_configs$earthquake$has_slope <- 1L
res_sl <- fit_hazard("earthquake", variant = "baseline", use_weights = 1,
                     dat = st_all$earthquake, output_dir = DRAW_DIR)
hazard_configs$earthquake$has_slope <- NULL
s_sl <- summarize_draws(res_sl$draws, hazard = "earthquake", variant = "slope_full_w1")
write.csv(s_sl, file.path(DRAW_DIR, "earthquake_slope_summary.csv"), row.names = FALSE)
sl_curve <- spatial_decay(res_sl$draws, hazard_configs$earthquake)
write.csv(data.frame(ring_km = ring_centers(hazard_configs$earthquake),
                     prob_e_slope = plogis(colMeans(sl_curve$logit_e))),
          file.path(DRAW_DIR, "earthquake_slope_spatial.csv"), row.names = FALSE)

log_line("earthquake unweighted sensitivity")
res_w0 <- fit_hazard("earthquake", variant = "baseline", use_weights = 0, dat = st_all$earthquake)
s_w0 <- summarize_draws(res_w0$draws, hazard = "earthquake", variant = "baseline_full_w0")
write.csv(s_w0, file.path(DRAW_DIR, "earthquake_w0_summary.csv"), row.names = FALSE)
tt0 <- temporal_trajectory(res_w0$draws, hazard_configs$earthquake)
write.csv(data.frame(time_hour = tt0$time_hour, prob_e = colMeans(tt0$prob_e)),
          file.path(DRAW_DIR, "earthquake_w0_temporal.csv"), row.names = FALSE)

log_line("PIPELINE_ALL_DONE")
