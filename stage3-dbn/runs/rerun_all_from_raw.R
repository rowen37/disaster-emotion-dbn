# =====================================================================
# Stage 3 - run script: rebuild three hazards from raw CSVs
# (deterministic parser) and rerun all baselines and moderator models.
# Background: fread truncation affected every hazard (median retention in
# the legacy landslide data 32.8%, debris flow 21.4%); earlier refactored
# EQ/LS/DF fits still used the truncated RDS files.
# Steps:
#   1) rebuild st_data for EQ/LS/DF from raw CSVs (deterministic parser,
#      no dedup - one file per event)
#   2) rerun the three baselines (full settings)
#   3) rebuild the typhoon mod_ring from the deduplicated 51 points
#   4) fit the 8 moderator models (pop/gdp x 4 hazards, all on new data)
# =====================================================================
source("/Users/rogerbqw/Downloads/DBNs/refactor/R/00_init.R")

log_line <- function(...) cat(format(Sys.time(), "[%H:%M:%S]"), ..., "\n"); flush.console()

# ---------- 1) rebuild st_data for the three hazards ----------
rebuild_hazard <- function(hazard) {
  cfg <- hazard_configs[[hazard]]
  files <- list.files(file.path(paths$raw_input, hazard), pattern = "[.]csv$", full.names = TRUE)
  log_line(hazard, "重建:", length(files), "个事件文件")
  reads <- lapply(files, read_hazard_csv)
  drop_sum <- Reduce(`+`, lapply(reads, function(x) {
    r <- attr(x, "drop_report"); r[is.na(r)] <- 0; r }))
  log_line(hazard, "解析丢弃合计:",
           paste(sprintf("%s=%d", names(drop_sum), drop_sum), collapse = " "))

  st_list <- lapply(seq_along(files), function(fi) {
    x <- reads[[fi]]
    if (is.null(x) || nrow(x) == 0) return(NULL)
    g <- grid_one_event(as.data.frame(x), cfg)
    if (is.null(g)) return(NULL)
    g$event_id <- tools::file_path_sans_ext(basename(files[fi]))
    g
  })
  st_data <- dplyr::bind_rows(st_list) |> attach_lag_idx()
  feat <- readr::read_csv(paths$event_feat, show_col_types = FALSE,
                          col_types = readr::cols(event_id = readr::col_character()))
  st_data <- st_data |>
    dplyr::mutate(event_id = as.character(event_id)) |>
    dplyr::left_join(feat, by = "event_id")

  old <- readRDS(file.path(paths$legacy_rds, paste0(hazard, "_st_data.rds")))
  n_old <- sum(old$N, na.rm = TRUE); n_new <- sum(st_data$N, na.rm = TRUE)
  log_line(sprintf("%s: 帖子数 旧(截断)=%d -> 新=%d (%.1f 倍) | 有观测格子 %d/%d",
                   hazard, n_old, n_new, n_new / n_old,
                   sum(!is.na(st_data$E_obs)), nrow(st_data)))
  saveRDS(st_data, file.path(paths$out_dir, paste0(hazard, "_st_data_v2.rds")))
  st_data
}

v2 <- list()
for (hz in c("earthquake", "landslide", "debris_flow")) v2[[hz]] <- rebuild_hazard(hz)

# ---------- 2) rerun the three baselines ----------
for (hz in c("earthquake", "landslide", "debris_flow")) {
  log_line(hz, "基线拟合(新数据)")
  res <- fit_hazard(hz, variant = "baseline", use_weights = 1,
                    dat = v2[[hz]], output_dir = paths$out_dir)
  summ <- summarize_draws(res$draws, hazard = hz, variant = "baseline_v2_full_w1")
  write_summary(summ, file.path(paths$out_dir, paste0(hz, "_baseline_v2_summary.csv")))
  th <- summ[summ$parameter == "theta1", ]
  log_line(sprintf("%s: theta1 = %.4f [%.4f, %.4f]", hz, th$mean, th$Q2.5, th$Q97.5))
}

# ---------- 3) rebuild the typhoon mod_ring (deduplicated, 51 points) ----------
ty_dat <- readRDS(file.path(paths$out_dir, "typhoon_st_data_dedup.rds"))
for (v in c("pop", "gdp")) {
  rp <- if (v == "pop") paths$pop_raster else paths$gdp_raster
  m <- build_mod_ring("typhoon", rp, dat = ty_dat)
  saveRDS(m, file.path(paths$out_dir, sprintf("modring_%s_typhoon_v2.rds", v)))
  log_line(sprintf("typhoon mod_ring[%s]: %dx%d", v, nrow(m), ncol(m)))
}

# ---------- 4) fit the 8 moderator models ----------
for (hz in c("earthquake", "landslide", "debris_flow", "typhoon")) {
  dat <- if (hz == "typhoon") ty_dat else v2[[hz]]
  for (v in c("pop", "gdp")) {
    rp <- if (v == "pop") paths$pop_raster else paths$gdp_raster
    if (hz == "typhoon") {
      mr <- readRDS(file.path(paths$out_dir, sprintf("modring_%s_typhoon_v2.rds", v)))
    } else {
      mr <- readRDS(file.path(paths$out_dir, sprintf("modring_%s_%s.rds", v, hz)))
    }
    log_line(hz, v, "调节效应拟合")
    res <- fit_hazard(hz, variant = "moderator", use_weights = 1,
                      dat = dat, mod_ring = mr)
    summ <- summarize_draws(res$draws, hazard = hz,
                            variant = paste0("mod_", v, "_v2_full_w1"))
    write_summary(summ, file.path(paths$out_dir, paste0(hz, "_mod_", v, "_v2_summary.csv")))
    for (p in c("theta1", "gamma_mod", "delta_mod")) {
      r <- summ[summ$parameter == p, ]
      log_line(sprintf("  %s %s: %.4f [%.4f, %.4f] %s", hz, p,
                       r$mean, r$Q2.5, r$Q97.5, ifelse(r$sig, "显著", "")))
    }
  }
}
log_line("全部完成")
