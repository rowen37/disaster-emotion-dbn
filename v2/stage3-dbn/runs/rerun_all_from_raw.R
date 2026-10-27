# =====================================================================
# 三灾种数据重建（确定性解析）+ 全部基线与调节效应重跑
# 背景：fread 截断影响所有灾种（滑坡旧数据中位保留率32.8%、泥石流21.4%），
#       此前重构版的 EQ/LS/DF 拟合仍基于旧 RDS（截断数据）。
# 流程：
#   1) EQ/LS/DF 从原始 CSV 重建 st_data（确定性解析器，无去重——每事件单文件）
#   2) 三灾种基线重跑（全设置）
#   3) 台风 mod_ring 基于去重后 51 点重建
#   4) 8 个调节效应拟合（pop/gdp × 4 灾种，全部基于新数据）
# 运行:  Rscript runs/rerun_all_from_raw.R
# =====================================================================
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1) stop("请使用 Rscript runs/<脚本名>.R 运行")
script_file <- normalizePath(sub("^--file=", "", script_arg))
source(file.path(dirname(script_file), "..", "R", "00_init.R"))

log_line <- function(...) cat(format(Sys.time(), "[%H:%M:%S]"), ..., "\n"); flush.console()

# ---------- 1) 三灾种 st_data 重建 ----------
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

  log_line(sprintf("%s: 帖子数 %d | 有观测格子 %d/%d",
                   hazard, sum(st_data$N, na.rm = TRUE),
                   sum(!is.na(st_data$E_obs)), nrow(st_data)))
  saveRDS(st_data, file.path(paths$legacy_rds, paste0(hazard, "_st_data.rds")))
  st_data
}

st_all <- list()
for (hz in c("earthquake", "landslide", "debris_flow")) st_all[[hz]] <- rebuild_hazard(hz)

# ---------- 2) 三灾种基线重跑 ----------
for (hz in c("earthquake", "landslide", "debris_flow")) {
  log_line(hz, "基线拟合(新数据)")
  res <- fit_hazard(hz, variant = "baseline", use_weights = 1,
                    dat = st_all[[hz]], output_dir = paths$out_dir)
  summ <- summarize_draws(res$draws, hazard = hz, variant = "baseline_full_w1")
  write_summary(summ, file.path(paths$out_dir, paste0(hz, "_baseline_summary.csv")))
  th <- summ[summ$parameter == "theta1", ]
  log_line(sprintf("%s: theta1 = %.4f [%.4f, %.4f]", hz, th$mean, th$Q2.5, th$Q97.5))
}

# ---------- 3) 台风 mod_ring 重建（去重后 51 点） ----------
ty_dat <- readRDS(file.path(paths$legacy_rds, "typhoon_st_data.rds"))
for (v in c("pop", "gdp")) {
  rp <- if (v == "pop") paths$pop_raster else paths$gdp_raster
  m <- build_mod_ring("typhoon", rp, dat = ty_dat)
  saveRDS(m, file.path(paths$out_dir, sprintf("modring_%s_typhoon.rds", v)))
  log_line(sprintf("typhoon mod_ring[%s]: %dx%d", v, nrow(m), ncol(m)))
}

# ---------- 4) 8 个调节效应拟合 ----------
for (hz in c("earthquake", "landslide", "debris_flow", "typhoon")) {
  dat <- if (hz == "typhoon") ty_dat else st_all[[hz]]
  for (v in c("pop", "gdp")) {
    rp <- if (v == "pop") paths$pop_raster else paths$gdp_raster
    if (hz == "typhoon") {
      mr <- readRDS(file.path(paths$out_dir, sprintf("modring_%s_typhoon.rds", v)))
    } else {
      mr <- readRDS(file.path(paths$out_dir, sprintf("modring_%s_%s.rds", v, hz)))
    }
    log_line(hz, v, "调节效应拟合")
    res <- fit_hazard(hz, variant = "moderator", use_weights = 1,
                      dat = dat, mod_ring = mr)
    summ <- summarize_draws(res$draws, hazard = hz,
                            variant = paste0("mod_", v, "_full_w1"))
    write_summary(summ, file.path(paths$out_dir, paste0(hz, "_mod_", v, "_summary.csv")))
    for (p in c("theta1", "gamma_mod", "delta_mod")) {
      r <- summ[summ$parameter == p, ]
      log_line(sprintf("  %s %s: %.4f [%.4f, %.4f] %s", hz, p,
                       r$mean, r$Q2.5, r$Q97.5, ifelse(r$sig, "显著", "")))
    }
  }
}
log_line("全部完成")
