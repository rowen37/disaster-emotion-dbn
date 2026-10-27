# =====================================================================
# 修复重跑：滑坡解析修复（空经济损失列）+ 剩余 6 个调节效应拟合
# mod_ring 全部按当前数据事件集重建，避免维度不匹配
# 运行:  Rscript runs/rerun_landslide_moderators.R
# =====================================================================
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1) stop("请使用 Rscript runs/<脚本名>.R 运行")
script_file <- normalizePath(sub("^--file=", "", script_arg))
source(file.path(dirname(script_file), "..", "R", "00_init.R"))
log_line <- function(...) cat(format(Sys.time(), "[%H:%M:%S]"), ..., "\n"); flush.console()

# ---------- 1) 滑坡重建（修复后解析器） ----------
cfg_ls <- hazard_configs$landslide
files <- list.files(file.path(paths$raw_input, "landslide"), pattern = "[.]csv$", full.names = TRUE)
reads <- lapply(files, read_hazard_csv)
drop_sum <- Reduce(`+`, lapply(reads, function(x) { r <- attr(x, "drop_report"); r[is.na(r)] <- 0; r }))
log_line("landslide 解析:", paste(sprintf("%s=%d", names(drop_sum), drop_sum), collapse = " "))
st_list <- lapply(seq_along(files), function(fi) {
  x <- reads[[fi]]; if (is.null(x) || nrow(x) == 0) return(NULL)
  g <- grid_one_event(as.data.frame(x), cfg_ls); if (is.null(g)) return(NULL)
  g$event_id <- tools::file_path_sans_ext(basename(files[fi])); g })
ls_data <- dplyr::bind_rows(st_list) |> attach_lag_idx()
feat <- readr::read_csv(paths$event_feat, show_col_types = FALSE,
                        col_types = readr::cols(event_id = readr::col_character()))
ls_data <- ls_data |> dplyr::mutate(event_id = as.character(event_id)) |>
  dplyr::left_join(feat, by = "event_id")
saveRDS(ls_data, file.path(paths$out_dir, "landslide_st_data.rds"))
log_line(sprintf("landslide: 帖子=%d 事件=%d 格子=%d", sum(ls_data$N, na.rm = TRUE),
                 length(unique(ls_data$event_id)), nrow(ls_data)))

# ---------- 2) 滑坡基线重跑 ----------
res <- fit_hazard("landslide", variant = "baseline", use_weights = 1,
                  dat = ls_data, output_dir = paths$out_dir)
summ <- summarize_draws(res$draws, hazard = "landslide", variant = "baseline_full_w1")
write_summary(summ, file.path(paths$out_dir, "landslide_baseline_summary.csv"))
th <- summ[summ$parameter == "theta1", ]
log_line(sprintf("landslide theta1 = %.4f [%.4f, %.4f]", th$mean, th$Q2.5, th$Q97.5))

# ---------- 3) mod_ring 按 v2 事件集重建（滑坡/泥石流） ----------
cur_data <- list(
  earthquake = readRDS(file.path(paths$out_dir, "earthquake_st_data.rds")),
  landslide = ls_data,
  debris_flow = readRDS(file.path(paths$legacy_rds, "debris_flow_st_data.rds")),
  typhoon = readRDS(file.path(paths$legacy_rds, "typhoon_st_data.rds"))
)
for (hz in c("landslide", "debris_flow")) {
  for (v in c("pop", "gdp")) {
    rp <- if (v == "pop") paths$pop_raster else paths$gdp_raster
    m <- build_mod_ring(hz, rp, dat = cur_data[[hz]])
    saveRDS(m, file.path(paths$out_dir, sprintf("modring_%s_%s.rds", v, hz)))
    log_line(hz, v, "mod_ring:", nrow(m), "x", ncol(m))
  }
}

# ---------- 4) 剩余 6 个调节拟合（tryCatch 防连锁） ----------
for (hz in c("landslide", "debris_flow", "typhoon")) {
  for (v in c("pop", "gdp")) {
    mr_file <- if (hz == "typhoon")
      file.path(paths$out_dir, sprintf("modring_%s_typhoon.rds", v))
    else file.path(paths$out_dir, sprintf("modring_%s_%s.rds", v, hz))
    mr <- readRDS(mr_file)
    log_line(hz, v, "调节效应拟合")
    ok <- tryCatch({
      res <- fit_hazard(hz, variant = "moderator", use_weights = 1,
                        dat = cur_data[[hz]], mod_ring = mr)
      summ <- summarize_draws(res$draws, hazard = hz, variant = paste0("mod_", v, "_full_w1"))
      write_summary(summ, file.path(paths$out_dir, paste0(hz, "_mod_", v, "_summary.csv")))
      for (p in c("theta1", "gamma_mod", "delta_mod")) {
        r <- summ[summ$parameter == p, ]
        log_line(sprintf("  %s %s: %.4f [%.4f, %.4f] %s", hz, p,
                         r$mean, r$Q2.5, r$Q97.5, ifelse(r$sig, "显著", "")))
      }
      TRUE
    }, error = function(e) { log_line("  FAILED:", conditionMessage(e)); FALSE })
  }
}
log_line("全部完成")
