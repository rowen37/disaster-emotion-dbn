# Rebuild all four canonical Stage 3 datasets from the current Stage 2
# predictions, then fit the four baselines and eight moderator models.
# Use build_spatiotemporal_data.R instead when only aggregation is wanted.

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1) stop("请使用 Rscript 运行本脚本")
script_file <- normalizePath(sub("^--file=", "", script_arg))
source(file.path(dirname(dirname(script_file)), "R", "00_init.R"))

log_line <- function(...) {
  cat(format(Sys.time(), "[%H:%M:%S]"), ..., "\n")
  flush.console()
}

dir.create(paths$stage3_data, showWarnings = FALSE, recursive = TRUE)
dir.create(paths$out_dir, showWarnings = FALSE, recursive = TRUE)

# 1) Canonical post -> grid aggregation.
for (hazard in c("earthquake", "landslide", "debris_flow")) {
  log_line(hazard, "从 Stage 2 RDS 重建时空格")
  preprocess_hazard(hazard)
}
log_line("typhoon", "按台风内去重后重建时空格")
typhoon_data <- dedup_typhoon()
saveRDS(
  typhoon_data,
  file.path(paths$stage3_data, "typhoon_st_data.rds"),
  compress = "gzip"
)

hazards <- names(hazard_configs)
data_all <- setNames(lapply(hazards, function(hazard) {
  readRDS(file.path(paths$stage3_data, paste0(hazard, "_st_data.rds")))
}), hazards)

# 2) Four baseline models.
for (hazard in hazards) {
  log_line(hazard, "基线拟合")
  result <- fit_hazard(
    hazard, variant = "baseline", use_weights = USE_POST_COUNT_WEIGHTS,
    dat = data_all[[hazard]], output_dir = paths$out_dir
  )
  summary <- summarize_draws(
    result$draws, hazard = hazard, variant = "baseline_full_w1"
  )
  write_summary(
    summary,
    file.path(paths$out_dir, paste0(hazard, "_baseline_summary.csv"))
  )
}

# 3) Population/GDP moderator matrices and models.
for (hazard in hazards) {
  for (moderator in c("pop", "gdp")) {
    raster_path <- if (moderator == "pop") paths$pop_raster else paths$gdp_raster
    log_line(hazard, moderator, "提取环形协变量并拟合调节模型")
    mod_ring <- build_mod_ring(
      hazard, raster_path, dat = data_all[[hazard]]
    )
    saveRDS(
      mod_ring,
      file.path(paths$stage3_data, sprintf("modring_%s_%s.rds", moderator, hazard)),
      compress = "gzip"
    )
    result <- fit_hazard(
      hazard, variant = "moderator", use_weights = USE_POST_COUNT_WEIGHTS,
      dat = data_all[[hazard]], mod_ring = mod_ring,
      output_dir = paths$out_dir
    )
    summary <- summarize_draws(
      result$draws, hazard = hazard,
      variant = paste0("moderator_", moderator, "_full_w1")
    )
    write_summary(
      summary,
      file.path(paths$out_dir, sprintf("%s_mod_%s_summary.csv", hazard, moderator))
    )
  }
}

log_line("全部 Stage 3 聚合与拟合完成")
