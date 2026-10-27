# Build the four canonical Stage 3 space-time-grid RDS files from the
# post-level RDS files produced by Stage 2 (data_stage2/).
# This script only validates and aggregates; it does not fit a DBN.

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1) {
  stop("请使用 Rscript stage3-dbn/runs/build_spatiotemporal_data.R 运行")
}
script_file <- normalizePath(sub("^--file=", "", script_arg))
source(file.path(dirname(dirname(script_file)), "R", "00_init.R"))

dir.create(paths$stage3_data, showWarnings = FALSE, recursive = TRUE)

for (hazard in c("earthquake", "landslide", "debris_flow")) {
  preprocess_hazard(hazard)
}

typhoon_data <- dedup_typhoon()
typhoon_file <- file.path(paths$stage3_data, "typhoon_st_data.rds")
saveRDS(typhoon_data, typhoon_file, compress = "gzip")
message(sprintf(
  "[typhoon] 已保存 %s（%d 行；E有效格 %d，B有效格 %d）",
  typhoon_file, nrow(typhoon_data),
  sum(!is.na(typhoon_data$E_obs)), sum(!is.na(typhoon_data$B_obs))
))

expected <- file.path(
  paths$stage3_data,
  paste0(c("earthquake", "landslide", "debris_flow", "typhoon"), "_st_data.rds")
)
stopifnot(all(file.exists(expected)))
message("四个 Stage 3 时空聚合 RDS 已生成；尚未开始 DBN 拟合。")
