# =====================================================================
# Stage 3 - DBN pipeline: unified loader.
# Sourcing this file loads every module and the hazard configuration.
# Usage:
#   source("<repo>/stage3-dbn/R/00_init.R")
#   res <- fit_hazard("earthquake", variant = "baseline", quick = TRUE)
# =====================================================================
stage3_root <- local({
  source_file <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  starts <- c(
    if (!is.null(source_file)) dirname(normalizePath(source_file)) else NULL,
    if (length(file_arg) > 0) dirname(normalizePath(sub("^--file=", "", file_arg[[1]]))) else NULL,
    getwd()
  )
  for (start in unique(starts)) {
    d <- start
    repeat {
      if (file.exists(file.path(d, "config/hazards.R"))) return(d)
      upd <- dirname(d)
      if (identical(upd, d)) break
      d <- upd
    }
  }
  stop("无法定位 stage3-dbn 根目录；请通过 Rscript 运行 runs/ 下的脚本，或 source 本文件")
})

suppressPackageStartupMessages({
  library(tidyverse)
  library(splines)
  library(data.table)
  library(jsonlite)
  library(cmdstanr)
})

source(file.path(stage3_root, "config/hazards.R"))
source(file.path(stage3_root, "R/preprocess.R"))
source(file.path(stage3_root, "R/fit_model.R"))
source(file.path(stage3_root, "R/summarize.R"))
source(file.path(stage3_root, "R/figures.R"))
source(file.path(stage3_root, "R/moderators.R"))
source(file.path(stage3_root, "R/typhoon_dedup.R"))
source(file.path(stage3_root, "R/evaluate.R"))

message("stage3-dbn 函数库已加载。灾种: ", paste(names(hazard_configs), collapse = ", "))
