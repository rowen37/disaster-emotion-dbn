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

# The default fitting implementation is selected by models/current.  Keeping
# the pointer outside the loader makes switching explicit and auditable.
if (!file.exists(paths$current_model)) {
  stop("Stage 3 current model pointer is missing: ", paths$current_model)
}
CURRENT_MODEL <- jsonlite::read_json(paths$current_model, simplifyVector = TRUE)
required_model_fields <- c("id", "fit_model_r", "stan_core", "stan_moderator")
missing_model_fields <- setdiff(required_model_fields, names(CURRENT_MODEL))
if (length(missing_model_fields) > 0) {
  stop("Current model manifest is missing: ", paste(missing_model_fields, collapse = ", "))
}
current_fit_model <- file.path(repo_root, CURRENT_MODEL$fit_model_r)
paths$stan_core <- file.path(repo_root, CURRENT_MODEL$stan_core)
paths$stan_mod <- file.path(repo_root, CURRENT_MODEL$stan_moderator)
current_model_files <- c(current_fit_model, paths$stan_core, paths$stan_mod)
if (any(!file.exists(current_model_files))) {
  stop(
    "Current model points to missing files: ",
    paste(current_model_files[!file.exists(current_model_files)], collapse = ", ")
  )
}
source(current_fit_model)
source(file.path(stage3_root, "R/summarize.R"))
source(file.path(stage3_root, "R/figures.R"))
source(file.path(stage3_root, "R/moderators.R"))
source(file.path(stage3_root, "R/diagnostics.R"))
source(file.path(stage3_root, "R/typhoon_dedup.R"))
source(file.path(stage3_root, "R/evaluate.R"))

message(
  "stage3-dbn 函数库已加载。current model: ", CURRENT_MODEL$id,
  "；灾种: ", paste(names(hazard_configs), collapse = ", ")
)
