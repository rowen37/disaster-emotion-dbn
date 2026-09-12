# =====================================================================
# Stage 3 - DBN pipeline: unified loader.
# Sourcing this file loads every module and the hazard configuration.
# Usage:
#   source("<repo>/stage3-dbn/R/00_init.R")
#   res <- fit_hazard("earthquake", variant = "baseline", quick = TRUE)
# =====================================================================
refactor_root <- local({
  known <- "/Users/rogerbqw/Downloads/DBNs/refactor"
  if (file.exists(file.path(known, "config/hazards.R"))) return(known)
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  cand <- if (length(file_arg) > 0) {
    # the calling script may sit at any depth; walk up until config/hazards.R is found
    d <- dirname(dirname(normalizePath(sub("^--file=", "", file_arg))))
    repeat {
      if (file.exists(file.path(d, "config/hazards.R"))) break
      upd <- dirname(d)
      if (identical(upd, d)) { d <- NULL; break }
      d <- upd
    }
    d
  } else NULL
  if (is.null(cand)) stop("无法定位 refactor 根目录，请检查路径")
  cand
})

suppressPackageStartupMessages({
  library(tidyverse)
  library(splines)
  library(data.table)
  library(jsonlite)
  library(cmdstanr)
})

source(file.path(refactor_root, "config/hazards.R"))
source(file.path(refactor_root, "R/preprocess.R"))
source(file.path(refactor_root, "R/fit_model.R"))
source(file.path(refactor_root, "R/summarize.R"))
source(file.path(refactor_root, "R/figures.R"))
source(file.path(refactor_root, "R/moderators.R"))
source(file.path(refactor_root, "R/typhoon_dedup.R"))
source(file.path(refactor_root, "R/evaluate.R"))

message("refactor 库已加载。灾种: ", paste(names(hazard_configs), collapse = ", "))
