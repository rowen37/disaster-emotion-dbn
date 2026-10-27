# Fit the pre-optimization Stage 3 model to the current v3 aggregated inputs.
#
# The original implementation is loaded from v2/stage3-dbn, where it is kept
# unchanged.  In particular this fit has no spatial-neighbour term, posting
# hurdle/count model, residualized dynamics, population/GDP decomposition, or
# added context-covariate regularization.

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1) stop("run this file with Rscript")
script_file <- normalizePath(sub("^--file=", "", script_arg))
v3_root <- dirname(dirname(script_file))
repo_root <- normalizePath(file.path(v3_root, "..", ".."))
original_root <- file.path(repo_root, "v2", "stage3-dbn")

# Keep the current v3 hazard definitions (including the corrected intensity
# fields) while replacing only the five model changes with the original core.
v3_config <- new.env(parent = globalenv())
assign("stage3_root", v3_root, envir = v3_config)
sys.source(file.path(v3_root, "config", "hazards.R"), envir = v3_config)
source(file.path(original_root, "R", "00_init.R"))

hazards <- c("landslide", "debris_flow")
hazard_configs[hazards] <- v3_config$hazard_configs[hazards]
fit_settings <- v3_config$fit_settings
input_dir <- file.path(v3_root, "data")
output_dir <- file.path(v3_root, "output", "original")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

all_results <- list()
for (hazard in hazards) {
  input_file <- file.path(input_dir, paste0(hazard, "_st_data.rds"))
  if (!file.exists(input_file)) stop("missing current v3 grid data: ", input_file)

  dat <- readRDS(input_file)
  message("[", hazard, "] original Stage 3 fit: ", nrow(dat), " grid cells")
  result <- fit_hazard(
    hazard = hazard,
    variant = "baseline",
    use_weights = 1,
    dat = dat,
    settings = fit_settings,
    output_dir = output_dir,
    quick = FALSE
  )

  params <- c("theta1", "theta2", "beta_e", "beta_b")
  missing_params <- setdiff(params, names(result$draws))
  if (length(missing_params) > 0) {
    stop("original model draws missing: ", paste(missing_params, collapse = ", "))
  }

  one <- result$draws |>
    dplyr::select(dplyr::all_of(params)) |>
    tidyr::pivot_longer(dplyr::everything(), names_to = "parameter") |>
    dplyr::group_by(parameter) |>
    dplyr::summarise(
      mean = mean(value),
      sd = stats::sd(value),
      Q2.5 = unname(stats::quantile(value, 0.025)),
      Q50 = unname(stats::quantile(value, 0.5)),
      Q97.5 = unname(stats::quantile(value, 0.975)),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      hazard = hazard,
      interval_excludes_zero = Q2.5 > 0 | Q97.5 < 0,
      .before = 1
    )
  all_results[[hazard]] <- one
}

combined <- dplyr::bind_rows(all_results)
result_file <- file.path(output_dir, "landslide_debris_original_core_results.csv")
utils::write.csv(combined, result_file, row.names = FALSE)

manifest <- list(
  run_type = "original_pre_optimization_stage3",
  input_data = normalizePath(input_dir),
  original_model_source = normalizePath(file.path(original_root, "stan", "dbn_core.stan")),
  hazards = hazards,
  bin_hours = BIN_HOURS,
  use_post_count_weights = TRUE,
  excluded_modifications = c(
    "adjacent-grid spatial lag",
    "posting-presence and positive-count joint model",
    "residualized dynamic coefficients",
    "population/GDP between-within decomposition",
    "new context-covariate diagnostics and regularization"
  ),
  reported_parameters = c("theta1", "theta2", "beta_e", "beta_b"),
  inference = fit_settings
)
jsonlite::write_json(
  manifest,
  file.path(output_dir, "original_model_manifest.json"),
  pretty = TRUE,
  auto_unbox = TRUE
)

message("completed: ", result_file)
