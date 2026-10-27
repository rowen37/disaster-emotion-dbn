# Fit the optimized Stage 3 model requested for landslide and debris flow.
# Input:  v3/stage2-behavior-extraction/output[/output.csv]/{hazard}.csv
# Output: v3/stage3-dbn/output/optimized/

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1) stop("run this file with Rscript")
script_file <- normalizePath(sub("^--file=", "", script_arg))
source(file.path(dirname(dirname(script_file)), "R", "00_init.R"))

# This is an explicit archived-optimization runner, so it must not inherit the
# default selected through models/current.  Resolve its own registered files
# to prevent an original-model fit from being mislabeled as optimized.
optimized_manifest <- jsonlite::read_json(
  file.path(stage3_root, "models", "optimized_five", "model.json"),
  simplifyVector = TRUE
)
paths$stan_core <- file.path(repo_root, optimized_manifest$stan_core)
paths$stan_mod <- file.path(repo_root, optimized_manifest$stan_moderator)
source(file.path(repo_root, optimized_manifest$fit_model_r))

args <- commandArgs(trailingOnly = TRUE)
quick <- "--quick" %in% args
hazards <- c("landslide", "debris_flow")
out_dir <- paths$optimized_out_dir
dir.create(paths$stage3_data, showWarnings = FALSE, recursive = TRUE)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

message("Stage 2 input directory: ", normalizePath(paths$stage2_output))
message("Optimized output directory: ", normalizePath(out_dir, mustWork = FALSE))
if (quick) message("QUICK validation fit enabled")

all_summaries <- list()
data_diagnostics <- list()
collinearity_overview <- list()

for (hazard in hazards) {
  message("[", hazard, "] aggregate CSV and attach population/GDP decomposition")
  dat <- preprocess_hazard(
    hazard,
    input_dir = paths$stage2_output,
    output_dir = paths$stage3_data
  )

  message("[", hazard, "] collinearity diagnostics")
  coldiag <- diagnose_collinearity(
    dat, hazard_configs[[hazard]], hazard, out_dir
  )
  collinearity_overview[[hazard]] <- coldiag$condition

  data_diagnostics[[hazard]] <- data.frame(
    hazard = hazard,
    source_file = attr(dat, "stage2_source_file"),
    grid_cells = nrow(dat),
    observed_cells = sum(dat$N > 0),
    empty_cells = sum(dat$N == 0),
    occupancy_rate = mean(dat$N > 0),
    positive_post_mean = mean(dat$N[dat$N > 0]),
    positive_post_median = stats::median(dat$N[dat$N > 0]),
    context_covariates = paste(CONTEXT_COVARIATES, collapse = ";"),
    residualized_dynamics = TRUE,
    previous_time_adjacent_ring_lag = TRUE,
    hurdle_presence_positive_count = TRUE,
    covariate_prior_sd = COVARIATE_PRIOR_SD
  )

  message("[", hazard, "] optimized DBN fit")
  result <- fit_hazard(
    hazard,
    variant = "baseline",
    use_weights = USE_POST_COUNT_WEIGHTS,
    dat = dat,
    output_dir = out_dir,
    quick = quick
  )
  summary <- summarize_optimized_draws(
    result$draws, hazard, hazard_configs[[hazard]]
  )
  all_summaries[[hazard]] <- summary
  utils::write.csv(
    summary,
    file.path(out_dir, paste0(hazard, "_optimized_parameter_summary.csv")),
    row.names = FALSE
  )
}

combined <- dplyr::bind_rows(all_summaries)
utils::write.csv(
  combined,
  file.path(out_dir, "landslide_debris_optimized_results.csv"),
  row.names = FALSE
)
utils::write.csv(
  dplyr::bind_rows(data_diagnostics),
  file.path(out_dir, "data_and_optimization_diagnostics.csv"),
  row.names = FALSE
)
utils::write.csv(
  dplyr::bind_rows(collinearity_overview),
  file.path(out_dir, "collinearity_overview.csv"),
  row.names = FALSE
)

manifest <- list(
  run_type = if (quick) "quick_validation" else "final_optimized",
  input_directory = normalizePath(paths$stage2_output),
  hazards = hazards,
  bin_hours = BIN_HOURS,
  behavior_probability = BEHAVIOR_PROBABILITY_COLUMN,
  emotion_score = "mean(angry + fear + sad from raw EMO_probs)",
  observation_models = list(emotion = "logit-normal", behavior = "beta"),
  optimizations = list(
    posting_hurdle = "Bernoulli grid presence plus shifted negative-binomial positive count, jointly loaded on latent residual states",
    residualized_dynamics = "beta_e/beta_b/theta1/theta2 operate on deviations after time, space, event, intensity, population and GDP baselines",
    population_gdp = "log1p ring value decomposed into standardized event-between and event-within components",
    collinearity = paste0("correlation/VIF/condition-number diagnostics plus N(0, ", COVARIATE_PRIOR_SD, ") covariate priors"),
    spatial_propagation = "mean previous-time residual of adjacent radial rings, coefficients rho_e/rho_b"
  ),
  context_covariate_order = CONTEXT_COVARIATES,
  core_output_parameters = c("theta1", "theta2", "beta_e", "beta_b")
)
jsonlite::write_json(
  manifest,
  file.path(out_dir, "optimized_model_manifest.json"),
  pretty = TRUE, auto_unbox = TRUE
)

fmt_interval <- function(mean, low, high) {
  sprintf("%.4f [%.4f, %.4f]", mean, low, high)
}
core_keep <- c(
  "theta1", "theta2", "beta_e", "beta_b", "rho_e", "rho_b",
  "omega_occ[1]", "omega_occ[2]", "omega_count[1]", "omega_count[2]"
)
core_report <- combined %>%
  dplyr::filter(parameter %in% core_keep) %>%
  dplyr::mutate(estimate_95 = fmt_interval(mean, Q2.5, Q97.5)) %>%
  dplyr::select(hazard, parameter, estimate_95, interval_excludes_zero)
diagnostic_report <- dplyr::bind_rows(data_diagnostics) %>%
  dplyr::select(hazard, grid_cells, observed_cells, empty_cells, occupancy_rate) %>%
  dplyr::left_join(
    dplyr::bind_rows(collinearity_overview) %>%
      dplyr::select(hazard, condition_number, max_vif),
    by = "hazard"
  )
md <- c(
  "# Optimized Stage 3 results",
  "",
  paste0("Final mean-field ADVI run; input: `", normalizePath(paths$stage2_output), "`.") ,
  "",
  "## Core and spatial/selection parameters",
  "",
  "| hazard | parameter | posterior mean [95% interval] | interval excludes 0 |",
  "|---|---|---:|:---:|",
  apply(core_report, 1, function(row) {
    sprintf("| %s | %s | %s | %s |", row[[1]], row[[2]], row[[3]], row[[4]])
  }),
  "",
  "## Data and collinearity diagnostics",
  "",
  "| hazard | grids | observed | empty | occupancy | condition number | max VIF |",
  "|---|---:|---:|---:|---:|---:|---:|",
  apply(diagnostic_report, 1, function(row) {
    sprintf("| %s | %s | %s | %s | %.3f | %.3f | %.3f |",
            row[[1]], row[[2]], row[[3]], row[[4]], as.numeric(row[[5]]),
            as.numeric(row[[6]]), as.numeric(row[[7]]))
  }),
  "",
  "## Implemented optimizations",
  "",
  "1. Grid presence and positive post counts are jointly modeled with a Bernoulli/shifted-NB hurdle.",
  "2. beta_e, beta_b, theta1 and theta2 use residual states after all structured baselines are removed.",
  "3. Population and GDP are split into standardized event-between and event-within components.",
  paste0("4. Correlation, VIF and condition-number diagnostics accompany N(0, ", COVARIATE_PRIOR_SD, ") covariate priors."),
  "5. rho_e and rho_b estimate propagation from adjacent rings at the previous 12-hour bin.",
  "",
  "All estimates describe raw model-derived text scores, not calibrated emotion or real-behavior probabilities."
)
writeLines(md, file.path(out_dir, "optimized_results_summary.md"))
message("completed: ", file.path(out_dir, "landslide_debris_optimized_results.csv"))
