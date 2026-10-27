# Rebuild and fit typhoon with the model selected by models/current.

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1) stop("run this file with Rscript")
script_file <- normalizePath(sub("^--file=", "", script_arg))
stage3_dir <- dirname(dirname(script_file))
source(file.path(stage3_dir, "R", "00_init.R"))

expected_model <- "original_no_five"
if (!identical(CURRENT_MODEL$id, expected_model)) {
  stop("models/current is ", CURRENT_MODEL$id, "; expected ", expected_model)
}

input_file <- file.path(
  repo_root, "v3", "stage2-behavior-extraction", "output.csv", "typhoon.csv"
)
if (!file.exists(input_file)) stop("missing requested input: ", input_file)

output_dir <- file.path(stage3_dir, "output", "current", "typhoon")
data_dir <- file.path(output_dir, "data")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

dat <- dedup_typhoon(input_file = input_file, report = TRUE)
aggregated_file <- file.path(data_dir, "typhoon_st_data.rds")
saveRDS(dat, aggregated_file, compress = "gzip")
if (!identical(normalizePath(attr(dat, "stage2_source_file")), normalizePath(input_file))) {
  stop("deduplication did not use the requested typhoon.csv")
}

preflight <- build_stan_data(
  dat, hazard_configs$typhoon, use_weights = 1, variant = "baseline"
)
expected_groups <- dplyr::n_distinct(dat$typhoon_id)
if (preflight$stan_data$has_group != 1L ||
    preflight$stan_data$G != expected_groups || expected_groups != 14L) {
  stop(
    "invalid typhoon nesting: has_group=", preflight$stan_data$has_group,
    ", fitted groups=", preflight$stan_data$G,
    ", data groups=", expected_groups
  )
}

result <- fit_hazard(
  hazard = "typhoon",
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
  stop("current model draws missing: ", paste(missing_params, collapse = ", "))
}

summary <- result$draws |>
  dplyr::select(dplyr::all_of(params)) |>
  tidyr::pivot_longer(dplyr::everything(), names_to = "parameter") |>
  dplyr::group_by(parameter) |>
  dplyr::summarise(
    mean = mean(value),
    sd = stats::sd(value),
    Q2.5 = unname(stats::quantile(value, 0.025)),
    Q50 = unname(stats::quantile(value, 0.5)),
    Q97.5 = unname(stats::quantile(value, 0.975)),
    interval_excludes_zero = Q2.5 > 0 | Q97.5 < 0,
    .groups = "drop"
  ) |>
  dplyr::mutate(hazard = "typhoon", model = CURRENT_MODEL$id, .before = 1)

summary_file <- file.path(output_dir, "typhoon_current_core_results.csv")
utils::write.csv(summary, summary_file, row.names = FALSE)

manifest <- list(
  model = CURRENT_MODEL$id,
  current_pointer = normalizePath(file.path(stage3_dir, "models", "current")),
  input_file = normalizePath(input_file),
  aggregated_data = normalizePath(aggregated_file),
  rows = nrow(dat),
  track_point_events = dplyr::n_distinct(dat$event_id),
  typhoon_groups = expected_groups,
  nested_group_model = TRUE,
  grid_cells_with_posts = sum(dat$N > 0),
  total_grid_cells = nrow(dat),
  bin_hours = BIN_HOURS,
  inference = fit_settings,
  reported_parameters = params
)
jsonlite::write_json(
  manifest,
  file.path(output_dir, "typhoon_current_manifest.json"),
  pretty = TRUE,
  auto_unbox = TRUE
)

message("completed: ", summary_file)
