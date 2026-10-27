# Rebuild and fit earthquake with the model selected by models/current.

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
  repo_root, "v3", "stage2-behavior-extraction", "output.csv", "earthquake.csv"
)
if (!file.exists(input_file)) stop("missing requested input: ", input_file)

output_dir <- file.path(stage3_dir, "output", "current", "earthquake")
data_dir <- file.path(output_dir, "data")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

dat <- preprocess_hazard(
  "earthquake",
  input_dir = dirname(input_file),
  output_dir = data_dir
)
if (!identical(normalizePath(attr(dat, "stage2_source_file")), normalizePath(input_file))) {
  stop("preprocessing did not use the requested earthquake.csv")
}

result <- fit_hazard(
  hazard = "earthquake",
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
  dplyr::mutate(hazard = "earthquake", model = CURRENT_MODEL$id, .before = 1)

summary_file <- file.path(output_dir, "earthquake_current_core_results.csv")
utils::write.csv(summary, summary_file, row.names = FALSE)

manifest <- list(
  model = CURRENT_MODEL$id,
  current_pointer = normalizePath(file.path(stage3_dir, "models", "current")),
  input_file = normalizePath(input_file),
  aggregated_data = normalizePath(file.path(data_dir, "earthquake_st_data.rds")),
  rows = nrow(dat),
  events = dplyr::n_distinct(dat$event_id),
  grid_cells_with_posts = sum(dat$N > 0),
  total_grid_cells = nrow(dat),
  bin_hours = BIN_HOURS,
  inference = fit_settings,
  reported_parameters = params
)
jsonlite::write_json(
  manifest,
  file.path(output_dir, "earthquake_current_manifest.json"),
  pretty = TRUE,
  auto_unbox = TRUE
)

message("completed: ", summary_file)
