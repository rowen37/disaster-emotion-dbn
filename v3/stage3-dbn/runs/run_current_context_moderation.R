# Fit a joint population/GDP supplemental model for all four hazards.
#
# The current pointer remains unchanged.  This script adds only the requested
# context main effects and theta1/theta2 moderation to original_no_five.

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1) stop("run this file with Rscript")
script_file <- normalizePath(sub("^--file=", "", script_arg))
stage3_dir <- dirname(dirname(script_file))
source(file.path(stage3_dir, "R", "00_init.R"))

if (!identical(CURRENT_MODEL$id, "original_no_five")) {
  stop("models/current is ", CURRENT_MODEL$id, "; expected original_no_five")
}

repo_dir <- normalizePath(file.path(stage3_dir, "..", ".."))
pop_path <- file.path(repo_dir, "data", "chn_pd_2020_1km.tif")
gdp_path <- file.path(repo_dir, "data", "gdp2020.tif")
stan_path <- file.path(stage3_dir, "stan", "dbn_context_moderation.stan")
required <- c(pop_path, gdp_path, stan_path)
if (any(!file.exists(required))) {
  stop("missing required files: ", paste(required[!file.exists(required)], collapse = ", "))
}

output_dir <- file.path(stage3_dir, "output", "current", "four_hazard", "context_moderation")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

data_files <- c(
  earthquake = file.path(stage3_dir, "output", "current", "earthquake", "data", "earthquake_st_data.rds"),
  landslide = file.path(stage3_dir, "data", "landslide_st_data.rds"),
  debris_flow = file.path(stage3_dir, "data", "debris_flow_st_data.rds"),
  typhoon = file.path(stage3_dir, "output", "current", "typhoon", "data", "typhoon_st_data.rds")
)
if (any(!file.exists(data_files))) {
  stop("missing current grid data: ", paste(data_files[!file.exists(data_files)], collapse = ", "))
}

standardize_log_matrix <- function(x, label) {
  if (!is.matrix(x) || any(!is.finite(x))) stop(label, " matrix contains non-finite values")
  logged <- log1p(pmax(x, 0))
  s <- stats::sd(as.vector(logged))
  if (!is.finite(s) || s <= 1e-8) stop(label, " has no usable variation")
  z <- (logged - mean(logged)) / s
  matrix(as.numeric(z), nrow = nrow(x), ncol = ncol(x))
}

make_context_init <- function(stan_data) {
  function() {
    x <- list(
      alpha_e = -1.0, alpha_b = -2.0,
      beta_e = 0.02, beta_b = 0.01,
      theta1 = 0.05, theta2 = 0.0,
      psi_e = rep(0.1, stan_data$P), psi_b = rep(0.1, stan_data$P),
      lambda_e = rep(0.1, stan_data$M), lambda_b = rep(0.1, stan_data$M),
      z_w_e = rep(0, stan_data$J), z_w_b = rep(0, stan_data$J),
      tau_w_e = 0.1, tau_w_b = 0.1,
      sigma_obs_e = 0.8, sigma_proc_e = 0.8,
      phi_b_beta = 1.0,
      gamma_e_cov = rep(0, stan_data$Gc),
      gamma_b_cov = rep(0, stan_data$Gc),
      gamma_e_context = c(0, 0),
      gamma_b_context = c(0, 0),
      delta1_context = c(0, 0),
      delta2_context = c(0, 0),
      e_true = rep(-1.0, stan_data$N),
      b_true = rep(-2.0, stan_data$N)
    )
    if (stan_data$has_jump == 1) x$jump_raw <- c(0.1, 0.1)
    if (stan_data$has_group == 1) {
      x$z_v_e_raw <- rep(0, stan_data$G)
      x$z_v_b_raw <- rep(0, stan_data$G)
      x$tau_v_e_arr <- 1
      x$tau_v_b_arr <- 1
    }
    x
  }
}

summarize_draws <- function(draws, hazard) {
  vars <- c(
    "theta1", "theta2", "beta_e", "beta_b",
    "gamma_e_context[1]", "gamma_e_context[2]",
    "gamma_b_context[1]", "gamma_b_context[2]",
    "delta1_context[1]", "delta1_context[2]",
    "delta2_context[1]", "delta2_context[2]"
  )
  missing <- setdiff(vars, names(draws))
  if (length(missing)) stop(hazard, " draws missing: ", paste(missing, collapse = ", "))
  labels <- c(
    theta1 = "theta1", theta2 = "theta2", beta_e = "beta_e", beta_b = "beta_b",
    `gamma_e_context[1]` = "population_main_E",
    `gamma_e_context[2]` = "gdp_main_E",
    `gamma_b_context[1]` = "population_main_B",
    `gamma_b_context[2]` = "gdp_main_B",
    `delta1_context[1]` = "population_moderates_theta1",
    `delta1_context[2]` = "gdp_moderates_theta1",
    `delta2_context[1]` = "population_moderates_theta2",
    `delta2_context[2]` = "gdp_moderates_theta2"
  )
  dplyr::bind_rows(lapply(vars, function(v) {
    x <- draws[[v]]
    q <- stats::quantile(x, c(0.025, 0.5, 0.975), names = FALSE)
    data.frame(
      hazard = hazard,
      parameter = unname(labels[[v]]),
      stan_parameter = v,
      mean = mean(x), sd = stats::sd(x),
      Q2.5 = q[1], Q50 = q[2], Q97.5 = q[3],
      probability_positive = mean(x > 0),
      interval_excludes_zero = q[1] > 0 || q[3] < 0,
      stringsAsFactors = FALSE
    )
  }))
}

model <- cmdstanr::cmdstan_model(stan_path)
all_summaries <- list()
all_collinearity <- list()

for (hazard in names(data_files)) {
  cfg <- hazard_configs[[hazard]]
  dat <- readRDS(data_files[[hazard]])
  centers <- dat |>
    dplyr::transmute(
      event_id = as.character(event_id),
      center_lon = as.numeric(center_lon),
      center_lat = as.numeric(center_lat)
    ) |>
    dplyr::distinct() |>
    dplyr::arrange(event_id)
  if (nrow(centers) != dplyr::n_distinct(dat$event_id) ||
      any(!is.finite(centers$center_lon)) || any(!is.finite(centers$center_lat))) {
    stop(hazard, " does not have one finite center per event")
  }

  message("[", hazard, "] extracting population and GDP rings")
  pop_raw <- extract_ring_values(pop_path, centers, cfg)
  gdp_raw <- extract_ring_values(gdp_path, centers, cfg)
  pop_z <- standardize_log_matrix(pop_raw, paste(hazard, "population"))
  gdp_z <- standardize_log_matrix(gdp_raw, paste(hazard, "GDP"))

  context_long <- data.frame(
    hazard = hazard,
    event_id = rep(centers$event_id, each = cfg$K),
    dist_bin = rep(seq_len(cfg$K), times = nrow(centers)),
    population_raw = as.vector(t(pop_raw)),
    gdp_raw = as.vector(t(gdp_raw)),
    population_log_z = as.vector(t(pop_z)),
    gdp_log_z = as.vector(t(gdp_z)),
    stringsAsFactors = FALSE
  )
  utils::write.csv(
    context_long,
    file.path(output_dir, paste0(hazard, "_context_rings.csv")),
    row.names = FALSE
  )

  r <- stats::cor(as.vector(pop_z), as.vector(gdp_z))
  vif <- 1 / (1 - r^2)
  condition_number <- kappa(cbind(as.vector(pop_z), as.vector(gdp_z)))
  all_collinearity[[hazard]] <- data.frame(
    hazard = hazard,
    event_rings = length(pop_z),
    population_gdp_correlation = r,
    vif_population = vif,
    vif_gdp = vif,
    condition_number = condition_number
  )

  dummy <- matrix(0, nrow = nrow(centers), ncol = cfg$K)
  packed <- build_stan_data(
    dat, cfg, use_weights = 1, variant = "moderator", mod_ring = dummy
  )
  stan_data <- packed$stan_data
  stan_data$mod_ring <- NULL
  stan_data$has_slope <- NULL
  stan_data$dist_norm <- NULL
  stan_data$has_coupling <- NULL
  stan_data$pop_ring <- pop_z
  stan_data$gdp_ring <- gdp_z

  message("[", hazard, "] fitting context moderation model: N=", stan_data$N)
  fit <- model$variational(
    data = stan_data,
    seed = fit_settings$seed,
    algorithm = "meanfield",
    iter = fit_settings$iter,
    grad_samples = fit_settings$grad_samples,
    elbo_samples = fit_settings$elbo_samples,
    output_samples = fit_settings$output_samples,
    init = make_context_init(stan_data),
    eta = fit_settings$eta,
    tol_rel_obj = fit_settings$tol_rel_obj
  )
  draws <- fit$draws(format = "df")
  saveRDS(
    draws,
    file.path(output_dir, paste0(hazard, "_context_moderation_draws.rds")),
    compress = "gzip"
  )
  all_summaries[[hazard]] <- summarize_draws(draws, hazard)
}

summary_table <- dplyr::bind_rows(all_summaries)
collinearity_table <- dplyr::bind_rows(all_collinearity)
utils::write.csv(
  summary_table,
  file.path(output_dir, "context_moderation_summary.csv"),
  row.names = FALSE
)
utils::write.csv(
  collinearity_table,
  file.path(output_dir, "context_collinearity.csv"),
  row.names = FALSE
)

manifest <- list(
  analysis = "joint population/GDP main-effects and theta1/theta2 moderation",
  base_model = CURRENT_MODEL$id,
  current_pointer_changed = FALSE,
  stan_file = normalizePath(stan_path),
  rasters = list(
    population = normalizePath(pop_path),
    population_md5 = unname(tools::md5sum(pop_path)),
    gdp = normalizePath(gdp_path),
    gdp_md5 = unname(tools::md5sum(gdp_path))
  ),
  transformation = "ring mean, negative raster values treated as missing, log1p, then z-score within hazard over all event-ring cells",
  fitted_jointly = c("population", "GDP"),
  inference = c(fit_settings, list(algorithm = "meanfield")),
  outputs = c("context_moderation_summary.csv", "context_collinearity.csv")
)
jsonlite::write_json(
  manifest,
  file.path(output_dir, "manifest.json"),
  pretty = TRUE,
  auto_unbox = TRUE
)

message("completed: ", output_dir)
