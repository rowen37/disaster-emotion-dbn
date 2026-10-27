# =====================================================================
# Stage 3 - DBN pipeline: single source of truth for hazard configs,
# fitting settings and paths. No magic numbers elsewhere.
# =====================================================================

TIME_MIN   <- -72   # hours before the event
TIME_MAX   <- 168   # hours after the event
BIN_HOURS  <- 12    # time-bin width
N_TIME_BINS <- (TIME_MAX - TIME_MIN) %/% BIN_HOURS   # 20

# Locked Stage 2 -> Stage 3 interface for the current analysis. All four
# hazards must use the same uncalibrated model score. Switching to calibrated
# probabilities later must be an explicit, all-hazard configuration change;
# there is no automatic fallback or per-file choice.
BEHAVIOR_PROBABILITY_MODE <- "raw"
BEHAVIOR_PROBABILITY_COLUMN <- "behavior_prob"

# Population/GDP enter the main model after a within-between decomposition.
# The order is part of the Stan data contract and is written to the output
# manifest by the optimized landslide/debris-flow runner.
CONTEXT_COVARIATES <- c(
  "pop_between_z", "pop_within_z", "gdp_between_z", "gdp_within_z"
)

# Regularizing prior used for all standardized disaster-intensity and
# population/GDP coefficients.  This is deliberately tighter than the old
# N(0, 1) prior because the four contextual variables can still be correlated.
COVARIATE_PRIOR_SD <- 0.5

# Main-model observation weighting: 1 makes each cell's observation precision
# depend on its post count N, shared by the emotion and behavior channels.
USE_POST_COUNT_WEIGHTS <- 1L

# Index of the first post-event time bin: time_gap = 0 falls in bin
# (0 - TIME_MIN)/BIN_HOURS + 1. A hard-coded `t >= 8` would activate
# the jump one bin too late.
JUMP_START <- (0 - TIME_MIN) %/% BIN_HOURS + 1          # = 7, i.e. the [0, +12h) bin
stopifnot(JUMP_START == 7)

hazard_configs <- list(
  earthquake = list(
    label = "earthquake", K = 10, ring_km = 5, max_dist_km = 50,
    df_dist = 6, df_time = 8, has_jump = 1,
    ev_covs = c("mag", "depth")
  ),
  landslide = list(
    label = "landslide", K = 10, ring_km = 5, max_dist_km = 50,
    df_dist = 6, df_time = 8, has_jump = 0,
    ev_covs = c("rain72h_max")
  ),
  debris_flow = list(
    label = "debris_flow", K = 10, ring_km = 5, max_dist_km = 50,
    df_dist = 6, df_time = 8, has_jump = 0,
    ev_covs = c("imax_1h")
  ),
  typhoon = list(
    label = "typhoon", K = 20, ring_km = 10, max_dist_km = 200,
    df_dist = 5, df_time = 8, has_jump = 0,
    ev_covs = c("typhoon_level"),
    group_col = "typhoon_id"   # nesting: track points (events) nested in typhoons (groups)
  )
)

# Final fitting settings: tol_rel_obj = 0.01 (ELBO relative improvement
# < 1% plateau criterion), iteration cap 30000 (most fits had not reached
# the 0.01 plateau at 20k).
fit_settings <- list(
  iter          = 30000,
  grad_samples  = 5,
  elbo_samples  = 200,
  output_samples = 1000,
  eta           = 0.1,
  tol_rel_obj   = 0.01,
  seed          = 123
)

# Centralized path management. Stage 1 writes data/<hazard>.rds; Stage 2 reads
# that directory and writes scored copies to data_stage2/ for Stage 3.
repo_root <- normalizePath(file.path(stage3_root, "..", ".."), mustWork = FALSE)
# Population/GDP rasters stay in the local data archive.
resource_dir <- "/Users/rogerbqw/Downloads/毕设数据/Thesis-ZJU/DBNs"
requested_stage2_output <- file.path(
  repo_root, "v3", "stage2-behavior-extraction", "output"
)
# The supplied directory is currently named output.csv even though it is a
# directory.  Prefer the requested path if it is later renamed, otherwise use
# the existing directory explicitly rather than silently falling back to an
# older data_stage2 archive.
resolved_stage2_output <- if (dir.exists(requested_stage2_output)) {
  requested_stage2_output
} else {
  file.path(repo_root, "v3", "stage2-behavior-extraction", "output.csv")
}
paths <- list(
  stage1_output = file.path(repo_root, "data"),
  stage2_input  = file.path(repo_root, "data"),
  stage2_output = resolved_stage2_output,
  stage3_data   = file.path(stage3_root, "data"),
  # Compatibility alias used by the original Stage 3 fitter selected through
  # models/current.  It points to the same current v3 aggregate data.
  legacy_rds    = file.path(stage3_root, "data"),
  event_list    = file.path(repo_root, "data", "eventlist.csv"),
  event_feat  = file.path(repo_root, "data", "eventlist.csv"),
  pop_raster  = file.path(resource_dir, "chn_pd_2020_1km.tif"),
  gdp_raster  = file.path(resource_dir, "gdp2020.tif"),
  stan_core   = file.path(stage3_root, "stan", "dbn_core.stan"),
  stan_mod    = file.path(stage3_root, "stan", "dbn_moderator.stan"),
  out_dir     = file.path(stage3_root, "output"),
  optimized_out_dir = file.path(stage3_root, "output", "optimized"),
  original_out_dir = file.path(stage3_root, "output", "original"),
  current_model = file.path(stage3_root, "models", "current", "model.json"),
  fig_dir     = file.path(stage3_root, "output", "figures")
)

# Ring centers: seq(ring_km/2, max - ring_km/2, length.out = K)
# K=10, 5-km rings -> 2.5, 7.5, ..., 47.5; K=20, 10-km rings -> 5, 15, ..., 195
ring_centers <- function(cfg) {
  seq(cfg$ring_km / 2, cfg$max_dist_km - cfg$ring_km / 2, length.out = cfg$K)
}

time_bin_centers <- function() {
  TIME_MIN + (seq_len(N_TIME_BINS) - 0.5) * BIN_HOURS   # -66, -54, ..., 162
}
