# =====================================================================
# Stage 3 - DBN pipeline: single source of truth for hazard configs,
# fitting settings and paths. No magic numbers elsewhere.
# =====================================================================

TIME_MIN   <- -72   # hours before the event
TIME_MAX   <- 168   # hours after the event
BIN_HOURS  <- 12    # time-bin width
N_TIME_BINS <- (TIME_MAX - TIME_MIN) %/% BIN_HOURS   # 20

# Index of the first post-event time bin: time_gap = 0 falls in bin
# (0 - TIME_MIN)/BIN_HOURS + 1. Fixes the off-by-one of the legacy
# `t >= 8`, which activated the jump one bin too late.
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
    ev_covs = c("volume", "max_rain")
  ),
  debris_flow = list(
    label = "debris_flow", K = 10, ring_km = 5, max_dist_km = 50,
    df_dist = 6, df_time = 8, has_jump = 0,
    ev_covs = c("volume", "max_rain")
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

# Centralized path management
# 原始微博与外部资源位于本机归档目录；模型输出一律写入本目录
resource_dir <- "/Users/rogerbqw/Downloads/毕设数据/Thesis-ZJU/DBNs"
paths <- list(
  raw_input   = "/Users/rogerbqw/Downloads/毕设数据/Thesis-ZJU/weibo_dbn/input",
  legacy_rds  = file.path(stage3_root, "data"),
  event_feat  = file.path(resource_dir, "event_features.csv"),
  pop_raster  = file.path(resource_dir, "chn_pd_2020_1km.tif"),
  gdp_raster  = file.path(resource_dir, "gdp2020.tif"),
  stan_core   = file.path(stage3_root, "stan", "dbn_core.stan"),
  stan_mod    = file.path(stage3_root, "stan", "dbn_moderator.stan"),
  out_dir     = file.path(stage3_root, "output"),
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
