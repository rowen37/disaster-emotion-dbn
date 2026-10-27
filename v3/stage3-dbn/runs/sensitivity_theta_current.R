# One-at-a-time post-fit sensitivity analysis for theta1 and theta2.
#
# The fitted posterior is kept fixed.  For each posterior draw, theta1/theta2
# are scaled and the behavior latent trajectory is updated recursively while
# preserving the fitted innovations and every other parameter:
#   delta_b[t] = beta_b * delta_b[t-1]
#                + (m1 - 1) * theta1 * e[t]
#                + (m2 - 1) * theta2 * e[t-1].
# Thus this is a local/post-fit parameter perturbation analysis, not a refit
# under a different prior or a causal intervention.

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1) stop("run this file with Rscript")
script_file <- normalizePath(sub("^--file=", "", script_arg))
stage3_dir <- dirname(dirname(script_file))
source(file.path(stage3_dir, "R", "00_init.R"))

if (!identical(CURRENT_MODEL$id, "original_no_five")) {
  stop("models/current is not original_no_five: ", CURRENT_MODEL$id)
}

draw_files <- c(
  earthquake = file.path(stage3_dir, "output", "current", "earthquake",
                         "earthquake_baseline_w1_draws.rds"),
  typhoon = file.path(stage3_dir, "output", "current", "typhoon",
                      "typhoon_baseline_w1_draws.rds"),
  landslide = file.path(stage3_dir, "output", "original",
                        "landslide_baseline_w1_draws.rds"),
  debris_flow = file.path(stage3_dir, "output", "original",
                          "debris_flow_baseline_w1_draws.rds")
)
data_files <- c(
  earthquake = file.path(stage3_dir, "output", "current", "earthquake", "data",
                         "earthquake_st_data.rds"),
  typhoon = file.path(stage3_dir, "output", "current", "typhoon", "data",
                      "typhoon_st_data.rds"),
  landslide = file.path(stage3_dir, "data", "landslide_st_data.rds"),
  debris_flow = file.path(stage3_dir, "data", "debris_flow_st_data.rds")
)
required <- c(draw_files, data_files)
if (any(!file.exists(required))) {
  stop("missing fit inputs: ", paste(required[!file.exists(required)], collapse = ", "))
}

baseline_error_file <- file.path(
  stage3_dir, "output", "current", "four_hazard", "posterior_predictive_error.csv"
)
if (!file.exists(baseline_error_file)) {
  stop("run evaluate_current_posterior_predictive.R first")
}
baseline_error <- utils::read.csv(baseline_error_file, check.names = FALSE)
required_curves <- c("E-space", "B-space", "E-time", "B-time")
if (!all(required_curves %in% baseline_error$curve)) {
  stop("posterior predictive error file does not contain all four required curves")
}

hazard_cn <- c(
  earthquake = "地震", typhoon = "台风",
  landslide = "滑坡", debris_flow = "泥石流"
)

# Baseline plus one-at-a-time deletion, weakening and strengthening.  The
# both-zero row is an ablation check rather than an OAT point.
scenarios <- data.frame(
  scenario = c(
    "baseline",
    "theta1_zero", "theta1_half", "theta1_plus50", "theta1_double",
    "theta2_zero", "theta2_half", "theta2_plus50", "theta2_double",
    "both_zero"
  ),
  target = c(
    "baseline",
    rep("theta1", 4),
    rep("theta2", 4),
    "both"
  ),
  theta1_multiplier = c(1, 0, 0.5, 1.5, 2, 1, 1, 1, 1, 0),
  theta2_multiplier = c(1, 1, 1, 1, 1, 0, 0.5, 1.5, 2, 0),
  stringsAsFactors = FALSE
)

ordered_state_columns <- function(draws, state) {
  pattern <- paste0("^", state, "\\[[0-9]+\\]$")
  columns <- grep(pattern, names(draws), value = TRUE)
  indices <- as.integer(sub(
    paste0("^", state, "\\[([0-9]+)\\]$"), "\\1", columns
  ))
  columns[order(indices)]
}

mean_by_level <- function(values, groups, levels) {
  vapply(levels, function(level) mean(values[groups == level]), numeric(1))
}

rmse <- function(observed, predicted) sqrt(mean((observed - predicted)^2))

parameter_rows <- list()
sensitivity_rows <- list()
prediction_rows <- list()

for (hazard in names(draw_files)) {
  message("[", hazard, "] theta sensitivity")
  draws <- readRDS(draw_files[[hazard]])
  dat <- readRDS(data_files[[hazard]])
  e_columns <- ordered_state_columns(draws, "e_true")
  b_columns <- ordered_state_columns(draws, "b_true")
  if (length(e_columns) != nrow(dat) || length(b_columns) != nrow(dat)) {
    stop(hazard, " latent-state/data mismatch")
  }

  theta1 <- as.numeric(draws$theta1)
  theta2 <- as.numeric(draws$theta2)
  beta_b <- as.numeric(draws$beta_b)
  n_draws <- length(theta1)

  for (parameter in c("theta1", "theta2")) {
    values <- if (parameter == "theta1") theta1 else theta2
    parameter_rows[[length(parameter_rows) + 1L]] <- data.frame(
      hazard = hazard,
      hazard_cn = unname(hazard_cn[hazard]),
      parameter = parameter,
      mean = mean(values),
      sd = stats::sd(values),
      Q2.5 = unname(stats::quantile(values, 0.025)),
      Q50 = unname(stats::quantile(values, 0.5)),
      Q97.5 = unname(stats::quantile(values, 0.975)),
      probability_positive = mean(values > 0),
      probability_negative = mean(values < 0),
      stringsAsFactors = FALSE
    )
  }

  observed_idx <- which(!is.na(dat$B_obs))
  time_levels <- sort(unique(dat$time_bin[observed_idx]))
  space_levels <- sort(unique(dat$dist_bin[observed_idx]))
  time_count <- tabulate(
    match(dat$time_bin[observed_idx], time_levels), nbins = length(time_levels)
  )
  space_count <- tabulate(
    match(dat$dist_bin[observed_idx], space_levels), nbins = length(space_levels)
  )
  observed_time <- mean_by_level(
    dat$B_obs[observed_idx], dat$time_bin[observed_idx], time_levels
  )
  observed_space <- mean_by_level(
    dat$B_obs[observed_idx], dat$dist_bin[observed_idx], space_levels
  )

  predicted_time_sum <- matrix(
    0, nrow = nrow(scenarios), ncol = length(time_levels)
  )
  predicted_space_sum <- matrix(
    0, nrow = nrow(scenarios), ncol = length(space_levels)
  )

  series <- split(
    seq_len(nrow(dat)),
    interaction(as.character(dat$event_id), dat$dist_bin, drop = TRUE)
  )
  for (indices in series) {
    indices <- indices[order(dat$time_bin[indices])]
    if (length(indices) != N_TIME_BINS ||
        !identical(as.integer(dat$time_bin[indices]), seq_len(N_TIME_BINS))) {
      stop(hazard, " has an incomplete event-ring time series")
    }
    expected_lag <- c(0L, head(indices, -1L))
    if (!identical(as.integer(dat$lag_idx[indices]), expected_lag)) {
      stop(hazard, " lag_idx does not match the event-ring time ordering")
    }

    e_state <- suppressWarnings(as.matrix(draws[, e_columns[indices], drop = FALSE]))
    b_state <- suppressWarnings(as.matrix(draws[, b_columns[indices], drop = FALSE]))

    for (scenario_index in seq_len(nrow(scenarios))) {
      delta_previous <- numeric(n_draws)
      delta_theta1 <-
        (scenarios$theta1_multiplier[scenario_index] - 1) * theta1
      delta_theta2 <-
        (scenarios$theta2_multiplier[scenario_index] - 1) * theta2

      for (position in seq_along(indices)) {
        delta_current <- beta_b * delta_previous +
          delta_theta1 * e_state[, position]
        if (position > 1L) {
          delta_current <- delta_current +
            delta_theta2 * e_state[, position - 1L]
        }

        cell <- indices[position]
        if (!is.na(dat$B_obs[cell])) {
          predictive_mean <- mean(stats::plogis(
            b_state[, position] + delta_current
          ))
          time_position <- match(dat$time_bin[cell], time_levels)
          space_position <- match(dat$dist_bin[cell], space_levels)
          predicted_time_sum[scenario_index, time_position] <-
            predicted_time_sum[scenario_index, time_position] + predictive_mean
          predicted_space_sum[scenario_index, space_position] <-
            predicted_space_sum[scenario_index, space_position] + predictive_mean
        }
        delta_previous <- delta_current
      }
    }
  }

  predicted_time <- sweep(predicted_time_sum, 2, time_count, "/")
  predicted_space <- sweep(predicted_space_sum, 2, space_count, "/")
  e_time_rmse <- baseline_error$RMSE_predictive_mean[
    baseline_error$hazard == hazard & baseline_error$curve == "E-time"
  ]
  e_space_rmse <- baseline_error$RMSE_predictive_mean[
    baseline_error$hazard == hazard & baseline_error$curve == "E-space"
  ]
  if (length(e_time_rmse) != 1L || length(e_space_rmse) != 1L) {
    stop(hazard, " missing baseline E RMSE")
  }

  b_time_rmse <- apply(
    predicted_time, 1, function(x) rmse(observed_time, x)
  )
  b_space_rmse <- apply(
    predicted_space, 1, function(x) rmse(observed_space, x)
  )
  baseline_index <- match("baseline", scenarios$scenario)

  one <- data.frame(
    hazard = hazard,
    hazard_cn = unname(hazard_cn[hazard]),
    model = CURRENT_MODEL$id,
    scenarios,
    theta1_posterior_mean_after_scaling =
      scenarios$theta1_multiplier * mean(theta1),
    theta2_posterior_mean_after_scaling =
      scenarios$theta2_multiplier * mean(theta2),
    `E-space` = rep(e_space_rmse, nrow(scenarios)),
    `B-space` = b_space_rmse,
    `E-time` = rep(e_time_rmse, nrow(scenarios)),
    `B-time` = b_time_rmse,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  one$delta_B_space_vs_baseline <- b_space_rmse - b_space_rmse[baseline_index]
  one$delta_B_time_vs_baseline <- b_time_rmse - b_time_rmse[baseline_index]
  one$percent_delta_B_space_vs_baseline <-
    100 * one$delta_B_space_vs_baseline / b_space_rmse[baseline_index]
  one$percent_delta_B_time_vs_baseline <-
    100 * one$delta_B_time_vs_baseline / b_time_rmse[baseline_index]
  sensitivity_rows[[hazard]] <- one

  for (scenario_index in seq_len(nrow(scenarios))) {
    prediction_rows[[length(prediction_rows) + 1L]] <- data.frame(
      hazard = hazard,
      scenario = scenarios$scenario[scenario_index],
      dimension = "space",
      bin = space_levels,
      center = vapply(
        space_levels,
        function(level) mean(dat$dist_center_km[
          observed_idx[dat$dist_bin[observed_idx] == level]
        ]),
        numeric(1)
      ),
      observed_B = observed_space,
      predicted_B = predicted_space[scenario_index, ],
      residual = observed_space - predicted_space[scenario_index, ],
      stringsAsFactors = FALSE
    )
    prediction_rows[[length(prediction_rows) + 1L]] <- data.frame(
      hazard = hazard,
      scenario = scenarios$scenario[scenario_index],
      dimension = "time",
      bin = time_levels,
      center = vapply(
        time_levels,
        function(level) mean(dat$time_center_hour[
          observed_idx[dat$time_bin[observed_idx] == level]
        ]),
        numeric(1)
      ),
      observed_B = observed_time,
      predicted_B = predicted_time[scenario_index, ],
      residual = observed_time - predicted_time[scenario_index, ],
      stringsAsFactors = FALSE
    )
  }

  rm(draws, dat, e_state, b_state)
  invisible(gc())
}

parameter_df <- do.call(rbind, parameter_rows)
sensitivity_df <- do.call(rbind, sensitivity_rows)
prediction_df <- do.call(rbind, prediction_rows)
rownames(parameter_df) <- NULL
rownames(sensitivity_df) <- NULL
rownames(prediction_df) <- NULL

stopifnot(
  nrow(parameter_df) == length(draw_files) * 2L,
  nrow(sensitivity_df) == length(draw_files) * nrow(scenarios),
  all(is.finite(as.matrix(sensitivity_df[, required_curves]))),
  all(sensitivity_df$`E-space` == rep(
    sensitivity_df$`E-space`[seq(1, nrow(sensitivity_df), by = nrow(scenarios))],
    each = nrow(scenarios)
  )),
  all(sensitivity_df$`E-time` == rep(
    sensitivity_df$`E-time`[seq(1, nrow(sensitivity_df), by = nrow(scenarios))],
    each = nrow(scenarios)
  ))
)

out_dir <- file.path(
  stage3_dir, "output", "current", "four_hazard", "theta_sensitivity"
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
summary_file <- file.path(out_dir, "theta_sensitivity_rmse.csv")
parameter_file <- file.path(out_dir, "theta_posterior_summary.csv")
prediction_file <- file.path(out_dir, "theta_sensitivity_B_curves.csv")
report_file <- file.path(out_dir, "theta_sensitivity_report.txt")
utils::write.csv(sensitivity_df, summary_file, row.names = FALSE)
utils::write.csv(parameter_df, parameter_file, row.names = FALSE)
utils::write.csv(prediction_df, prediction_file, row.names = FALSE)

report <- c(
  "Current Stage 3 theta1/theta2 后验扰动敏感性分析",
  paste0("模型: ", CURRENT_MODEL$id),
  "方法: 固定当前后验与创新项，分别将 theta1/theta2 缩放为 0、0.5、1.5、2 倍，并递归传播 beta_b 滞后效应。",
  "这不是重新拟合，也不是因果干预；E方程不含theta，因此E-space和E-time是结构性不变的负对照。",
  "RMSE列顺序: E-space, B-space, E-time, B-time。",
  "",
  "灾害 | 情景 | theta1倍数 | theta2倍数 | E-space | B-space | E-time | B-time | ΔB-space | ΔB-time",
  vapply(seq_len(nrow(sensitivity_df)), function(i) {
    z <- sensitivity_df[i, ]
    sprintf(
      "%s | %s | %.1f | %.1f | %.6f | %.6f | %.6f | %.6f | %+.6f | %+.6f",
      z$hazard_cn, z$scenario, z$theta1_multiplier, z$theta2_multiplier,
      z$`E-space`, z$`B-space`, z$`E-time`, z$`B-time`,
      z$delta_B_space_vs_baseline, z$delta_B_time_vs_baseline
    )
  }, character(1))
)
writeLines(report, report_file)
cat(paste(report, collapse = "\n"), "\n")
message("saved: ", summary_file)

