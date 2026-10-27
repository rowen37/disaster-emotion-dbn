# Time- and space-marginal posterior predictive errors for the four current
# Stage 3 fits. Replicated cells are aggregated into E-time, B-time, E-space,
# and B-space curves before the error metrics are calculated.
# Unlike R/evaluate.R, this script uses the complete observation models:
#   E_obs(logit) ~ Normal(e_true, sigma_obs_e / sqrt(N))
#   B_obs        ~ Beta(logit^-1(b_true) * phi*N, ...)

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

hazard_cn <- c(
  earthquake = "地震", typhoon = "台风",
  landslide = "滑坡", debris_flow = "泥石流"
)

ordered_state_columns <- function(draws, state) {
  pattern <- paste0("^", state, "\\[[0-9]+\\]$")
  columns <- grep(pattern, names(draws), value = TRUE)
  indices <- as.integer(sub(paste0("^", state, "\\[([0-9]+)\\]$"), "\\1", columns))
  columns[order(indices)]
}

summarize_channel <- function(hazard, channel, dat, draws, state_columns,
                              chunk_size = 400L) {
  observed <- if (channel == "E") dat$E_obs else dat$B_obs
  observed_idx <- which(!is.na(observed))
  n_obs <- length(observed_idx)
  n_draws <- nrow(draws)
  if (n_obs == 0) stop(hazard, " ", channel, " has no observed cells")

  prediction <- numeric(n_obs)
  lower <- numeric(n_obs)
  upper <- numeric(n_obs)
  dimension_specs <- list(
    time = list(bin = dat$time_bin, center = dat$time_center_hour),
    space = list(bin = dat$dist_bin, center = dat$dist_center_km)
  )
  marginal_levels <- lapply(
    dimension_specs,
    function(x) sort(unique(x$bin[observed_idx]))
  )
  marginal_counts <- Map(
    function(spec, levels) {
      tabulate(match(spec$bin[observed_idx], levels), nbins = length(levels))
    },
    dimension_specs, marginal_levels
  )
  marginal_replicated_sums <- lapply(
    marginal_levels,
    function(levels) matrix(0, nrow = n_draws, ncol = length(levels))
  )

  chunks <- split(observed_idx, ceiling(seq_along(observed_idx) / chunk_size))
  output_pos <- 1L
  for (cell_idx in chunks) {
    columns <- state_columns[cell_idx]
      # posterior::draws_df warns whenever a column subset drops chain metadata;
      # the values remain valid, so suppress only that known class-drop warning.
      state <- suppressWarnings(as.matrix(draws[, columns, drop = FALSE]))
    counts <- pmax(as.numeric(dat$N[cell_idx]), 1)

    if (channel == "E") {
      sigma <- as.numeric(draws$sigma_obs_e)
      sd_matrix <- outer(sigma, 1 / sqrt(counts))
      replicated <- matrix(
        stats::rnorm(
          length(state), mean = as.vector(state), sd = as.vector(sd_matrix)
        ),
        nrow = n_draws, ncol = length(cell_idx)
      )
      replicated <- stats::plogis(replicated)
    } else {
      probability <- stats::plogis(state)
      precision <- outer(as.numeric(draws$phi_b_beta), counts)
      alpha <- pmax(probability * precision, 1e-8)
      beta <- pmax((1 - probability) * precision, 1e-8)
      replicated <- matrix(
        stats::rbeta(length(probability), as.vector(alpha), as.vector(beta)),
        nrow = n_draws, ncol = length(cell_idx)
      )
    }

    positions <- output_pos:(output_pos + length(cell_idx) - 1L)
    prediction[positions] <- colMeans(replicated)
    intervals <- apply(replicated, 2, stats::quantile, c(0.025, 0.975))
    lower[positions] <- intervals[1, ]
    upper[positions] <- intervals[2, ]

    for (dimension in names(dimension_specs)) {
      group_index <- match(
        dimension_specs[[dimension]]$bin[cell_idx],
        marginal_levels[[dimension]]
      )
      for (group in unique(group_index)) {
        selected <- which(group_index == group)
        marginal_replicated_sums[[dimension]][, group] <-
          marginal_replicated_sums[[dimension]][, group] +
          rowSums(replicated[, selected, drop = FALSE])
      }
    }
    output_pos <- output_pos + length(cell_idx)
    rm(state, replicated, intervals)
  }

  obs <- observed[observed_idx]
  residual <- obs - prediction
  cell_output <- data.frame(
    hazard = hazard,
    channel = channel,
    cell_index = observed_idx,
    event_id = dat$event_id[observed_idx],
    dist_bin = dat$dist_bin[observed_idx],
    time_bin = dat$time_bin[observed_idx],
    N = dat$N[observed_idx],
    observed = obs,
    predictive_mean = prediction,
    predictive_Q2.5 = lower,
    predictive_Q97.5 = upper,
    residual = residual,
    stringsAsFactors = FALSE
  )

  marginal_summaries <- list()
  marginal_details <- list()
  for (dimension in names(dimension_specs)) {
    levels <- marginal_levels[[dimension]]
    counts <- marginal_counts[[dimension]]
    replicated_marginal <- sweep(
      marginal_replicated_sums[[dimension]], 2, counts, "/"
    )
    observed_marginal <- vapply(
      levels,
      function(level) mean(
        observed[observed_idx][
          dimension_specs[[dimension]]$bin[observed_idx] == level
        ]
      ),
      numeric(1)
    )
    center_marginal <- vapply(
      levels,
      function(level) mean(
        dimension_specs[[dimension]]$center[observed_idx][
          dimension_specs[[dimension]]$bin[observed_idx] == level
        ],
        na.rm = TRUE
      ),
      numeric(1)
    )
    predictive_marginal <- colMeans(replicated_marginal)
    interval_marginal <- apply(
      replicated_marginal, 2, stats::quantile, c(0.025, 0.975)
    )
    residual_marginal <- observed_marginal - predictive_marginal
    observed_matrix <- matrix(
      observed_marginal,
      nrow = n_draws, ncol = length(levels), byrow = TRUE
    )
    residual_draws <- observed_matrix - replicated_marginal
    rmse_draw <- sqrt(rowMeans(residual_draws^2))
    mae_draw <- rowMeans(abs(residual_draws))
    curve <- paste(channel, dimension, sep = "-")

    marginal_summaries[[dimension]] <- data.frame(
      hazard = hazard,
      hazard_cn = unname(hazard_cn[hazard]),
      curve = curve,
      channel = channel,
      dimension = dimension,
      observed_bins = length(levels),
      observed_cells = n_obs,
      RMSE_predictive_mean = sqrt(mean(residual_marginal^2)),
      MAE_predictive_mean = mean(abs(residual_marginal)),
      mean_error_observed_minus_prediction = mean(residual_marginal),
      predictive_coverage_95 = mean(
        observed_marginal >= interval_marginal[1, ] &
          observed_marginal <= interval_marginal[2, ]
      ),
      mean_predictive_interval_width = mean(
        interval_marginal[2, ] - interval_marginal[1, ]
      ),
      replicated_RMSE_mean = mean(rmse_draw),
      replicated_RMSE_Q2.5 = unname(stats::quantile(rmse_draw, 0.025)),
      replicated_RMSE_Q97.5 = unname(stats::quantile(rmse_draw, 0.975)),
      replicated_MAE_mean = mean(mae_draw),
      replicated_MAE_Q2.5 = unname(stats::quantile(mae_draw, 0.025)),
      replicated_MAE_Q97.5 = unname(stats::quantile(mae_draw, 0.975)),
      posterior_draws = n_draws,
      stringsAsFactors = FALSE
    )
    marginal_details[[dimension]] <- data.frame(
      hazard = hazard,
      hazard_cn = unname(hazard_cn[hazard]),
      curve = curve,
      channel = channel,
      dimension = dimension,
      bin = levels,
      center = center_marginal,
      observed_cells = counts,
      observed_mean = observed_marginal,
      predictive_mean = predictive_marginal,
      predictive_Q2.5 = interval_marginal[1, ],
      predictive_Q97.5 = interval_marginal[2, ],
      residual = residual_marginal,
      stringsAsFactors = FALSE
    )
  }

  list(
    summary = do.call(rbind, marginal_summaries),
    marginals = do.call(rbind, marginal_details),
    cells = cell_output
  )
}

set.seed(123)
summaries <- list()
marginals <- list()
cells <- list()

for (hazard in names(draw_files)) {
  message("[", hazard, "] posterior predictive simulation")
  draws <- readRDS(draw_files[[hazard]])
  dat <- readRDS(data_files[[hazard]])
  e_columns <- ordered_state_columns(draws, "e_true")
  b_columns <- ordered_state_columns(draws, "b_true")
  if (length(e_columns) != nrow(dat) || length(b_columns) != nrow(dat)) {
    stop(
      hazard, " latent-state/data mismatch: E=", length(e_columns),
      ", B=", length(b_columns), ", data=", nrow(dat)
    )
  }

  e_result <- summarize_channel(hazard, "E", dat, draws, e_columns)
  b_result <- summarize_channel(hazard, "B", dat, draws, b_columns)
  summaries[[length(summaries) + 1L]] <- e_result$summary
  summaries[[length(summaries) + 1L]] <- b_result$summary
  marginals[[length(marginals) + 1L]] <- e_result$marginals
  marginals[[length(marginals) + 1L]] <- b_result$marginals
  cells[[length(cells) + 1L]] <- e_result$cells
  cells[[length(cells) + 1L]] <- b_result$cells

  rm(draws, dat, e_result, b_result)
  invisible(gc())
}

summary_df <- do.call(rbind, summaries)
marginal_df <- do.call(rbind, marginals)
cell_df <- do.call(rbind, cells)
curve_order <- c("E-time", "B-time", "E-space", "B-space")
hazard_order <- names(draw_files)
summary_df <- summary_df[
  order(match(summary_df$hazard, hazard_order), match(summary_df$curve, curve_order)),
]
marginal_df <- marginal_df[
  order(
    match(marginal_df$hazard, hazard_order),
    match(marginal_df$curve, curve_order),
    marginal_df$bin
  ),
]
rownames(summary_df) <- NULL
rownames(marginal_df) <- NULL
stopifnot(
  nrow(summary_df) == length(hazard_order) * length(curve_order),
  all(summary_df$curve == rep(curve_order, times = length(hazard_order)))
)
stopifnot(all(summary_df$RMSE_predictive_mean + 1e-12 >= summary_df$MAE_predictive_mean))

out_dir <- file.path(stage3_dir, "output", "current", "four_hazard")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
summary_file <- file.path(out_dir, "posterior_predictive_error.csv")
marginal_file <- file.path(out_dir, "posterior_predictive_marginals.csv")
cell_file <- file.path(out_dir, "posterior_predictive_cells.csv")
report_file <- file.path(out_dir, "posterior_predictive_error.txt")
utils::write.csv(summary_df, summary_file, row.names = FALSE)
utils::write.csv(marginal_df, marginal_file, row.names = FALSE)
utils::write.csv(cell_df, cell_file, row.names = FALSE)

report <- c(
  "Current Stage 3 四灾害后验预测误差",
  paste0("模型: ", CURRENT_MODEL$id),
  "四类曲线: E-time, B-time, E-space, B-space。",
  "每条曲线先在有观测的时空单元内按时间窗或距离环取等权均值，再跨窗/环计算误差。",
  "观测值与后验预测均值均在模型score尺度；这是样本内边际曲线误差，不是独立测试集误差。",
  "95%覆盖率来自聚合后的完整观测模型后验预测复制分布。",
  "",
  "灾害 | 曲线 | bins | cells | RMSE | MAE | 偏差(obs-pred) | 95%覆盖率 | 平均区间宽度 | 复制RMSE均值[95%区间]",
  vapply(seq_len(nrow(summary_df)), function(i) {
    z <- summary_df[i, ]
    sprintf(
      "%s | %s | %d | %d | %.6f | %.6f | %+.6f | %.1f%% | %.6f | %.6f [%.6f, %.6f]",
      z$hazard_cn, z$curve, z$observed_bins, z$observed_cells,
      z$RMSE_predictive_mean, z$MAE_predictive_mean,
      z$mean_error_observed_minus_prediction,
      100 * z$predictive_coverage_95,
      z$mean_predictive_interval_width,
      z$replicated_RMSE_mean, z$replicated_RMSE_Q2.5,
      z$replicated_RMSE_Q97.5
    )
  }, character(1))
)
writeLines(report, report_file)
cat(paste(report, collapse = "\n"), "\n")
message("saved: ", summary_file)
message("saved: ", marginal_file)
