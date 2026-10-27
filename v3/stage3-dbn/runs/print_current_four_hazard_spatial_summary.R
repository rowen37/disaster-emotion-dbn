# Print spatial peaks and curve profiles for the four fits in the current run.

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
if (any(!file.exists(draw_files))) {
  stop("missing posterior draws: ", paste(draw_files[!file.exists(draw_files)], collapse = ", "))
}

hazard_cn <- c(
  earthquake = "地震", typhoon = "台风",
  landslide = "滑坡", debris_flow = "泥石流"
)

shape_description <- function(y, x) {
  delta <- diff(y)
  signs <- sign(delta)
  signs <- signs[signs != 0]
  if (length(signs) == 0) {
    return(list(shape = "平坦", directions = "平坦", turns = "无"))
  }
  runs <- signs[c(TRUE, diff(signs) != 0)]
  direction_labels <- ifelse(runs > 0, "升", "降")
  directions <- paste(direction_labels, collapse = "-")

  shape <- if (length(runs) == 1 && runs[1] > 0) {
    "单调上升"
  } else if (length(runs) == 1) {
    "单调下降"
  } else if (length(runs) == 2 && identical(runs, c(-1, 1))) {
    "U形（先降后升）"
  } else if (length(runs) == 2 && identical(runs, c(1, -1))) {
    "倒U形（先升后降）"
  } else {
    paste0("多拐点（", directions, "）")
  }

  raw_signs <- sign(delta)
  nonzero_idx <- which(raw_signs != 0)
  turn_idx <- integer()
  if (length(nonzero_idx) >= 2) {
    nz_signs <- raw_signs[nonzero_idx]
    changes <- which(diff(nz_signs) != 0)
    if (length(changes) > 0) turn_idx <- nonzero_idx[changes] + 1L
  }
  turns <- if (length(turn_idx) == 0) "无" else {
    paste0(format(x[turn_idx], trim = TRUE), " km", collapse = "; ")
  }
  list(shape = shape, directions = directions, turns = turns)
}

summaries <- list()
profiles <- list()
peak_distributions <- list()
report_lines <- c(
  "Current Stage 3 四灾害空间峰值与曲线报告",
  paste0("模型: ", CURRENT_MODEL$id),
  "峰值抽样占比 = 后验抽样中峰值位于后验均值曲线峰值距离环的比例。",
  "score 为潜在 logit 空间曲线经 inverse-logit 后的模型得分尺度，不是真实概率。",
  ""
)

for (hazard in names(draw_files)) {
  message("[", hazard, "] reading draws and calculating spatial curves")
  draws <- readRDS(draw_files[[hazard]])
  curve <- spatial_decay(draws, hazard_configs[[hazard]])
  x <- curve$dist_km
  channel_objects <- list(E = curve$logit_e, B = curve$logit_b)
  channel_results <- list()

  for (channel in names(channel_objects)) {
    logit_mat <- channel_objects[[channel]]
    score_mat <- stats::plogis(logit_mat)
    mean_logit <- colMeans(logit_mat)
    mean_score <- colMeans(score_mat)
    q_score <- apply(score_mat, 2, stats::quantile, c(0.025, 0.975))
    mean_peak_idx <- which.max(mean_score)
    draw_peak_idx <- max.col(score_mat, ties.method = "first")
    peak_counts <- tabulate(draw_peak_idx, nbins = length(x))
    peak_shares <- peak_counts / length(draw_peak_idx)
    modal_peak_idx <- which.max(peak_counts)
    shape <- shape_description(mean_score, x)

    channel_results[[channel]] <- list(
      mean_score = mean_score,
      q_score = q_score,
      mean_logit = mean_logit
    )
    summaries[[length(summaries) + 1L]] <- data.frame(
      hazard = hazard,
      hazard_cn = unname(hazard_cn[hazard]),
      channel = channel,
      mean_curve_peak_km = x[mean_peak_idx],
      peak_score_mean = mean_score[mean_peak_idx],
      peak_score_Q2.5 = q_score[1, mean_peak_idx],
      peak_score_Q97.5 = q_score[2, mean_peak_idx],
      draw_peak_share_at_mean_peak = peak_shares[mean_peak_idx],
      modal_draw_peak_km = x[modal_peak_idx],
      modal_draw_peak_share = peak_shares[modal_peak_idx],
      score_min = min(mean_score),
      score_max = max(mean_score),
      score_range = diff(range(mean_score)),
      logit_min = min(mean_logit),
      logit_max = max(mean_logit),
      logit_range = diff(range(mean_logit)),
      endpoint_change = mean_score[length(mean_score)] - mean_score[1],
      shape = shape$shape,
      direction_sequence = shape$directions,
      turning_points = shape$turns,
      stringsAsFactors = FALSE
    )
    peak_distributions[[length(peak_distributions) + 1L]] <- data.frame(
      hazard = hazard,
      hazard_cn = unname(hazard_cn[hazard]),
      channel = channel,
      distance_km = x,
      peak_draw_count = peak_counts,
      peak_draw_share = peak_shares,
      stringsAsFactors = FALSE
    )
  }

  profiles[[length(profiles) + 1L]] <- data.frame(
    hazard = hazard,
    hazard_cn = unname(hazard_cn[hazard]),
    distance_km = x,
    E_mean = channel_results$E$mean_score,
    E_Q2.5 = channel_results$E$q_score[1, ],
    E_Q97.5 = channel_results$E$q_score[2, ],
    B_mean = channel_results$B$mean_score,
    B_Q2.5 = channel_results$B$q_score[1, ],
    B_Q97.5 = channel_results$B$q_score[2, ],
    stringsAsFactors = FALSE
  )

  one_summary <- do.call(rbind, tail(summaries, 2))
  one_profile <- tail(profiles, 1)[[1]]
  report_lines <- c(
    report_lines,
    paste0("=== ", hazard_cn[hazard], " (", hazard, ") ==="),
    vapply(seq_len(nrow(one_summary)), function(i) {
      z <- one_summary[i, ]
      sprintf(
        "%s: 峰值 %.1f km, 峰值score %.6f [%.6f, %.6f], 峰值抽样占比 %.1f%%; 范围 %.6f-%.6f (振幅 %.6f); %s; 拐点 %s",
        z$channel, z$mean_curve_peak_km, z$peak_score_mean,
        z$peak_score_Q2.5, z$peak_score_Q97.5,
        100 * z$draw_peak_share_at_mean_peak,
        z$score_min, z$score_max, z$score_range,
        z$shape, z$turning_points
      )
    }, character(1)),
    "距离(km) | E均值 | B均值",
    apply(one_profile[, c("distance_km", "E_mean", "B_mean")], 1, function(z) {
      sprintf("%8.1f | %.6f | %.6f", z[1], z[2], z[3])
    }),
    ""
  )

  rm(draws, curve, channel_objects, channel_results)
  invisible(gc())
}

summary_df <- do.call(rbind, summaries)
profile_df <- do.call(rbind, profiles)
peak_distribution_df <- do.call(rbind, peak_distributions)
output_dir <- file.path(stage3_dir, "output", "current", "four_hazard_spatial")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

utils::write.csv(
  summary_df,
  file.path(output_dir, "four_hazard_spatial_peak_summary.csv"),
  row.names = FALSE
)
utils::write.csv(
  profile_df,
  file.path(output_dir, "four_hazard_spatial_curve_profiles.csv"),
  row.names = FALSE
)
utils::write.csv(
  peak_distribution_df,
  file.path(output_dir, "four_hazard_peak_draw_distribution.csv"),
  row.names = FALSE
)
report_file <- file.path(output_dir, "four_hazard_spatial_report.txt")
writeLines(report_lines, report_file)
cat(paste(report_lines, collapse = "\n"), "\n")
message("saved: ", report_file)
