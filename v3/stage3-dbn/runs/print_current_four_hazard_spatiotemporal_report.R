# Combine temporal peaks/profiles with the existing spatial report for all
# four hazards fitted by the current Stage 3 model.

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1) stop("run this file with Rscript")
script_file <- normalizePath(sub("^--file=", "", script_arg))
stage3_dir <- dirname(dirname(script_file))
source(file.path(stage3_dir, "R", "00_init.R"))

if (!identical(CURRENT_MODEL$id, "original_no_five")) {
  stop("models/current is not original_no_five: ", CURRENT_MODEL$id)
}

output_dir <- file.path(stage3_dir, "output", "current", "four_hazard_spatial")
spatial_summary_file <- file.path(output_dir, "four_hazard_spatial_peak_summary.csv")
spatial_profile_file <- file.path(output_dir, "four_hazard_spatial_curve_profiles.csv")
if (!file.exists(spatial_summary_file) || !file.exists(spatial_profile_file)) {
  stop("run print_current_four_hazard_spatial_summary.R first")
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

shape_description <- function(y, x, unit) {
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
    changes <- which(diff(raw_signs[nonzero_idx]) != 0)
    if (length(changes) > 0) turn_idx <- nonzero_idx[changes] + 1L
  }
  turns <- if (length(turn_idx) == 0) "无" else {
    paste0(sprintf("%+g", x[turn_idx]), " ", unit, collapse = "; ")
  }
  list(shape = shape, directions = directions, turns = turns)
}

temporal_summaries <- list()
temporal_profiles <- list()

for (hazard in names(draw_files)) {
  message("[", hazard, "] calculating temporal peaks")
  draws <- readRDS(draw_files[[hazard]])
  curve <- temporal_trajectory(draws, hazard_configs[[hazard]])
  x <- curve$time_hour
  starts <- TIME_MIN + (seq_along(x) - 1L) * BIN_HOURS
  ends <- starts + BIN_HOURS
  channel_objects <- list(E = curve$logit_e, B = curve$logit_b)
  channel_results <- list()

  for (channel in names(channel_objects)) {
    logit_mat <- channel_objects[[channel]]
    score_mat <- stats::plogis(logit_mat)
    mean_score <- colMeans(score_mat)
    mean_logit <- colMeans(logit_mat)
    q_score <- apply(score_mat, 2, stats::quantile, c(0.025, 0.975))
    mean_peak_idx <- which.max(mean_score)
    draw_peak_idx <- max.col(score_mat, ties.method = "first")
    peak_counts <- tabulate(draw_peak_idx, nbins = length(x))
    peak_shares <- peak_counts / length(draw_peak_idx)
    modal_peak_idx <- which.max(peak_counts)
    shape <- shape_description(mean_score, x, "h")

    channel_results[[channel]] <- list(mean = mean_score, q = q_score)
    temporal_summaries[[length(temporal_summaries) + 1L]] <- data.frame(
      hazard = hazard,
      hazard_cn = unname(hazard_cn[hazard]),
      channel = channel,
      peak_time_center_hour = x[mean_peak_idx],
      peak_window_start_hour = starts[mean_peak_idx],
      peak_window_end_hour = ends[mean_peak_idx],
      peak_score_mean = mean_score[mean_peak_idx],
      peak_score_Q2.5 = q_score[1, mean_peak_idx],
      peak_score_Q97.5 = q_score[2, mean_peak_idx],
      draw_peak_share_at_mean_peak = peak_shares[mean_peak_idx],
      modal_draw_peak_time_center_hour = x[modal_peak_idx],
      modal_draw_peak_share = peak_shares[modal_peak_idx],
      score_min = min(mean_score),
      score_max = max(mean_score),
      score_range = diff(range(mean_score)),
      endpoint_change = mean_score[length(mean_score)] - mean_score[1],
      shape = shape$shape,
      direction_sequence = shape$directions,
      turning_points = shape$turns,
      stringsAsFactors = FALSE
    )
  }

  temporal_profiles[[length(temporal_profiles) + 1L]] <- data.frame(
    hazard = hazard,
    hazard_cn = unname(hazard_cn[hazard]),
    time_bin = seq_along(x),
    time_center_hour = x,
    window_start_hour = starts,
    window_end_hour = ends,
    E_mean = channel_results$E$mean,
    E_Q2.5 = channel_results$E$q[1, ],
    E_Q97.5 = channel_results$E$q[2, ],
    B_mean = channel_results$B$mean,
    B_Q2.5 = channel_results$B$q[1, ],
    B_Q97.5 = channel_results$B$q[2, ],
    stringsAsFactors = FALSE
  )

  rm(draws, curve, channel_objects, channel_results)
  invisible(gc())
}

temporal_summary <- do.call(rbind, temporal_summaries)
temporal_profile <- do.call(rbind, temporal_profiles)
spatial_summary <- utils::read.csv(spatial_summary_file, check.names = FALSE)
spatial_profile <- utils::read.csv(spatial_profile_file, check.names = FALSE)

temporal_summary_file <- file.path(output_dir, "four_hazard_temporal_peak_summary.csv")
temporal_profile_file <- file.path(output_dir, "four_hazard_temporal_curve_profiles.csv")
utils::write.csv(temporal_summary, temporal_summary_file, row.names = FALSE)
utils::write.csv(temporal_profile, temporal_profile_file, row.names = FALSE)

md <- c(
  "# Current Stage 3 四灾害时间—空间峰值与曲线报告",
  "",
  paste0("- 模型：`", CURRENT_MODEL$id, "`"),
  paste0("- 时间分辨率：", BIN_HOURS, " 小时；时间为相对灾害发生时刻。"),
  "- 峰值抽样占比：后验抽样中峰值位于后验均值曲线峰值时间窗/距离环的比例。",
  "- score：潜在曲线经 inverse-logit 转换的模型得分尺度，不是真实概率。",
  ""
)

for (hazard in names(draw_files)) {
  ts <- temporal_summary[temporal_summary$hazard == hazard, ]
  tp <- temporal_profile[temporal_profile$hazard == hazard, ]
  ss <- spatial_summary[spatial_summary$hazard == hazard, ]
  sp <- spatial_profile[spatial_profile$hazard == hazard, ]

  md <- c(
    md,
    paste0("## ", hazard_cn[hazard], "（", hazard, "）"),
    "",
    "### 时间峰值",
    "",
    "| 曲线 | 峰值时间中心 | 对应12小时窗 | 峰值score [95%区间] | 峰值抽样占比 | 变化范围/振幅 | 形状 |",
    "|---|---:|---:|---:|---:|---:|---|",
    vapply(seq_len(nrow(ts)), function(i) {
      z <- ts[i, ]
      sprintf(
        "| %s | %+g h | [%+g, %+g) h | %.6f [%.6f, %.6f] | %.1f%% | %.6f–%.6f / %.6f | %s |",
        z$channel, z$peak_time_center_hour,
        z$peak_window_start_hour, z$peak_window_end_hour,
        z$peak_score_mean, z$peak_score_Q2.5, z$peak_score_Q97.5,
        100 * z$draw_peak_share_at_mean_peak,
        z$score_min, z$score_max, z$score_range, z$shape
      )
    }, character(1)),
    "",
    "### 逐时间窗曲线",
    "",
    "| 时间中心 | 12小时窗 | E均值 | E 95%区间 | B均值 | B 95%区间 |",
    "|---:|---:|---:|---:|---:|---:|",
    apply(tp, 1, function(z) {
      sprintf(
        "| %+g h | [%+g, %+g) | %.6f | [%.6f, %.6f] | %.6f | [%.6f, %.6f] |",
        as.numeric(z[["time_center_hour"]]), as.numeric(z[["window_start_hour"]]),
        as.numeric(z[["window_end_hour"]]), as.numeric(z[["E_mean"]]),
        as.numeric(z[["E_Q2.5"]]), as.numeric(z[["E_Q97.5"]]),
        as.numeric(z[["B_mean"]]), as.numeric(z[["B_Q2.5"]]),
        as.numeric(z[["B_Q97.5"]])
      )
    }),
    "",
    "### 空间峰值",
    "",
    "| 曲线 | 峰值距离 | 峰值score [95%区间] | 峰值抽样占比 | 变化范围/振幅 | 形状 |",
    "|---|---:|---:|---:|---:|---|",
    vapply(seq_len(nrow(ss)), function(i) {
      z <- ss[i, ]
      sprintf(
        "| %s | %.1f km | %.6f [%.6f, %.6f] | %.1f%% | %.6f–%.6f / %.6f | %s |",
        z$channel, z$mean_curve_peak_km,
        z$peak_score_mean, z$peak_score_Q2.5, z$peak_score_Q97.5,
        100 * z$draw_peak_share_at_mean_peak,
        z$score_min, z$score_max, z$score_range, z$shape
      )
    }, character(1)),
    "",
    "### 逐距离环曲线",
    "",
    "| 距离 | E均值 | E 95%区间 | B均值 | B 95%区间 |",
    "|---:|---:|---:|---:|---:|",
    apply(sp, 1, function(z) {
      sprintf(
        "| %.1f km | %.6f | [%.6f, %.6f] | %.6f | [%.6f, %.6f] |",
        as.numeric(z[["distance_km"]]), as.numeric(z[["E_mean"]]),
        as.numeric(z[["E_Q2.5"]]), as.numeric(z[["E_Q97.5"]]),
        as.numeric(z[["B_mean"]]), as.numeric(z[["B_Q2.5"]]),
        as.numeric(z[["B_Q97.5"]])
      )
    }),
    ""
  )
}

report_file <- file.path(output_dir, "four_hazard_spatiotemporal_report.md")
writeLines(md, report_file)

print_table <- temporal_summary[, c(
  "hazard_cn", "channel", "peak_time_center_hour",
  "peak_window_start_hour", "peak_window_end_hour", "peak_score_mean",
  "peak_score_Q2.5", "peak_score_Q97.5", "draw_peak_share_at_mean_peak"
)]
print(print_table, row.names = FALSE)
message("saved: ", report_file)
