# Generate 16 compact current-model figures: E/B temporal and E/B spatial
# curves for each of four hazards. Each panel gets its own adaptive y-axis.

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1) stop("run this file with Rscript")
script_file <- normalizePath(sub("^--file=", "", script_arg))
stage3_dir <- dirname(dirname(script_file))

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
})

source_dir <- file.path(stage3_dir, "output", "current", "four_hazard")
temporal_profile_file <- file.path(source_dir, "four_hazard_temporal_curve_profiles.csv")
temporal_peak_file <- file.path(source_dir, "four_hazard_temporal_peak_summary.csv")
spatial_profile_file <- file.path(source_dir, "four_hazard_spatial_curve_profiles.csv")
spatial_peak_file <- file.path(source_dir, "four_hazard_spatial_peak_summary.csv")
required <- c(
  temporal_profile_file, temporal_peak_file,
  spatial_profile_file, spatial_peak_file
)
if (any(!file.exists(required))) {
  stop("missing curve inputs: ", paste(required[!file.exists(required)], collapse = ", "))
}

temporal_profile <- readr::read_csv(temporal_profile_file, show_col_types = FALSE)
temporal_peak <- readr::read_csv(temporal_peak_file, show_col_types = FALSE)
spatial_profile <- readr::read_csv(spatial_profile_file, show_col_types = FALSE)
spatial_peak <- readr::read_csv(spatial_peak_file, show_col_types = FALSE)

out_dir <- file.path(stage3_dir, "output", "current", "four_hazard", "figs")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

hazards <- c("earthquake", "typhoon", "landslide", "debris_flow")
short <- c(earthquake = "eq", typhoon = "ty", landslide = "ls", debris_flow = "df")
titles <- c(
  earthquake = "Earthquake", typhoon = "Typhoon",
  landslide = "Landslide", debris_flow = "Debris flow"
)
channel_names <- c(E = "Emotion score", B = "Behavior score")
channel_colors <- c(E = "#C74343", B = "#2869A8")

adaptive_limits <- function(low, high) {
  limits <- range(c(low, high), finite = TRUE)
  span <- diff(limits)
  pad <- max(span * 0.10, 0.0025)
  c(max(0, limits[1] - pad), min(1, limits[2] + pad))
}

base_theme <- theme_classic(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 14, margin = margin(b = 4)),
    plot.subtitle = element_text(size = 10.5, color = "grey35", margin = margin(b = 8)),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10.5, color = "grey15"),
    plot.margin = margin(12, 18, 12, 12)
  )

make_one <- function(hazard, channel, domain = c("temporal", "spatial")) {
  domain <- match.arg(domain)
  color <- unname(channel_colors[channel])

  if (domain == "temporal") {
    dat <- temporal_profile[temporal_profile$hazard == hazard, ]
    peak <- temporal_peak[
      temporal_peak$hazard == hazard & temporal_peak$channel == channel, ]
    x <- dat$time_center_hour
    mean_y <- dat[[paste0(channel, "_mean")]]
    low_y <- dat[[paste0(channel, "_Q2.5")]]
    high_y <- dat[[paste0(channel, "_Q97.5")]]
    plot_dat <- data.frame(x = x, mean = mean_y, low = low_y, high = high_y)
    y_limits <- adaptive_limits(low_y, high_y)
    peak_x <- peak$peak_time_center_hour
    peak_y <- peak$peak_score_mean
    peak_label <- sprintf(
      "Peak %+g h  |  %.3f  |  %.1f%% draws",
      peak_x, peak_y, 100 * peak$draw_peak_share_at_mean_peak
    )
    file <- sprintf("%s_t_%s.png", short[[hazard]], tolower(channel))
    x_label <- "Time relative to event (hours)"
    subtitle <- sprintf(
      "%s; peak window [%+g, %+g) h",
      channel_names[[channel]], peak$peak_window_start_hour,
      peak$peak_window_end_hour
    )
    breaks <- sort(unique(c(seq(-72, 168, by = 48), 0)))
    p <- ggplot(plot_dat, aes(x = x, y = mean)) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "grey55", linewidth = 0.45) +
      geom_ribbon(aes(ymin = low, ymax = high), fill = color, alpha = 0.16) +
      geom_line(color = color, linewidth = 1.05) +
      geom_point(color = color, size = 1.8) +
      geom_point(data = data.frame(x = peak_x, mean = peak_y),
                 aes(x = x, y = mean), inherit.aes = FALSE,
                 shape = 21, size = 3.4, stroke = 0.9,
                 color = color, fill = "white") +
      annotate(
        "label", x = min(x) + 0.03 * diff(range(x)),
        y = y_limits[2] - 0.04 * diff(y_limits), label = peak_label,
        hjust = 0, vjust = 1, size = 3.5,
        linewidth = 0.2, label.padding = unit(0.18, "lines"),
        color = "grey10", fill = "white"
      ) +
      scale_x_continuous(breaks = breaks, limits = range(x))
  } else {
    dat <- spatial_profile[spatial_profile$hazard == hazard, ]
    peak <- spatial_peak[
      spatial_peak$hazard == hazard & spatial_peak$channel == channel, ]
    x <- dat$distance_km
    mean_y <- dat[[paste0(channel, "_mean")]]
    low_y <- dat[[paste0(channel, "_Q2.5")]]
    high_y <- dat[[paste0(channel, "_Q97.5")]]
    plot_dat <- data.frame(x = x, mean = mean_y, low = low_y, high = high_y)
    y_limits <- adaptive_limits(low_y, high_y)
    peak_x <- peak$mean_curve_peak_km
    peak_y <- peak$peak_score_mean
    peak_label <- sprintf(
      "Peak %.1f km  |  %.3f  |  %.1f%% draws",
      peak_x, peak_y, 100 * peak$draw_peak_share_at_mean_peak
    )
    file <- sprintf("%s_s_%s.png", short[[hazard]], tolower(channel))
    x_label <- "Distance from event center (km)"
    subtitle <- paste0(channel_names[[channel]], "; posterior mean and 95% interval")
    p <- ggplot(plot_dat, aes(x = x, y = mean)) +
      geom_ribbon(aes(ymin = low, ymax = high), fill = color, alpha = 0.16) +
      geom_line(color = color, linewidth = 1.05) +
      geom_point(color = color, size = 1.8) +
      geom_point(data = data.frame(x = peak_x, mean = peak_y),
                 aes(x = x, y = mean), inherit.aes = FALSE,
                 shape = 21, size = 3.4, stroke = 0.9,
                 color = color, fill = "white") +
      annotate(
        "label", x = min(x) + 0.03 * diff(range(x)),
        y = y_limits[2] - 0.04 * diff(y_limits), label = peak_label,
        hjust = 0, vjust = 1, size = 3.5,
        linewidth = 0.2, label.padding = unit(0.18, "lines"),
        color = "grey10", fill = "white"
      ) +
      scale_x_continuous(breaks = pretty(range(x), n = 6), limits = range(x))
  }

  p <- p +
    coord_cartesian(ylim = y_limits, expand = FALSE) +
    scale_y_continuous(labels = scales::label_number(accuracy = 0.001)) +
    labs(
      title = paste0(titles[[hazard]], " — ", channel_names[[channel]]),
      subtitle = subtitle,
      x = x_label,
      y = "Model score"
    ) +
    base_theme

  target <- file.path(out_dir, file)
  ggsave(target, p, width = 7.4, height = 4.8, dpi = 300, bg = "white")
  data.frame(
    file = file,
    hazard = hazard,
    domain = domain,
    channel = channel,
    y_min = y_limits[1],
    y_max = y_limits[2],
    stringsAsFactors = FALSE
  )
}

manifest <- list()
for (hazard in hazards) {
  for (channel in c("E", "B")) {
    manifest[[length(manifest) + 1L]] <- make_one(hazard, channel, "temporal")
    manifest[[length(manifest) + 1L]] <- make_one(hazard, channel, "spatial")
  }
}
manifest <- do.call(rbind, manifest)
write.csv(manifest, file.path(out_dir, "manifest.csv"), row.names = FALSE)

files <- sort(list.files(out_dir, pattern = "\\.png$"))
if (length(files) != 16L) stop("expected 16 PNG figures, found ", length(files))
cat(paste(files, collapse = "\n"), "\n")
