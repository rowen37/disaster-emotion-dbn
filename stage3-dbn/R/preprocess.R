# =====================================================================
# Stage 3 - DBN pipeline: raw post CSVs -> space-time grid RDS.
# Fixes two hard-coded values of the legacy preprocess.R:
#   (1) expand_grid(dist_bin = 1:20) -> expand_grid(1:K) per hazard
#       config (the old form created phantom rings 11..20 for the
#       50-km hazards, overflowing the Stan upper bound)
#   (2) pmin(..., 20) -> pmin(..., K)
# =====================================================================

parse_emo <- function(json_str) {
  if (is.na(json_str) || json_str == "") return(c(angry = 0, fear = 0, sad = 0))
  clean_str <- gsub('""', '"', json_str, fixed = TRUE)
  probs <- tryCatch(jsonlite::fromJSON(clean_str), error = function(e) NULL)
  if (is.null(probs)) return(c(angry = 0, fear = 0, sad = 0))
  c(
    angry = ifelse(is.null(probs$angry), 0, probs$angry),
    fear  = ifelse(is.null(probs$fear),  0, probs$fear),
    sad   = ifelse(is.null(probs$sad),   0, probs$sad)
  )
}

# One event -> space-time grid (self-contained: EMO_probs parsing happens inside)
grid_one_event <- function(df, cfg) {
  df <- df %>%
    dplyr::filter(time_gap >= TIME_MIN, time_gap < TIME_MAX)
  if (nrow(df) == 0) return(NULL)

  df <- df %>%
    dplyr::mutate(
      dlon = (lon - center_lon) * 111.32 * cos(pi / 180 * (lat + center_lat) / 2),
      dlat = (lat - center_lat) * 110.574,
      dist_km = sqrt(dlon^2 + dlat^2)
    ) %>%
    dplyr::filter(dist_km <= cfg$max_dist_km)
  if (nrow(df) == 0) return(NULL)

  df <- df %>%
    dplyr::mutate(
      angry = purrr::map_dbl(EMO_probs, ~ parse_emo(.x)["angry"]),
      fear  = purrr::map_dbl(EMO_probs, ~ parse_emo(.x)["fear"]),
      sad   = purrr::map_dbl(EMO_probs, ~ parse_emo(.x)["sad"]),
      dist_bin = pmin(floor(dist_km / cfg$ring_km) + 1, cfg$K),   # upper bound = K, no hard-coded 20
      time_bin = floor((time_gap - TIME_MIN) / BIN_HOURS) + 1
    )

  agg <- df %>%
    dplyr::group_by(dist_bin, time_bin) %>%
    dplyr::summarise(
      E_obs = mean(angry + fear + sad, na.rm = TRUE),
      B_obs = mean(raw_prob, na.rm = TRUE),
      N = dplyr::n(),
      .groups = "drop"
    )

  # complete only the K x T grid that exists for this hazard (phantom-ring fix)
  tidyr::expand_grid(dist_bin = 1:cfg$K, time_bin = 1:N_TIME_BINS) %>%
    dplyr::left_join(agg, by = c("dist_bin", "time_bin")) %>%
    dplyr::mutate(
      dist_center_km = (dist_bin - 0.5) * cfg$ring_km,
      time_center_hour = TIME_MIN + (time_bin - 0.5) * BIN_HOURS
    )
}

# All events of one hazard
preprocess_hazard <- function(hazard, input_dir = paths$raw_input, output_dir = paths$legacy_rds) {
  cfg <- hazard_configs[[hazard]]
  if (is.null(cfg)) stop("未知灾种: ", hazard)

  csv_files <- list.files(file.path(input_dir, hazard), pattern = "\\.csv$", full.names = TRUE)
  if (length(csv_files) == 0) stop("目录中没有CSV文件: ", file.path(input_dir, hazard))

  message(sprintf("[%s] 处理 %d 个事件...", hazard, length(csv_files)))
  all_events <- purrr::map(csv_files, function(file) {
    event_id <- tools::file_path_sans_ext(basename(file))
    df <- data.table::fread(file, showProgress = FALSE) %>%
      dplyr::select(time_gap, lon, lat, center_lon, center_lat, EMO_probs, raw_prob)
    out <- grid_one_event(df, cfg)
    if (is.null(out)) return(NULL)
    out$event_id <- event_id
    out
  })
  st_data <- dplyr::bind_rows(all_events)

  # attach lag_idx (always rebuilt here; legacy RDS versions were inconsistent)
  st_data <- attach_lag_idx(st_data)

  feat <- readr::read_csv(paths$event_feat, show_col_types = FALSE,
                          col_types = readr::cols(event_id = readr::col_character()))
  st_data <- st_data %>%
    dplyr::mutate(event_id = as.character(event_id)) %>%
    dplyr::left_join(feat, by = "event_id")

  out_file <- file.path(output_dir, paste0(hazard, "_st_data.rds"))
  saveRDS(st_data, out_file, compress = "gzip")
  message(sprintf("[%s] 已保存 %s（%d 行）", hazard, out_file, nrow(st_data)))
  invisible(st_data)
}

# Lag index: row of time_bin - 1 within the same event and ring; 0 for the first bin.
# Always rebuilt on load; never trust a lag_idx column shipped inside an RDS.
attach_lag_idx <- function(dat) {
  dat <- dat %>% dplyr::select(-dplyr::any_of(c("lag_idx", "row_id", "lag_row")))
  dat <- dat %>%
    dplyr::arrange(event_id, dist_bin, time_bin) %>%
    dplyr::group_by(event_id, dist_bin) %>%
    dplyr::mutate(row_id = dplyr::cur_group_rows()) %>%
    dplyr::ungroup()

  lag_map <- dat %>%
    dplyr::select(event_id, dist_bin, time_bin, lag_row = row_id) %>%
    dplyr::mutate(time_bin = time_bin + 1)

  dat %>%
    dplyr::left_join(lag_map, by = c("event_id", "dist_bin", "time_bin")) %>%
    dplyr::mutate(lag_idx = dplyr::if_else(is.na(lag_row), 0L, as.integer(lag_row))) %>%
    dplyr::select(-lag_row, -row_id)
}
