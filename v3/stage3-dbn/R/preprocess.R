# =====================================================================
# Stage 3 - DBN pipeline: Stage 2 post RDS -> space-time grid RDS.
# Ring and bin indices always derive from the hazard config:
#   (1) the completed grid is expand_grid(1:K x 1:T), never a hard-coded
#       20-ring layout (which would create phantom rings for the 50-km
#       hazards and overflow the Stan upper bound)
#   (2) dist_bin is capped at K
# =====================================================================

parse_emo <- function(json_str) {
  missing_result <- c(angry = NA_real_, fear = NA_real_, sad = NA_real_)
  if (length(json_str) != 1 || is.na(json_str) || trimws(json_str) == "") {
    return(missing_result)
  }
  clean_str <- gsub('""', '"', json_str, fixed = TRUE)
  probs <- tryCatch(jsonlite::fromJSON(clean_str), error = function(e) NULL)
  if (is.null(probs)) return(missing_result)

  keys <- c("angry", "fear", "sad")
  values <- vapply(keys, function(key) {
    value <- probs[[key]]
    if (is.null(value) || length(value) != 1) return(NA_real_)
    suppressWarnings(as.numeric(value))
  }, numeric(1))
  names(values) <- keys

  # Invalid or incomplete model output is missing data, not neutral emotion.
  if (any(!is.finite(values)) || any(values < 0 | values > 1) ||
      sum(values) > 1 + 1e-6) {
    return(missing_result)
  }
  values
}

mean_finite_or_na <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) NA_real_ else mean(x)
}

# Resolve the one explicitly configured Stage 2 behavior score without using
# a thresholded label or span presence. No fallback is allowed.
resolve_behavior_probability <- function(df, context = "Stage 2 data") {
  selected <- BEHAVIOR_PROBABILITY_COLUMN
  if (!(selected %in% names(df))) {
    stop(context, " 缺少当前锁定的行为概率字段: ", selected)
  }
  original <- df[[selected]]
  values <- suppressWarnings(as.numeric(original))
  invalid <- is.na(values) | !is.finite(values) | values < 0 | values > 1
  if (any(invalid)) {
    stop(
      context, " 的 ", selected, " 有 ", sum(invalid),
      " 个缺失、非数值或超出 [0,1] 的值"
    )
  }
  df$behavior_prob_used <- values
  list(data = df, column = selected)
}

validate_grid_columns <- function(df, context = "event") {
  numeric_cols <- c("time_gap", "lon", "lat", "center_lon", "center_lat")
  bad <- vapply(numeric_cols, function(col) {
    !is.numeric(df[[col]]) || any(!is.finite(df[[col]]))
  }, logical(1))
  if (any(bad)) {
    stop(context, " 的数值字段非法: ", paste(numeric_cols[bad], collapse = ", "))
  }
  if (any(df$lon < -180 | df$lon > 180 | df$center_lon < -180 |
          df$center_lon > 180 | df$lat < -90 | df$lat > 90 |
          df$center_lat < -90 | df$center_lat > 90)) {
    stop(context, " 含超出经纬度合法范围的坐标")
  }
  invisible(TRUE)
}

validate_event_centers <- function(posts, registry, context) {
  post_centers <- posts %>%
    dplyr::distinct(event_id, center_lon, center_lat)
  duplicated_centers <- post_centers %>%
    dplyr::count(event_id) %>%
    dplyr::filter(n != 1)
  if (nrow(duplicated_centers) > 0) {
    stop(context, " 的同一 event_id 对应多个事件中心")
  }
  check <- post_centers %>%
    dplyr::left_join(
      registry %>% dplyr::select(
        event_id, registry_lon = center_lon, registry_lat = center_lat
      ),
      by = "event_id"
    )
  if (any(is.na(check$registry_lon)) ||
      any(abs(check$center_lon - check$registry_lon) > 1e-6) ||
      any(abs(check$center_lat - check$registry_lat) > 1e-6)) {
    stop(context, " 的帖子事件中心与 data/eventlist.csv 不一致")
  }
  invisible(TRUE)
}

haversine_km <- function(lon, lat, center_lon, center_lat) {
  lon <- lon * pi / 180
  lat <- lat * pi / 180
  center_lon <- center_lon * pi / 180
  center_lat <- center_lat * pi / 180
  a <- sin((lat - center_lat) / 2)^2 +
    cos(lat) * cos(center_lat) * sin((lon - center_lon) / 2)^2
  6371.0088 * 2 * asin(pmin(1, sqrt(a)))
}

# One event -> space-time grid (self-contained: EMO_probs parsing happens inside)
grid_one_event <- function(df, cfg) {
  required <- c(
    "time_gap", "lon", "lat", "center_lon", "center_lat", "EMO_probs"
  )
  missing <- setdiff(required, names(df))
  if (length(missing) > 0) {
    stop("grid_one_event 缺少字段: ", paste(missing, collapse = ", "))
  }
  validate_grid_columns(df, "grid_one_event")
  if (!("behavior_prob_used" %in% names(df))) {
    resolved <- resolve_behavior_probability(df, "grid_one_event")
    df <- resolved$data
  }

  df <- df %>%
    dplyr::filter(time_gap >= TIME_MIN, time_gap < TIME_MAX)
  if (nrow(df) == 0) return(NULL)

  df <- df %>%
    dplyr::mutate(
      dist_km = haversine_km(lon, lat, center_lon, center_lat)
    ) %>%
    dplyr::filter(dist_km <= cfg$max_dist_km)
  if (nrow(df) == 0) return(NULL)

  df <- df %>%
    dplyr::mutate(
      angry = purrr::map_dbl(EMO_probs, ~ parse_emo(.x)["angry"]),
      fear  = purrr::map_dbl(EMO_probs, ~ parse_emo(.x)["fear"]),
      sad   = purrr::map_dbl(EMO_probs, ~ parse_emo(.x)["sad"]),
      E_post = angry + fear + sad,
      dist_bin = pmin(floor(dist_km / cfg$ring_km) + 1, cfg$K),   # upper bound = K, no hard-coded 20
      time_bin = floor((time_gap - TIME_MIN) / BIN_HOURS) + 1
    )

  agg <- df %>%
    dplyr::group_by(dist_bin, time_bin) %>%
    dplyr::summarise(
      E_obs = mean_finite_or_na(E_post),
      B_obs = mean_finite_or_na(behavior_prob_used),
      N = dplyr::n(),
      .groups = "drop"
    )

  # complete only the K x T grid that exists for this hazard (phantom-ring fix)
  tidyr::expand_grid(dist_bin = 1:cfg$K, time_bin = 1:N_TIME_BINS) %>%
    dplyr::left_join(agg, by = c("dist_bin", "time_bin")) %>%
    dplyr::mutate(
      N = dplyr::coalesce(as.integer(N), 0L),
      dist_center_km = (dist_bin - 0.5) * cfg$ring_km,
      time_center_hour = TIME_MIN + (time_bin - 0.5) * BIN_HOURS
    )
}

# Read the explicitly selected Stage 2 table.  The optimized landslide and
# debris-flow run is supplied as CSV; RDS remains supported for the other
# canonical workflows.
read_stage2_posts <- function(input_dir, hazard) {
  candidates <- file.path(input_dir, paste0(hazard, c(".rds", ".csv")))
  existing <- candidates[file.exists(candidates)]
  if (length(existing) != 1) {
    stop(
      "expected exactly one Stage 2 input for ", hazard, " in ", input_dir,
      "; found: ", paste(existing, collapse = ", ")
    )
  }
  path <- existing[[1]]
  if (grepl("\\.rds$", path, ignore.case = TRUE)) {
    posts <- readRDS(path)
  } else {
    posts <- readr::read_csv(
      path, show_col_types = FALSE,
      col_types = readr::cols(event_id = readr::col_character())
    )
  }
  attr(posts, "stage2_source_file") <- normalizePath(path)
  posts
}

# All events of one hazard. The input is the post-level Stage 2 table, not a
# Stage 3 grid RDS.
preprocess_hazard <- function(hazard, input_dir = paths$stage2_output,
                              output_dir = paths$stage3_data) {
  cfg <- hazard_configs[[hazard]]
  if (is.null(cfg)) stop("未知灾种: ", hazard)
  posts <- read_stage2_posts(input_dir, hazard)
  input_file <- attr(posts, "stage2_source_file")
  posts <- posts %>% dplyr::mutate(event_id = as.character(event_id))
  required <- c(
    "event_id", "time_gap", "lon", "lat", "center_lon", "center_lat",
    "EMO_probs"
  )
  missing <- setdiff(required, names(posts))
  if (length(missing) > 0) stop(hazard, " 缺少字段: ", paste(missing, collapse = ", "))
  resolved <- resolve_behavior_probability(posts, hazard)
  posts <- resolved$data
  behavior_probability_column <- resolved$column
  validate_grid_columns(posts, hazard)

  registry <- readr::read_csv(
    paths$event_list, show_col_types = FALSE,
    col_types = readr::cols(event_id = readr::col_character())
  ) %>% dplyr::filter(disaster_type == hazard)
  if (any(is.na(posts$event_id)) || anyDuplicated(registry$event_id)) {
    stop(hazard, " 的 event_id 缺失或 eventlist 中重复")
  }
  if (!setequal(unique(posts$event_id), registry$event_id)) {
    stop(hazard, " 的事件集合与 data/eventlist.csv 不一致")
  }
  validate_event_centers(posts, registry, hazard)

  event_frames <- split(posts, posts$event_id)
  message(sprintf(
    "[%s] 处理 %d 个事件；行为概率使用 %s...",
    hazard, length(event_frames), behavior_probability_column
  ))
  all_events <- purrr::imap(event_frames, function(df, event_id) {
    df <- df %>%
      dplyr::select(
        time_gap, lon, lat, center_lon, center_lat, EMO_probs,
        behavior_prob_used
      )
    out <- grid_one_event(df, cfg)
    if (is.null(out)) return(NULL)
    out$event_id <- event_id
    out
  })
  no_grid <- names(all_events)[vapply(all_events, is.null, logical(1))]
  if (length(no_grid) > 0) {
    stop(
      hazard, " 有事件在配置的时空窗口内没有帖子: ",
      paste(no_grid, collapse = ", ")
    )
  }
  st_data <- dplyr::bind_rows(all_events)

  # attach lag_idx (always rebuilt here, never trusted from a stored RDS)
  st_data <- attach_lag_idx(st_data)

  feat <- registry %>%
    dplyr::select(event_id, disaster_type, center_lon, center_lat,
                  mag, depth, volume, rain72h_max, imax_1h, typhoon_level)
  st_data <- st_data %>%
    dplyr::mutate(event_id = as.character(event_id)) %>%
    dplyr::left_join(feat, by = "event_id")

  # Population/GDP are main-model confounders, not separate moderator fits.
  # The helper performs log1p transformation and event within-between
  # decomposition before attaching the four standardized ring covariates.
  st_data <- attach_context_covariates(hazard, st_data)

  attr(st_data, "behavior_probability_column") <- behavior_probability_column
  attr(st_data, "behavior_probability_calibrated") <-
    identical(behavior_probability_column, "behavior_prob_calibrated")
  attr(st_data, "stage2_source_file") <- input_file

  out_file <- file.path(output_dir, paste0(hazard, "_st_data.rds"))
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  saveRDS(st_data, out_file, compress = "gzip")
  message(sprintf(
    "[%s] 已保存 %s（%d 行；E有效格 %d，B有效格 %d）",
    hazard, out_file, nrow(st_data),
    sum(!is.na(st_data$E_obs)), sum(!is.na(st_data$B_obs))
  ))
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
