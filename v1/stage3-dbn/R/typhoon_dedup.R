# =====================================================================
# Stage 3 - DBN pipeline: typhoon post deduplication + nested structure.
# Problem: the same post appears in the CSVs of several track points of
#          the same typhoon (57.8% duplication for typhoon 401); the
#          legacy pipeline counted them as independent observations
#          (pseudo-replication).
# Rule (interpretable): post unique key = created_at + text + lon + lat;
#          within one typhoon a post is attributed to the spatially
#          nearest track point (closest-approach principle); overlaps
#          across different typhoons are kept (one city hit by successive
#          typhoons belongs to two distinct events).
# Output: st_data with event_id = track point (e.g. 401-1) and
#          typhoon_id = typhoon number (e.g. 401).
# =====================================================================

typhoon_of <- function(path_id) sub("-[0-9]+$", "", path_id)

# Read one event CSV: deterministic line-by-line parser (header-driven,
# shared by all four hazards).
# Background: fread either truncates silently on rows with unescaped
# quotes (the legacy pipeline lost large fractions of posts this way) or
# guesses the column layout non-deterministically. This parser is fully
# deterministic:
#   1) the header contains no commas; splitting on commas yields column
#      names and positions (layouts differ by hazard: typhoon has 20
#      columns including generated_at/typhoon level, the others 18 with
#      Chinese dates);
#   2) the first 4 columns (lon, lat, created_at, time_gap) are comma-free;
#   3) the text field ends at the anchor ",<event number>,20" (matching
#      both 2024/ and 2024年 date styles);
#   4) after the anchor, consume [event number, n date columns,
#      center_lon, center_lat] (n derived from the header), skip the
#      hazard-feature and EMO columns, then locate EMO_probs as the first
#      brace pair (the JSON is flat);
#   5) raw_prob is taken from the right at position ncols - raw_col + 1
#      (header-derived).
# Any row failing a step is dropped and counted (never silently).
read_hazard_csv <- function(file) {
  lines <- readLines(file, warn = FALSE)
  lines <- lines[nzchar(trimws(lines))]
  n0 <- length(lines)
  if (n0 == 0) return(NULL)

  header <- strsplit(lines[1], ",", fixed = TRUE)[[1]]
  ncols <- length(header)
  col_idx <- function(nm) match(nm, header)
  i_centerlon <- col_idx("center_lon")
  i_centerlat <- col_idx("center_lat")
  i_eventid <- col_idx("event_id")
  i_emo <- col_idx("EMO")
  i_raw <- col_idx("raw_prob")
  n_date_cols <- i_centerlon - i_eventid - 1   # date columns between event_id and center_lon
  raw_from_right <- ncols - i_raw + 1
  lines <- lines[-1]                            # drop header
  n0 <- length(lines)

  make_empty <- function() data.table::data.table(
    created_at = character(), text = character(),
    lon = numeric(), lat = numeric(), time_gap = numeric(),
    center_lon = numeric(), center_lat = numeric(),
    EMO_probs = character(), raw_prob = numeric(),
    dist_to_center = numeric(), stringsAsFactors = FALSE)

  if (any(is.na(c(i_centerlon, i_centerlat, i_eventid, i_emo, i_raw))) ||
      n_date_cols < 1 || i_emo <= i_centerlat) {
    warning("表头结构异常: ", basename(file)); return(make_empty())
  }
  # columns between center_lat and EMO_probs (hazard features + EMO label)
  n_skip_after_center <- i_emo - 1 - i_centerlat

  n_drop_front <- 0L; n_drop_anchor <- 0L; n_drop_struct <- 0L
  res <- vector("list", n0)

  for (i in seq_len(n0)) {
    line <- lines[i]
    m <- regexec("^([^,]+),([^,]+),([^,]+),([^,]+),", line)
    if (m[[1]][1] < 0) { n_drop_front <- n_drop_front + 1L; next }
    vals <- regmatches(line, m)[[1]][-1]
    sub_from <- attr(m[[1]], "match.length")[1]

    # anchor ",<event number>,20" (matches 2024/9/1 and 2024年9月1日 alike)
    a <- regexpr(",[0-9]{1,3},20", substring(line, sub_from))
    if (a < 0) { n_drop_anchor <- n_drop_anchor + 1L; next }
    comma_pos <- sub_from + a - 1L
    text_val <- substring(line, sub_from, comma_pos - 1L)

    # consume: event number + n_date_cols dates + center_lon + center_lat (last two captured)
    skip_pat <- paste0("^,[0-9]{1,3},",
                       paste(replicate(n_date_cols, "20[^,]+,"), collapse = ""),
                       "([^,]+),([^,]+),")
    rest <- substring(line, comma_pos)
    m2 <- regexec(skip_pat, rest)
    if (m2[[1]][1] < 0) { n_drop_struct <- n_drop_struct + 1L; next }
    after_center <- substring(rest, attr(m2[[1]], "match.length")[1] + 1L)
    # skip hazard features + EMO label (fields may be empty, e.g. ",,happy," when losses are missing)
    if (n_skip_after_center > 0) {
      m3 <- regexec(paste0("^(", paste(replicate(n_skip_after_center, "[^,]*,"), collapse = ""), ")"), after_center)
      if (m3[[1]][1] < 0) { n_drop_struct <- n_drop_struct + 1L; next }
      after_center <- substring(after_center, attr(m3[[1]], "match.length")[1] + 1L)
    }
    mv <- regmatches(rest, m2)[[1]]
    center_lon <- suppressWarnings(as.numeric(mv[2]))
    center_lat <- suppressWarnings(as.numeric(mv[3]))

    # EMO_probs = first brace pair
    b1 <- regexpr("[{]", after_center); b2 <- regexpr("[}]", after_center)
    if (b1 < 0 || b2 < 0 || b2 < b1) { n_drop_struct <- n_drop_struct + 1L; next }
    emo_json <- substring(after_center, b1, b2)

    tks <- strsplit(substring(after_center, b2 + 1L), ",", fixed = TRUE)[[1]]
    if (length(tks) < raw_from_right) { n_drop_struct <- n_drop_struct + 1L; next }
    raw_prob <- suppressWarnings(as.numeric(tks[raw_from_right]))

    res[[i]] <- data.table::data.table(
      created_at = vals[3], text = text_val,
      lon = suppressWarnings(as.numeric(vals[1])),
      lat = suppressWarnings(as.numeric(vals[2])),
      time_gap = suppressWarnings(as.numeric(vals[4])),
      center_lon = center_lon, center_lat = center_lat,
      EMO_probs = emo_json, raw_prob = raw_prob,
      stringsAsFactors = FALSE)
  }

  out <- data.table::rbindlist(Filter(Negate(is.null), res))
  if (nrow(out) == 0) out <- make_empty()
  n_bad_num <- sum(!is.finite(out$lon) | !is.finite(out$lat) |
                     !is.finite(out$time_gap) | !is.finite(out$raw_prob) |
                     out$raw_prob < 0 | out$raw_prob > 1 |
                     !is.finite(out$center_lon) | !is.finite(out$center_lat))
  if (nrow(out) > 0) {
    out <- out[is.finite(lon) & is.finite(lat) & is.finite(time_gap) &
                 is.finite(raw_prob) & raw_prob >= 0 & raw_prob <= 1 &
                 is.finite(center_lon) & is.finite(center_lat)]
  }
  out$dist_to_center <- sqrt(
    ((out$lon - out$center_lon) * 111.32 * cos(pi / 180 * (out$lat + out$center_lat) / 2))^2 +
      ((out$lat - out$center_lat) * 110.574)^2)
  attr(out, "drop_report") <- c(
    总行数 = n0 + 1L, 前段失败 = n_drop_front, 锚点失败 = n_drop_anchor,
    结构失败 = n_drop_struct, 数值非法 = n_bad_num, 保留 = nrow(out))
  out
}

# Typhoon wrapper: attach track-point / typhoon ids
read_path_csv <- function(file) {
  out <- read_hazard_csv(file)
  if (is.null(out) || nrow(out) == 0) return(out)
  out$path_id <- tools::file_path_sans_ext(basename(file))
  out$typhoon_id <- typhoon_of(out$path_id)
  out
}

# Deduplicate and build st_data.
# typhoons: NULL = all 14 typhoons; otherwise a subset for testing, e.g. c("401","405")
dedup_typhoon <- function(typhoons = NULL,
                          input_dir = file.path(paths$raw_input, "typhoon"),
                          cfg = hazard_configs$typhoon,
                          report = TRUE) {
  files <- list.files(input_dir, pattern = "[.]csv$", full.names = TRUE)
  ids <- tools::file_path_sans_ext(basename(files))
  if (!is.null(typhoons)) {
    keep <- typhoon_of(ids) %in% typhoons
    files <- files[keep]; ids <- ids[keep]
  }

  reads <- lapply(files, read_path_csv)
  drop_reports <- do.call(rbind, lapply(reads, function(x) {
    r <- attr(x, "drop_report"); if (is.null(r)) return(NULL)
    as.data.frame(as.list(r))
  }))
  if (report && nrow(drop_reports) > 0) {
    cat("[解析丢弃报告] 各文件合计:\n")
    print(colSums(drop_reports))
  }
  all_posts <- data.table::rbindlist(reads)

  n_raw <- nrow(all_posts)
  # post unique key (within one typhoon)
  all_posts$post_key <- paste(all_posts$created_at, all_posts$text,
                              all_posts$lon, all_posts$lat, sep = "|")
  # keep, per key, the row nearest to a track point -- dedup scope = one
  # typhoon (across typhoons overlaps are kept: each typhoon is a distinct
  # event and the overlapping post is a legitimate observation for both;
  # consistent with cross-event handling for earthquakes).
  # The earlier global dedup wrongly reassigned overlaps of successive
  # typhoons (401-410), costing typhoon 410 77% of its potential data
  # (49,414 -> 27,958).
  data.table::setorder(all_posts, typhoon_id, post_key, dist_to_center)
  deduped <- all_posts[!duplicated(paste(all_posts$typhoon_id, all_posts$post_key))]
  n_dedup <- nrow(deduped)
  if (report) {
    message(sprintf("[dedup] 原始行数 %d -> 唯一帖子 %d（去除 %.1f%% 重复计入）",
                   n_raw, n_dedup, 100 * (1 - n_dedup / n_raw)))
  }

  # aggregate per track point into space-time cells (reuses grid_one_event)
  grids <- split(deduped, deduped$path_id)
  st_list <- lapply(names(grids), function(pid) {
    g <- grid_one_event(grids[[pid]], cfg)
    if (is.null(g)) return(NULL)
    g$event_id <- pid
    g$typhoon_id <- typhoon_of(pid)
    g
  })
  st_data <- dplyr::bind_rows(st_list)

  st_data <- attach_lag_idx(st_data)

  feat <- readr::read_csv(paths$event_feat, show_col_types = FALSE,
                          col_types = readr::cols(event_id = readr::col_character()))
  st_data <- st_data %>%
    dplyr::mutate(event_id = as.character(event_id)) %>%
    dplyr::left_join(feat, by = "event_id")

  if (report) {
    message(sprintf("[dedup] st_data: %d 路径点 x %d环 x %d窗, 有观测格子 %d/%d",
                   length(unique(st_data$event_id)), cfg$K, N_TIME_BINS,
                   sum(!is.na(st_data$E_obs)), nrow(st_data)))
  }
  st_data
}
