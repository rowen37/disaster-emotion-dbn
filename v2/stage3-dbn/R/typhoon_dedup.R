# =====================================================================
# 台风帖子去重 + 嵌套数据结构
# 问题：同一条帖子出现在同台风多个路径点的 CSV 里（台风401重复率57.8%），
#       旧流程把它们当作独立观测重复计入似然（伪复制）。
# 规则（可解释）：帖子唯一键 = created_at + text + lon + lat；
#       同一台风内，帖子归属于【空间距离最近的路径点】（最近接近原则）；
#       不同台风间的重叠不去重（同城市先后受不同台风影响属不同事件）。
# 输出：st_data，event_id = 路径点（如 401-1），typhoon_id = 台风号（如 401）
# =====================================================================

typhoon_of <- function(path_id) sub("-[0-9]+$", "", path_id)

# 读单个事件 CSV —— 确定性逐行解析器（表头驱动，通用于四灾种）
# 背景：fread 在含未转义引号的行上要么静默截断（旧管道大量丢帖），
#       要么非确定性误判列结构。本解析器行为完全确定：
#   1) 表头无逗号，按逗号切分得到列名与列位置（灾种间布局不同：
#      台风 20 列含 generated_at/台风等级；其余 18 列中文日期）；
#   2) 前 4 列（lon,lat,created_at,time_gap）均为无逗号字段；
#   3) 文本列由锚点 ",<事件号>,20"（匹配 2024/ 与 2024年 两种日期）界定结束；
#   4) 锚点后消耗 [事件号, 日期列×n, center_lon, center_lat]（n 由表头推导），
#      再按列数跳过灾害特征列与 EMO 词，定位 EMO_probs 的首对花括号（JSON 无嵌套）；
#   5) raw_prob 的右数位置 = 总列数 - raw_prob列位 + 1（表头推导）。
# 任一步失败的行：丢弃并计数（不静默）。
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
  n_date_cols <- i_centerlon - i_eventid - 1   # event_id 与 center_lon 之间的日期列数
  raw_from_right <- ncols - i_raw + 1
  lines <- lines[-1]                            # 去表头
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
  # center_lat 之后、EMO_probs 之前的列数（灾害特征列 + EMO 词）
  n_skip_after_center <- i_emo - 1 - i_centerlat

  n_drop_front <- 0L; n_drop_anchor <- 0L; n_drop_struct <- 0L
  res <- vector("list", n0)

  for (i in seq_len(n0)) {
    line <- lines[i]
    m <- regexec("^([^,]+),([^,]+),([^,]+),([^,]+),", line)
    if (m[[1]][1] < 0) { n_drop_front <- n_drop_front + 1L; next }
    vals <- regmatches(line, m)[[1]][-1]
    sub_from <- attr(m[[1]], "match.length")[1]

    # 锚点 ",<事件号>,20"（兼容 2024/9/1 与 2024年9月1日）
    a <- regexpr(",[0-9]{1,3},20", substring(line, sub_from))
    if (a < 0) { n_drop_anchor <- n_drop_anchor + 1L; next }
    comma_pos <- sub_from + a - 1L
    text_val <- substring(line, sub_from, comma_pos - 1L)

    # 消耗: 事件号 + n_date_cols 个日期 + center_lon + center_lat（后两者捕获）
    skip_pat <- paste0("^,[0-9]{1,3},",
                       paste(replicate(n_date_cols, "20[^,]+,"), collapse = ""),
                       "([^,]+),([^,]+),")
    rest <- substring(line, comma_pos)
    m2 <- regexec(skip_pat, rest)
    if (m2[[1]][1] < 0) { n_drop_struct <- n_drop_struct + 1L; next }
    after_center <- substring(rest, attr(m2[[1]], "match.length")[1] + 1L)
    # 跳过灾害特征列 + EMO 词（单个字段，允许为空——如经济损失缺失时 ",,happy,"）
    if (n_skip_after_center > 0) {
      m3 <- regexec(paste0("^(", paste(replicate(n_skip_after_center, "[^,]*,"), collapse = ""), ")"), after_center)
      if (m3[[1]][1] < 0) { n_drop_struct <- n_drop_struct + 1L; next }
      after_center <- substring(after_center, attr(m3[[1]], "match.length")[1] + 1L)
    }
    mv <- regmatches(rest, m2)[[1]]
    center_lon <- suppressWarnings(as.numeric(mv[2]))
    center_lat <- suppressWarnings(as.numeric(mv[3]))

    # EMO_probs = 首对花括号
    b1 <- regexpr("[{]", after_center); b2 <- regexpr("[}]", after_center)
    if (b1 < 0 || b2 < 0 || b2 < b1) { n_drop_struct <- n_drop_struct + 1L; next }
    emo_json <- substring(after_center, b1, b2)

    tks <- strsplit(substring(after_center, b2 + 1L), ",", fixed = TRUE)[[1]]
    if (length(tks) < raw_from_right) { n_drop_struct <- n_drop_struct + 1L; next }
    # raw_prob 按“距行尾的固定列数”从右侧定位：pred_span 等中间列含逗号会
    # 使左侧 token 计数漂移，必须从右数而不是从左数。
    raw_prob <- suppressWarnings(as.numeric(tks[length(tks) - raw_from_right + 1L]))

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

# 台风兼容入口：附路径点/台风编号
read_path_csv <- function(file) {
  out <- read_hazard_csv(file)
  if (is.null(out) || nrow(out) == 0) return(out)
  out$path_id <- tools::file_path_sans_ext(basename(file))
  out$typhoon_id <- typhoon_of(out$path_id)
  out
}

# 去重 + 生成 st_data
# typhoons: NULL = 全部14场；否则子集（测试用），如 c("401","405")
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
  # 帖子唯一键（同台风内）
  all_posts$post_key <- paste(all_posts$created_at, all_posts$text,
                              all_posts$lon, all_posts$lat, sep = "|")
  # 每键保留距最近路径点的行——去重范围 = 台风内（跨台风保留：每场台风是
  # 独立事件，重叠帖对两场都是合法观测；与地震跨事件处理一致）。
  # 注意不能用全局去重：前后脚台风（401-410）的重叠帖会被全归空间最近方，
  # 台风410曾因此损失 77% 潜在数据（49,414 -> 27,958）。
  data.table::setorder(all_posts, typhoon_id, post_key, dist_to_center)
  deduped <- all_posts[!duplicated(paste(all_posts$typhoon_id, all_posts$post_key))]
  n_dedup <- nrow(deduped)
  if (report) {
    message(sprintf("[dedup] 原始行数 %d -> 唯一帖子 %d（去除 %.1f%% 重复计入）",
                   n_raw, n_dedup, 100 * (1 - n_dedup / n_raw)))
  }

  # 按路径点聚合为时空格子（复用 grid_one_event）
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
