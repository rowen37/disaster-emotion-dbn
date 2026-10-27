# =====================================================================
# 环级调节变量提取：从栅格（人口密度 chn_pd_2020_1km.tif / GDP gdp2020.tif）
# 为每个事件的每个距离环提取均值 -> [J x K] 标准化矩阵
# 取代旧 landslide_pop_variation.R / *_gdp_variation.R 里逐灾种复制的提取代码
# =====================================================================

# 通用：给定事件中心表（event_id, center_lon, center_lat）与灾种配置，
# 返回 [J x K] 环均值矩阵（行顺序 = event_id 字典序，与 fit_hazard 一致）
extract_ring_values <- function(raster_path, event_centers, cfg,
                                neg_to_na = TRUE) {
  J <- nrow(event_centers)
  K <- cfg$K
  ring_bounds <- seq(0, cfg$max_dist_km, by = cfg$max_dist_km / K)

  rast <- raster::raster(raster_path)

  centers_sf <- sf::st_as_sf(event_centers,
                             coords = c("center_lon", "center_lat"), crs = 4326)

  make_ring <- function(center_sf, inner_km, outer_km) {
    utm_zone <- floor((sf::st_coordinates(center_sf)[1] + 180) / 6) + 1
    crs_utm <- paste0("+proj=utm +zone=", utm_zone,
                      " +datum=WGS84 +units=m +no_defs")
    center_utm <- sf::st_transform(center_sf, crs_utm)
    ring <- sf::st_difference(
      sf::st_buffer(center_utm, dist = outer_km * 1000),
      sf::st_buffer(center_utm, dist = inner_km * 1000)
    )
    sf::st_transform(ring, crs = sf::st_crs(rast))
  }

  ring_list <- list(); counter <- 1
  for (j in seq_len(J)) {
    for (k in seq_len(K)) {
      ring <- tryCatch(
        make_ring(centers_sf[j, ], ring_bounds[k], ring_bounds[k + 1]),
        error = function(e) NULL
      )
      if (!is.null(ring)) {
        ring$event_row <- j
        ring$ring_idx <- k
        ring_list[[counter]] <- ring
        counter <- counter + 1
      }
    }
  }
  rings_sf <- do.call(rbind, ring_list)

  vals <- exactextractr::exact_extract(rast, rings_sf,
                                       fun = "mean", progress = FALSE)
  if (neg_to_na) vals[vals < 0] <- NA   # GDP/人口栅格的 nodata 常为负值

  mat <- matrix(NA_real_, nrow = J, ncol = K)
  for (i in seq_len(nrow(rings_sf))) {
    mat[rings_sf$event_row[i], rings_sf$ring_idx[i]] <- vals[i]
  }

  # 缺失环用同事件其余环均值填充（与旧逻辑一致）
  for (j in seq_len(J)) {
    row_mean <- mean(mat[j, ], na.rm = TRUE)
    if (!is.finite(row_mean)) row_mean <- 0
    mat[j, is.na(mat[j, ])] <- row_mean
  }
  mat
}

# 标准化（全域统一，与旧逻辑一致）
standardize_mat <- function(mat) {
  s <- scale(as.vector(mat))
  matrix(as.numeric(s), nrow = nrow(mat), ncol = ncol(mat))
}

# 一步到位：事件中心从 event_features.csv 或 RDS 提取
# 返回可直接传给 fit_hazard(mod_ring = ...) 的标准化矩阵
# dat 可传入已有的 st_data（如台风去重后的 51 点数据），默认从 rds_dir 读
build_mod_ring <- function(hazard, raster_path,
                           rds_dir = paths$legacy_rds,
                           feat_csv = paths$event_feat,
                           dat = NULL) {
  cfg <- hazard_configs[[hazard]]
  if (is.null(dat)) {
    dat <- readRDS(file.path(rds_dir, paste0(hazard, "_st_data.rds")))
  }
  feat <- readr::read_csv(feat_csv, show_col_types = FALSE,
                          col_types = readr::cols(event_id = readr::col_character()))

  ids <- sort(unique(as.character(dat$event_id)))   # 字典序 = fit_hazard 的事件序
  centers <- feat %>%
    dplyr::mutate(event_id = as.character(event_id)) %>%
    dplyr::filter(event_id %in% ids) %>%
    dplyr::arrange(event_id)

  # 回退：个别路径点缺坐标时，先试同台风编号（401-x）相邻路径点，
  # 再试剥掉路径后缀的裸台风号（单路径点台风在 features 里登记为 "405" 而非 "405-1"）
  if (nrow(centers) < length(ids)) {
    missing_ids <- setdiff(ids, centers$event_id)
    base <- feat %>% dplyr::mutate(event_id = as.character(event_id))
    for (mid in missing_ids) {
      prefix <- sub("-[0-9]+$", "", mid)
      donor <- base[base$event_id %in% c(prefix, grep(paste0("^", prefix, "-"), base$event_id, value = TRUE)), ]
      if (nrow(donor) > 0) {
        new_row <- donor[1, ]
        new_row$event_id <- mid
        new_row$center_lon <- mean(donor$center_lon)
        new_row$center_lat <- mean(donor$center_lat)
        centers <- rbind(centers, new_row)
        message(sprintf("  [回退] %s 用 %s 的坐标", mid, donor$event_id[1]))
      } else stop("事件缺坐标且无同台风路径点可回退: ", mid)
    }
    centers <- centers %>% dplyr::arrange(event_id)
  }
  stopifnot(setequal(centers$event_id, ids))

  message(sprintf("[%s] 提取 %d 个事件 x %d 环 ...", hazard, nrow(centers), cfg$K))
  mat <- extract_ring_values(raster_path, centers, cfg)
  standardize_mat(mat)
}
