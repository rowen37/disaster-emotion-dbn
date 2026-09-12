# =====================================================================
# Stage 3 - DBN pipeline: ring-level moderator extraction.
# Extracts per-event, per-ring means from a raster (population density
# chn_pd_2020_1km.tif / GDP gdp2020.tif) and returns a standardized
# [J x K] matrix for dbn_moderator.stan.
# =====================================================================

# Generic: given event centers (event_id, center_lon, center_lat) and a hazard
# config, return the [J x K] ring-mean matrix (rows in event_id order, aligned
# with fit_hazard)
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
  if (neg_to_na) vals[vals < 0] <- NA   # GDP/population rasters encode nodata as negatives

  mat <- matrix(NA_real_, nrow = J, ncol = K)
  for (i in seq_len(nrow(rings_sf))) {
    mat[rings_sf$event_row[i], rings_sf$ring_idx[i]] <- vals[i]
  }

  # fill missing rings with the event's mean over its remaining rings
  for (j in seq_len(J)) {
    row_mean <- mean(mat[j, ], na.rm = TRUE)
    if (!is.finite(row_mean)) row_mean <- 0
    mat[j, is.na(mat[j, ])] <- row_mean
  }
  mat
}

# standardize over all rings at once
standardize_mat <- function(mat) {
  s <- scale(as.vector(mat))
  matrix(as.numeric(s), nrow = nrow(mat), ncol = ncol(mat))
}

# One-stop helper: event centers come from event_features.csv or the RDS.
# Returns the standardized matrix ready for fit_hazard(mod_ring = ...).
# dat can be an existing st_data (e.g. the deduplicated 51-point typhoon
# set); by default it is read from rds_dir.
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

  ids <- sort(unique(as.character(dat$event_id)))   # lexicographic = fit_hazard event order
  centers <- feat %>%
    dplyr::mutate(event_id = as.character(event_id)) %>%
    dplyr::filter(event_id %in% ids) %>%
    dplyr::arrange(event_id)

  # Fallback for track points missing coordinates: try sibling track points of
  # the same typhoon (401-x) first, then the bare typhoon id (single-point
  # typhoons are registered as "405" rather than "405-1" in the features table)
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
