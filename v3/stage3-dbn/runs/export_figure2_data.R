# =====================================================================
# Stage 3 - run script: export plotting data for the Python figure
# renderer (make_figures.py).
# Output: output/figure_2/_data/*.csv + meta.json
# Covers: Fig1 density/forest, Fig2 temporal curves, Fig3 spatial decay,
# Fig4 per-ring simulated trajectories, FigS1/FigS2 maps (grids +
# polygons + masks).
# =====================================================================
script_file <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]]))
source(file.path(dirname(dirname(script_file)), "R", "00_init.R"))
suppressPackageStartupMessages({
  library(terra); library(sf)
})

outd <- file.path(paths$out_dir, "figure_2", "_data")
dir.create(outd, showWarnings = FALSE, recursive = TRUE)

# unified column order (matches paper Figs 2/3/4)
hz_order <- c("earthquake", "landslide", "debris_flow", "typhoon")
draws_file <- function(hz) file.path(paths$out_dir, paste0(hz, "_baseline_w1_draws.rds"))
dat_file <- function(hz) file.path(paths$stage3_data, paste0(hz, "_st_data.rds"))

all_draws <- setNames(lapply(hz_order, function(h) readRDS(draws_file(h))), hz_order)
all_dat   <- setNames(lapply(hz_order, function(h) readRDS(dat_file(h))), hz_order)

write_grid <- function(r, name) {
  mm <- terra::as.matrix(r, wide = TRUE)   # row 1 = north
  stopifnot(is.matrix(mm), ncol(mm) == terra::ncol(r))
  write.table(mm, file.path(outd, paste0(name, ".csv")), sep = ",",
              row.names = FALSE, col.names = FALSE, na = "NA")
  e <- c(terra::xmin(r), terra::xmax(r), terra::ymin(r), terra::ymax(r))
  list(file = paste0(name, ".csv"), nrow = nrow(mm), ncol = ncol(mm),
       ext = e)
}

# ---------------- Fig 1: theta1 draws + density + forest ----------------
write.csv(do.call(rbind, lapply(hz_order, function(h)
  data.frame(hazard = h, theta1 = all_draws[[h]]$theta1))),
  file.path(outd, "fig1_theta1_draws.csv"), row.names = FALSE)

write.csv(do.call(rbind, lapply(hz_order, function(h) {
  dd <- density(all_draws[[h]]$theta1)
  data.frame(hazard = h, x = dd$x, y = dd$y)
})), file.path(outd, "fig1_theta1_density.csv"), row.names = FALSE)

plist <- c("theta1", "theta2", "beta_e", "beta_b")
write.csv(do.call(rbind, lapply(hz_order, function(h)
  do.call(rbind, lapply(plist, function(p) {
    v <- all_draws[[h]][[p]]
    data.frame(hazard = h, param = p, mean = mean(v), median = median(v),
               lo = unname(quantile(v, .025)), hi = unname(quantile(v, .975)))
  })))),
  file.path(outd, "fig1_forest.csv"), row.names = FALSE)

# ---------------- Fig 2: temporal trajectories (probability scale) + observations ----------------
rows <- list(); orows <- list()
for (h in hz_order) {
  tt <- temporal_trajectory(all_draws[[h]], hazard_configs[[h]])
  ob <- obs_temporal(all_dat[[h]])
  for (ch in c("E", "B")) {
    m <- tt[[if (ch == "E") "prob_e" else "prob_b"]]
    q <- apply(m, 2, quantile, c(.025, .975))
    rows[[length(rows) + 1]] <- data.frame(
      hazard = h, channel = ch, time_hour = tt$time_hour,
      mean = colMeans(m), lo = q[1, ], hi = q[2, ])
  }
  orows[[length(orows) + 1]] <- data.frame(
    hazard = h, time_hour = ob$time_hour, obs_E = ob$mean_E, obs_B = ob$mean_B)
  cat(sprintf("[Fig2] %s E 最多到 %.3f | B 最多到 %.3f\n", h,
              max(rows[[length(rows) - 1]]$hi), max(rows[[length(rows)]]$hi)))
}
write.csv(do.call(rbind, rows), file.path(outd, "fig2_temporal.csv"), row.names = FALSE)
write.csv(do.call(rbind, orows), file.path(outd, "fig2_obs.csv"), row.names = FALSE)

# ---------------- Fig 3: spatial decay (probability scale) + observations ----------------
obs_ring_full <- function(dat, var, K) {
  d <- dat[!is.na(dat[[var]]), ]
  agg <- tapply(d[[var]], d$dist_bin, mean)
  out <- rep(NA_real_, K)
  out[as.integer(names(agg))] <- as.numeric(agg)
  out
}
rows <- list(); orows <- list()
for (h in hz_order) {
  sd_ <- spatial_decay(all_draws[[h]], hazard_configs[[h]])
  K <- hazard_configs[[h]]$K
  for (ch in c("E", "B")) {
    m <- sd_[[if (ch == "E") "prob_e" else "prob_b"]]
    q <- apply(m, 2, quantile, c(.025, .975))
    rows[[length(rows) + 1]] <- data.frame(
      hazard = h, channel = ch, dist_km = sd_$dist_km,
      mean = colMeans(m), lo = q[1, ], hi = q[2, ])
  }
  orows[[length(orows) + 1]] <- data.frame(
    hazard = h, dist_km = sd_$dist_km,
    obs_E = obs_ring_full(all_dat[[h]], "E_obs", K),
    obs_B = obs_ring_full(all_dat[[h]], "B_obs", K))
}
write.csv(do.call(rbind, rows), file.path(outd, "fig3_spatial.csv"), row.names = FALSE)
write.csv(do.call(rbind, orows), file.path(outd, "fig3_obs.csv"), row.names = FALSE)

# ---------------- Fig 4: per-ring simulated trajectories (same algorithm as make_paper_outputs.R) ----------------
ring_traj_export <- function(d, cfg, show_rings, ns = 400) {
  T_spline <- splines::ns(seq_len(N_TIME_BINS), df = cfg$df_time)
  B_spline <- splines::ns(ring_centers(cfg), df = cfg$df_dist)
  set.seed(123)
  idx <- sample(seq_len(nrow(d)), ns)
  th <- time_bin_centers()
  out <- list()
  sim <- lapply(show_rings, function(k) {
    list(E = matrix(NA_real_, ns, N_TIME_BINS),
         B = matrix(NA_real_, ns, N_TIME_BINS))
  })
  gv <- function(i, pat) as.numeric(as.matrix(d[i, grep(pat, names(d)), drop = FALSE]))
  for (s in seq_len(ns)) {
    i <- idx[s]
    tc_e <- as.numeric(gv(i, "^psi_e") %*% t(T_spline))
    tc_b <- as.numeric(gv(i, "^psi_b") %*% t(T_spline))
    tc_e <- tc_e - mean(tc_e); tc_b <- tc_b - mean(tc_b)
    le <- gv(i, "^lambda_e"); lb <- gv(i, "^lambda_b")
    je <- if (cfg$has_jump == 1) ifelse(seq_len(N_TIME_BINS) >= JUMP_START, d$jump_e[i], 0) else rep(0, N_TIME_BINS)
    jb <- if (cfg$has_jump == 1) ifelse(seq_len(N_TIME_BINS) >= JUMP_START, d$jump_b[i], 0) else rep(0, N_TIME_BINS)
    for (r in seq_along(show_rings)) {
      k <- show_rings[r]
      f_e <- as.numeric(B_spline[k, ] %*% le)
      f_b <- as.numeric(B_spline[k, ] %*% lb)
      e <- b <- numeric(N_TIME_BINS)
      e[1] <- d$alpha_e[i] + tc_e[1] + f_e + je[1] + rnorm(1, 0, d$sigma_proc_e[i])
      b[1] <- d$alpha_b[i] + tc_b[1] + f_b + jb[1] + d$theta1[i] * e[1] + rnorm(1)
      for (t in 2:N_TIME_BINS) {
        e[t] <- d$alpha_e[i] + tc_e[t] + f_e + je[t] + d$beta_e[i] * e[t - 1] +
          rnorm(1, 0, d$sigma_proc_e[i])
        b[t] <- d$alpha_b[i] + tc_b[t] + f_b + jb[t] + d$beta_b[i] * b[t - 1] +
          d$theta1[i] * e[t] + d$theta2[i] * e[t - 1] + rnorm(1)
      }
      sim[[r]]$E[s, ] <- plogis(e); sim[[r]]$B[s, ] <- plogis(b)
    }
  }
  for (r in seq_along(show_rings)) {
    k <- show_rings[r]
    lab <- paste0((k - 1) * cfg$ring_km, "-", k * cfg$ring_km, " km")
    for (ch in c("E", "B")) {
      m <- sim[[r]][[ch]]
      q <- apply(m, 2, quantile, c(.025, .975))
      out[[length(out) + 1]] <- data.frame(
        hazard = cfg$label, channel = ch, ring = lab, ring_idx = k,
        time_hour = th, mean = colMeans(m), lo = q[1, ], hi = q[2, ])
    }
  }
  do.call(rbind, out)
}

rows <- list()
for (h in hz_order) {
  cfg <- hazard_configs[[h]]
  show <- if (cfg$K == 10) c(1, 3, 5, 7, 10) else c(1, 5, 10, 15, 20)
  rows[[length(rows) + 1]] <- ring_traj_export(all_draws[[h]], cfg, show)
  cat("[Fig4]", h, "完成\n")
}
write.csv(do.call(rbind, rows), file.path(outd, "fig4_rings.csv"), row.names = FALSE)

# =====================================================================
# FigS1 / FigS2: map data (grids + polygons + masks + representative events)
# =====================================================================
pop_r <- terra::rast(paths$pop_raster)
pop_r[pop_r < 0] <- NA

buf_poly <- function(lon, lat, radius_m) {
  pt <- sf::st_as_sf(data.frame(lon = lon, lat = lat), coords = c("lon", "lat"), crs = 4326)
  uz <- floor((lon + 180) / 6) + 1
  crs_utm <- paste0("+proj=utm +zone=", uz, " +datum=WGS84 +units=m +no_defs")
  sf::st_transform(sf::st_buffer(sf::st_transform(pt, crs_utm), radius_m), 4326)
}
poly_to_df <- function(geom, ...) {
  m <- sf::st_coordinates(geom)
  data.frame(x = m[, "X"], y = m[, "Y"], ...)
}

meta <- list()

# ---- S1: earthquakes ----
eq_centers <- unique(all_dat$earthquake[, c("event_id", "center_lon", "center_lat")])
eq_centers$event_id <- as.character(eq_centers$event_id)
eq_pe <- read.csv(file.path(paths$out_dir, "supp_ring_population_perevent_earthquake.csv"),
                  colClasses = c(event_id = "character"))
stopifnot(setequal(eq_pe$event_id, eq_centers$event_id))
eq_pe <- eq_pe[match(eq_centers$event_id, eq_pe$event_id), ]
far <- eq_pe$ring10
i_hi  <- which.max(far)
i_mid <- which.min(abs(far - stats::median(far)))

meta$s1$overview <- write_grid(terra::aggregate(pop_r, fact = 8, fun = "mean"), "s1_overview_grid")
rings50 <- do.call(rbind, lapply(seq_len(nrow(eq_centers)), function(j)
  poly_to_df(buf_poly(eq_centers$center_lon[j], eq_centers$center_lat[j], 50e3),
             event_id = eq_centers$event_id[j])))
write.csv(rings50, file.path(outd, "s1_rings_50km.csv"), row.names = FALSE)
write.csv(eq_centers, file.path(outd, "s1_centers.csv"), row.names = FALSE)

meta$s1$zoom <- list()
zrings <- list(); ann <- list()
for (zj in c(i_hi, i_mid)) {
  j <- zj
  lon <- eq_centers$center_lon[j]; lat <- eq_centers$center_lat[j]
  meta$s1$zoom[[length(meta$s1$zoom) + 1]] <- list(
    id = zj,
    event_id = eq_centers$event_id[j], lon = lon, lat = lat,
    far_dens = unname(far[j]),
    tag = if (zj == i_hi) "highest" else "median",
    grid = write_grid(terra::crop(pop_r, terra::ext(buf_poly(lon, lat, 68e3))),
                      sprintf("s1_zoom%d_grid", zj)))
  for (km in seq(10, 50, by = 10))
    zrings[[length(zrings) + 1]] <- poly_to_df(
      buf_poly(lon, lat, km * 1000), zoom = zj, km = km)
  ann[[length(ann) + 1]] <- poly_to_df(
    sf::st_difference(buf_poly(lon, lat, 50e3), buf_poly(lon, lat, 40e3)), zoom = zj)
}
write.csv(do.call(rbind, zrings), file.path(outd, "s1_zoom_rings.csv"), row.names = FALSE)
write.csv(do.call(rbind, ann), file.path(outd, "s1_zoom_annulus.csv"), row.names = FALSE)

# ---- S2: typhoons ----
ty <- all_dat$typhoon
ty_c <- unique(ty[, c("event_id", "typhoon_id", "center_lon", "center_lat")])
ty_c$event_id <- as.character(ty_c$event_id)
ty_c$typhoon_id <- as.character(ty_c$typhoon_id)
if (any(is.na(ty_c$center_lon) | is.na(ty_c$center_lat))) {   # coordinate fallback, same rule as moderators.R
  feat <- utils::read.csv(paths$event_feat, colClasses = "character")
  feat$center_lon <- as.numeric(feat$center_lon)
  feat$center_lat <- as.numeric(feat$center_lat)
  for (b in which(is.na(ty_c$center_lon) | is.na(ty_c$center_lat))) {
    prefix <- sub("-[0-9]+$", "", ty_c$event_id[b])
    cand <- feat[feat$event_id %in%
      c(prefix, grep(paste0("^", prefix, "-"), feat$event_id, value = TRUE)), ]
    if (nrow(cand) > 0) {
      ty_c$center_lon[b] <- mean(cand$center_lon)
      ty_c$center_lat[b] <- mean(cand$center_lat)
    }
  }
  stopifnot(!any(is.na(ty_c$center_lon) | is.na(ty_c$center_lat)))
}

ty_pe <- read.csv(file.path(paths$out_dir, "supp_ring_population_perevent_typhoon.csv"),
                  colClasses = c(event_id = "character"))
ty_pe <- ty_pe[match(ty_c$event_id, ty_pe$event_id), ]
contrast <- ty_pe$V20 - ty_pe$V2
i_zoom <- which.max(contrast)
zp <- ty_c[i_zoom, ]

corridor_geoms <- do.call(c, lapply(seq_len(nrow(ty_c)), function(j)
  sf::st_geometry(buf_poly(ty_c$center_lon[j], ty_c$center_lat[j], 200e3))))
corridor <- sf::st_union(corridor_geoms)
bb <- as.numeric(sf::st_bbox(corridor))
ex_ty <- terra::ext(bb[1] - 0.6, bb[3] + 0.6, bb[2] - 0.6, bb[4] + 0.6)

meta$s2$overview <- write_grid(
  terra::aggregate(terra::crop(pop_r, ex_ty), fact = 4, fun = "mean"), "s2_overview_grid")
mkgrid <- terra::rast(ex_ty, resolution = 0.04, crs = "+proj=longlat +datum=WGS84")
mk <- terra::rasterize(terra::vect(sf::st_as_sf(corridor)), mkgrid, field = 1, background = 0)
meta$s2$corridor_mask <- write_grid(mk, "s2_corridor_mask")
tracks <- do.call(rbind, lapply(split(ty_c, ty_c$typhoon_id), function(g) {
  g <- g[order(g$event_id), ]
  data.frame(typhoon_id = g$typhoon_id[1], pt = seq_len(nrow(g)),
             x = g$center_lon, y = g$center_lat)
}))
write.csv(tracks, file.path(outd, "s2_tracks.csv"), row.names = FALSE)
write.csv(ty_c, file.path(outd, "s2_points.csv"), row.names = FALSE)

meta$s2$zoom <- list(
  event_id = zp$event_id, lon = zp$center_lon, lat = zp$center_lat,
  near10 = unname(ty_pe$V1[i_zoom]), mid100 = unname(ty_pe$V10[i_zoom]),
  ring_dens = as.numeric(ty_pe[i_zoom, paste0("V", 1:20)]),
  grid = write_grid(terra::crop(pop_r, terra::ext(buf_poly(zp$center_lon, zp$center_lat, 260e3))),
                    "s2_zoom_grid"))
zrings2 <- do.call(rbind, lapply(seq(20, 200, by = 20), function(km)
  poly_to_df(buf_poly(zp$center_lon, zp$center_lat, km * 1000), km = km)))
write.csv(zrings2, file.path(outd, "s2_zoom_rings.csv"), row.names = FALSE)
# mirror of S1: the zoom highlights the 90-100 km annulus (the ring quoted in the title)
ann2 <- poly_to_df(sf::st_difference(
  buf_poly(zp$center_lon, zp$center_lat, 100e3),
  buf_poly(zp$center_lon, zp$center_lat, 90e3)))
write.csv(ann2, file.path(outd, "s2_zoom_annulus.csv"), row.names = FALSE)

jsonlite::write_json(meta, file.path(outd, "meta.json"), auto_unbox = TRUE, digits = 10)
cat("导出完成 ->", outd, "\n")
print(list.files(outd))
