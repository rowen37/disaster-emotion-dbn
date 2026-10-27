# =====================================================================
# Stage 3 - run script: supplementary figures.
#   1) Fig3_spatial_split.pdf -- (a) emotion baseline (probability
#      scale, Delta-corrected) row, (b) behavior baseline row; columns
#      = four hazards, matching the Fig1/Fig4 ordering
#   2) FigS1_eq_ring_population.png -- 50-km rings of 37 earthquake
#      events over the 2020 1-km population raster (country overview +
#      two representative zooms: highest far-ring density / median)
#   3) FigS2_ty_ring_population.png -- 200-km collection corridor of 14
#      typhoons / 52 track points + a representative 10-km ring zoom
#      (near-ring over sea vs. densely populated mid/far rings)
#   Side output: output/supp_ring_population_perevent*.csv (per-event
#   raw ring densities)
# =====================================================================
script_file <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]]))
source(file.path(dirname(dirname(script_file)), "R", "00_init.R"))
suppressPackageStartupMessages({
  library(terra); library(sf); library(viridisLite)
})
stopifnot(requireNamespace("exactextractr", quietly = TRUE),
          requireNamespace("raster", quietly = TRUE))

fig_dir <- paths$fig_dir
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

hz_order <- c("earthquake", "typhoon", "landslide", "debris_flow")
hz_cn <- c(earthquake = "Earthquake", typhoon = "Typhoon",
           landslide = "Landslide", debris_flow = "Debris flow")
draws_file <- function(hz) file.path(paths$out_dir, paste0(hz, "_baseline_w1_draws.rds"))
dat_file <- function(hz) file.path(paths$stage3_data, paste0(hz, "_st_data.rds"))
all_draws <- setNames(lapply(hz_order, function(h) readRDS(draws_file(h))), hz_order)
all_dat   <- setNames(lapply(hz_order, function(h) readRDS(dat_file(h))), hz_order)

# =====================================================================
# 1) Fig 3 split: probability scale, emotion/behavior rows, four hazard columns
# =====================================================================
obs_ring_full <- function(dat, var, K) {
  d <- dat[!is.na(dat[[var]]), ]
  agg <- tapply(d[[var]], d$dist_bin, mean)
  out <- rep(NA_real_, K)
  out[as.integer(names(agg))] <- as.numeric(agg)
  out
}

grDevices::pdf(file.path(fig_dir, "Fig3_spatial_split.pdf"),
               width = 13, height = 7.4, pointsize = 11)
par(mfrow = c(2, 4), mar = c(4, 4.2, 2.2, 1), las = 1,
    cex.axis = 0.8, cex.lab = 0.95, mgp = c(2, 0.6, 0))
rows <- list(list(v = "E", tag = "(a)"), list(v = "B", tag = "(b)"))
for (rs in rows) {
  for (h in hz_order) {
    cfg <- hazard_configs[[h]]
    dc <- ring_centers(cfg)
    sdec <- spatial_decay(all_draws[[h]], cfg)
    prob <- if (rs$v == "E") sdec$prob_e else sdec$prob_b
    ci <- apply(prob, 2, stats::quantile, c(0.025, 0.975))
    obs <- obs_ring_full(all_dat[[h]], if (rs$v == "E") "E_obs" else "B_obs", cfg$K)
    col <- if (rs$v == "E") "#C0392B" else "#2471A3"
    ym <- max(c(ci, obs), na.rm = TRUE) * 1.06
    plot(dc, colMeans(prob), type = "n", xlim = c(0, cfg$max_dist_km),
         ylim = c(0, ym), xlab = "Distance from event center (km)",
         ylab = if (rs$v == "E") "Negative emotion probability"
                else "Behavior probability (relative score)",
         main = if (rs$v == "E") hz_cn[h] else "", cex.main = 1)
    polygon(c(dc, rev(dc)), c(ci[1, ], rev(ci[2, ])),
            col = grDevices::adjustcolor(col, 0.12), border = NA)
    lines(dc, colMeans(prob), col = col, lwd = 2)
    keep <- !is.na(obs)
    points(dc[keep], obs[keep], pch = 16, cex = 0.85)
    if (h == hz_order[1])   # row tag (a)/(b) at the top-left of each row's first panel
      graphics::text(par("usr")[1], par("usr")[4], rs$tag, adj = c(-0.15, 1.25),
                     font = 2, cex = 1.1, xpd = NA)
  }
}
grDevices::dev.off()
cat("[1/3] Fig3_spatial_split.pdf 完成\n")

# =====================================================================
# shared: population raster + ring buffer geometry
# =====================================================================
pop_r <- terra::rast(paths$pop_raster)
pop_r[pop_r < 0] <- NA    # nodata is negative (same convention as moderators.R)

buf_poly <- function(lon, lat, radius_m) {
  pt <- sf::st_as_sf(data.frame(lon = lon, lat = lat), coords = c("lon", "lat"), crs = 4326)
  uz <- floor((lon + 180) / 6) + 1
  crs_utm <- paste0("+proj=utm +zone=", uz, " +datum=WGS84 +units=m +no_defs")
  sf::st_transform(sf::st_buffer(sf::st_transform(pt, crs_utm), radius_m), 4326)
}

ring_mat_for <- function(hz) {   # per-event ring densities (raw, with same-event fill fallback)
  d <- all_dat[[hz]]
  centers <- unique(d[, c("event_id", "center_lon", "center_lat")])
  centers$event_id <- as.character(centers$event_id)
  mat <- extract_ring_values(paths$pop_raster, centers, hazard_configs[[hz]])
  list(centers = centers, mat = mat)
}

# =====================================================================
# 2) Fig S1: earthquake rings on population (overview + 2 zooms)
# =====================================================================
eq <- ring_mat_for("earthquake")
cfg_eq <- hazard_configs$earthquake
colnames(eq$mat) <- paste0("ring", seq_len(cfg_eq$K))
write.csv(cbind(event_id = eq$centers$event_id, as.data.frame(eq$mat)),
          file.path(paths$out_dir, "supp_ring_population_perevent_earthquake.csv"),
          row.names = FALSE)

i_hi  <- which.max(eq$mat[, cfg_eq$K])                                  # densest far ring
i_mid <- which.min(abs(eq$mat[, cfg_eq$K] - stats::median(eq$mat[, cfg_eq$K])))  # typical
zoom_eq <- c(i_hi, i_mid)

grDevices::png(file.path(fig_dir, "FigS1_eq_ring_population.png"),
               width = 10.5, height = 9.6, units = "in", res = 300)
graphics::layout(matrix(c(1, 1, 2, 3), 2, 2, byrow = TRUE), heights = c(1.25, 1))
# ---- overview: country raster + 37 x 50-km rings ----
ov <- terra::aggregate(pop_r, fact = 4)
terra::plot(log10(ov + 1), col = viridisLite::viridis(120),
            axes = FALSE, mar = c(2, 2, 2.5, 5),
            main = "Earthquake events: 50-km collection rings on 2020 population density")
for (j in seq_len(nrow(eq$centers)))
  plot(buf_poly(eq$centers$center_lon[j], eq$centers$center_lat[j], 50e3),
       add = TRUE, border = grDevices::adjustcolor("#C0392B", 0.9), lwd = 1.2)
points(eq$centers$center_lon, eq$centers$center_lat, pch = 16, cex = 0.5)
mtext("log10(persons/km2 + 1)", side = 4, las = 0, cex = 0.8)
# ---- zooms: two representative events, 5-km ring boundaries + highlighted 40-50 km annulus ----
for (j in zoom_eq) {
  ex <- terra::ext(buf_poly(eq$centers$center_lon[j], eq$centers$center_lat[j], 68e3))
  rc <- terra::crop(pop_r, ex)
  terra::plot(log10(rc + 1), col = viridisLite::viridis(120), axes = FALSE,
              mar = c(2, 2, 2.5, 5),
              main = sprintf("%s: 45-50 km ring = %.0f pers./km2",
                             eq$centers$event_id[j], eq$mat[j, cfg_eq$K]))
  for (km in seq(10, 50, by = 10))
    plot(buf_poly(eq$centers$center_lon[j], eq$centers$center_lat[j], km * 1000),
         add = TRUE, border = "white", lwd = 1)
  plot(sf::st_difference(
         buf_poly(eq$centers$center_lon[j], eq$centers$center_lat[j], 50e3),
         buf_poly(eq$centers$center_lon[j], eq$centers$center_lat[j], 40e3)),
       add = TRUE, col = grDevices::adjustcolor("#C0392B", 0.35),
       border = grDevices::adjustcolor("#C0392B", 0.9), lwd = 1.5)
  points(eq$centers$center_lon[j], eq$centers$center_lat[j], pch = 16,
         cex = 1.1, col = "#C0392B")
}
grDevices::dev.off()
cat("[2/3] FigS1_eq_ring_population.png 完成\n")

# =====================================================================
# 3) Fig S2: typhoon tracks + 200-km corridor + representative track-point zoom
# =====================================================================
ty <- all_dat$typhoon
ty_c <- unique(ty[, c("event_id", "typhoon_id", "center_lon", "center_lat")])
ty_c$event_id <- as.character(ty_c$event_id)
ty_c$typhoon_id <- as.character(ty_c$typhoon_id)
# coordinate fallback (same rule as build_mod_ring in moderators.R): try the bare
# typhoon id first, then sibling track points
if (any(is.na(ty_c$center_lon) | is.na(ty_c$center_lat))) {
  feat <- utils::read.csv(paths$event_feat, colClasses = "character")
  feat$center_lon <- as.numeric(feat$center_lon)
  feat$center_lat <- as.numeric(feat$center_lat)
  bad <- which(is.na(ty_c$center_lon) | is.na(ty_c$center_lat))
  for (b in bad) {
    prefix <- sub("-[0-9]+$", "", ty_c$event_id[b])
    cand <- feat[feat$event_id %in%
      c(prefix, grep(paste0("^", prefix, "-"), feat$event_id, value = TRUE)), ]
    if (nrow(cand) > 0) {
      ty_c$center_lon[b] <- mean(cand$center_lon)
      ty_c$center_lat[b] <- mean(cand$center_lat)
      message(sprintf("  [回退] %s 用 %s 的坐标", ty_c$event_id[b], cand$event_id[1]))
    }
  }
  stopifnot(!any(is.na(ty_c$center_lon) | is.na(ty_c$center_lat)))
}
ty_ring <- list(
  centers = ty_c[, c("event_id", "center_lon", "center_lat")],
  mat = extract_ring_values(paths$pop_raster,
         ty_c[, c("event_id", "center_lon", "center_lat")], hazard_configs$typhoon))
cfg_ty <- hazard_configs$typhoon
write.csv(cbind(event_id = ty_ring$centers$event_id,
                as.data.frame(ty_ring$mat)),
          file.path(paths$out_dir, "supp_ring_population_perevent_typhoon.csv"),
          row.names = FALSE)

contrast <- ty_ring$mat[, cfg_ty$K] - ty_ring$mat[, 2]   # far- vs near-ring density contrast
i_zoom <- which.max(contrast)
zp <- ty_ring$centers[i_zoom, ]

tracks <- lapply(split(ty_c, ty_c$typhoon_id), function(g) {
  g <- g[order(g$event_id), ]
  if (nrow(g) >= 2) sf::st_sf(
    typhoon_id = g$typhoon_id[1],
    geom = sf::st_sfc(sf::st_linestring(as.matrix(g[, c("center_lon", "center_lat")])),
                      crs = 4326))
})
tracks_sf <- do.call(rbind, tracks)
corridor_geoms <- do.call(c, lapply(seq_len(nrow(ty_c)), function(j)
  sf::st_geometry(buf_poly(ty_c$center_lon[j], ty_c$center_lat[j], 200e3))))
corridor <- sf::st_sf(geom = sf::st_sfc(sf::st_union(corridor_geoms), crs = 4326))

grDevices::png(file.path(fig_dir, "FigS2_ty_ring_population.png"),
               width = 10.5, height = 9.6, units = "in", res = 300)
graphics::layout(matrix(c(1, 1, 2, 3), 2, 2, byrow = TRUE), heights = c(1.25, 1))
# ---- overview: southeast coast raster + corridor + tracks ----
ex_ty <- terra::ext(corridor)
ex_ty <- terra::ext(ex_ty[1] - 0.6, ex_ty[2] + 0.6, ex_ty[3] - 0.6, ex_ty[4] + 0.6)
ov_ty <- terra::crop(pop_r, ex_ty)
terra::plot(log10(ov_ty + 1), col = viridisLite::viridis(120), axes = FALSE,
            mar = c(2, 2, 2.5, 5),
            main = "Typhoon tracks and 200-km collection corridor on 2020 population density")
plot(corridor, add = TRUE, col = grDevices::adjustcolor("#2471A3", 0.10),
     border = grDevices::adjustcolor("#2471A3", 0.55), lwd = 1.2)
plot(tracks_sf, add = TRUE, lwd = 1.4, col = "#2471A3")
points(ty_c$center_lon, ty_c$center_lat, pch = 16, cex = 0.45, col = "#2471A3")
mtext("log10(persons/km2 + 1)", side = 4, las = 0, cex = 0.8)
# ---- zoom: representative track point, 10-km rings out to 200 km ----
ex_z <- terra::ext(buf_poly(zp$center_lon, zp$center_lat, 260e3))
rz <- terra::crop(pop_r, ex_z)
terra::plot(log10(rz + 1), col = viridisLite::viridis(120), axes = FALSE,
            mar = c(2, 2, 2.5, 5),
            main = sprintf("Track point %s: 0-10 km = %.0f, 90-100 km = %.0f pers./km2",
                           zp$event_id, ty_ring$mat[i_zoom, 1], ty_ring$mat[i_zoom, 10]))
for (km in seq(20, 200, by = 20))
  plot(buf_poly(zp$center_lon, zp$center_lat, km * 1000),
       add = TRUE, border = "white", lwd = 1)
points(zp$center_lon, zp$center_lat, pch = 16, cex = 1.1, col = "#2471A3")
# fourth panel: far/near ring density bars (rings 1-20 of this track point)
graphics::par(mar = c(4.5, 4.5, 2.5, 1), las = 1)
ring_dens <- ty_ring$mat[i_zoom, ]
graphics::barplot(ring_dens, names.arg = seq_len(cfg_ty$K),
        col = grDevices::adjustcolor("#2471A3", 0.65),
        xlab = "Distance ring (10-km bins)", ylab = "Persons/km2",
        main = sprintf("Ring density, track point %s", zp$event_id),
        cex.names = 0.65, cex.axis = 0.8, cex.lab = 0.9)
grDevices::dev.off()
cat("[3/3] FigS2_ty_ring_population.png 完成\n")
cat("输出文件:\n")
print(list.files(fig_dir, pattern = "Fig3_spatial_split|FigS1|FigS2"))
