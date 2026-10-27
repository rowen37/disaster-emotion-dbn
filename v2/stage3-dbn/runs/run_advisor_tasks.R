# =====================================================================
# 导师六项要求中的计算类任务：1 空值分布图 / 2 无耦合对照 / 5 图4图例 / 6 跨台风重叠
# 运行:  Rscript runs/run_advisor_tasks.R
# =====================================================================
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1) stop("请使用 Rscript runs/<脚本名>.R 运行")
script_file <- normalizePath(sub("^--file=", "", script_arg))
source(file.path(dirname(script_file), "..", "R", "00_init.R"))
log_line <- function(...) cat(format(Sys.time(), "[%H:%M:%S]"), ..., "\n"); flush.console()

dat_all <- list(
  earthquake = readRDS(file.path(paths$legacy_rds, "earthquake_st_data.rds")),
  landslide = readRDS(file.path(paths$legacy_rds, "landslide_st_data.rds")),
  debris_flow = readRDS(file.path(paths$legacy_rds, "debris_flow_st_data.rds")),
  typhoon = readRDS(file.path(paths$legacy_rds, "typhoon_st_data.rds")))
draws_all <- setNames(lapply(names(dat_all), function(h) readRDS(file.path(paths$out_dir,
  if (h == "typhoon") "typhoon_baseline_w1_draws.rds" else paste0(h, "_baseline_w1_draws.rds")))),
  names(dat_all))

# ============ 任务1：帖子空值分布图（每灾种 环×时间 观测格占比热图）============
log_line("任务1: 空值分布图")
pdf(file.path(paths$fig_dir, "Missingness_by_ring_time.pdf"), width = 12, height = 10)
par(mfrow = c(2, 2), mar = c(4, 4, 2.5, 1), las = 1)
cols_hz <- c(earthquake = "#C0392B", typhoon = "#2471A3",
             landslide = "#1E8449", debris_flow = "#B7950B")
for (h in names(dat_all)) {
  d <- dat_all[[h]]
  cfg <- hazard_configs[[h]]
  obs <- d %>% dplyr::group_by(dist_bin, time_bin) %>%
    dplyr::summarise(p_obs = mean(!is.na(E_obs)), .groups = "drop") %>%
    tidyr::pivot_wider(names_from = time_bin, values_from = p_obs)
  m <- as.matrix(obs[, -1]); rownames(m) <- obs$dist_bin
  hm <- t(m[, as.character(1:N_TIME_BINS)])
  image(x = time_bin_centers(), y = ring_centers(cfg), z = hm,
        col = grDevices::colorRampPalette(c("white", cols_hz[h]))(20),
        xlab = "Hours relative to event", ylab = "Distance from center (km)",
        main = sprintf("%s: %.0f%% cells observed", h, 100 * mean(!is.na(d$E_obs))),
        cex.main = 0.9)
}
dev.off()
cat("  saved Missingness_by_ring_time.pdf\n")

# ============ 任务2：无耦合对照（θ₁=θ₂=0），行为拟合对比 ============
log_line("任务2: 无耦合对照拟合 x4")
rmse_f <- function(o, f) sqrt(mean((o - f)^2, na.rm = TRUE))
cmp_rows <- list()
for (h in names(dat_all)) {
  hazard_configs[[h]]$has_coupling <- 0L
  res0 <- fit_hazard(h, variant = "baseline", use_weights = 1, dat = dat_all[[h]])
  hazard_configs[[h]]$has_coupling <- NULL
  d0 <- res0$draws
  saveRDS(d0, file.path(paths$out_dir, paste0(h, "_nocoupling_w1_draws.rds")))

  cfg <- hazard_configs[[h]]
  cmp <- data.frame(hazard = h, theta1_coupled = mean(draws_all[[h]]$theta1))
  for (side in c("coupled", "uncoupled")) {
    d <- if (side == "coupled") draws_all[[h]] else d0
    tt <- temporal_trajectory(d, cfg)
    ob <- obs_temporal(dat_all[[h]])
    mrg <- merge(data.frame(time_hour = time_bin_centers(),
                            B_fit = colMeans(tt$prob_b)), ob, by = "time_hour")
    cmp[[paste0("B_RMSE_temp_", side)]] <- rmse_f(mrg$mean_B, mrg$B_fit)
    sd_curve <- spatial_decay(d, cfg)
    b_fit <- plogis(colMeans(sd_curve$logit_b))
    obs_b <- obs_spatial(dat_all[[h]], "B_obs")
    cmp[[paste0("B_RMSE_spat_", side)]] <- rmse_f(obs_b, b_fit)
  }
  cmp_rows[[h]] <- cmp
  log_line(sprintf("  %s: 行为时间RMSE %.4f -> %.4f | 空间RMSE %.4f -> %.4f",
    h, cmp$B_RMSE_temp_coupled, cmp$B_RMSE_temp_uncoupled,
    cmp$B_RMSE_spat_coupled, cmp$B_RMSE_spat_uncoupled))
}
write.csv(do.call(rbind, cmp_rows),
          file.path(paths$out_dir, "nocoupling_comparison.csv"), row.names = FALSE)

# ============ 任务5：图4 图例修正（均匀选环 + 边界核对）============
log_line("任务5: 图4 图例修正")
# 在内存中替换 make_paper_outputs.R 的选环行后执行（不改写原文件）
s <- readLines(file.path(dirname(script_file), "make_paper_outputs.R"))
s <- gsub("show <- unique\\(round\\(seq\\(1, K, length.out = 5\\)\\)\\)",
          "show <- if (K == 10) c(1, 3, 5, 7, 10) else c(1, 5, 10, 15, 20)", s)
eval(parse(text = paste(s, collapse = "\n")), envir = environment())
log_line("  图4已按 c(1,3,5,7,10)/c(1,5,10,15,20) 重生成，图例=环定义边界")

# ============ 任务6：跨台风帖子重叠比例 ============
log_line("任务6: 跨台风重叠")
files <- list.files(file.path(paths$raw_input, "typhoon"), pattern = "[.]csv$", full.names = TRUE)
reads <- lapply(files, read_hazard_csv)
names(reads) <- tools::file_path_sans_ext(basename(files))
keys <- lapply(reads, function(x) paste(x$created_at, x$text, x$lon, x$lat, sep = "|"))
allk <- table(unlist(keys))
multi <- names(allk[allk > 1])
# 同台风内多路径点 vs 跨台风
tyr_of <- function(pid) sub("-[0-9]+$", "", pid)
cross <- 0L; within <- 0L
pair_cnt <- table(NA)
for (k in multi) {
  owners <- names(keys)[sapply(keys, function(x) k %in% x)]
  tys <- unique(tyr_of(owners))
  if (length(tys) > 1) {
    cross <- cross + 1L
    for (i in 1:(length(tys) - 1)) for (j in (i + 1):length(tys)) {
      pk <- paste(sort(c(tys[i], tys[j])), collapse = "-")
      pair_cnt[pk] <- pair_cnt[pk] + 1L }
  } else within <- within + 1L
}
tot <- sum(sapply(keys, length)); uni <- length(unique(unlist(keys)))
log_line(sprintf("  总行 %d | 唯一帖 %d | 同台风内多路径点 %d | 跨台风重叠 %d (%.2f%% of 唯一帖)",
                 tot, uni, within, cross, 100 * cross / uni))
log_line("  跨台风对（重叠帖数，降序前8）:")
print(sort(pair_cnt[pair_cnt > 0], decreasing = TRUE)[1:8])
saveRDS(list(total_rows = tot, unique_posts = uni,
             within_typhoon = within, cross_typhoon = cross,
             pairs = pair_cnt),
        file.path(paths$out_dir, "typhoon_overlap_stats.rds"))
log_line("全部完成")
