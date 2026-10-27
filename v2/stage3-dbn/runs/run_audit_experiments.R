# =====================================================================
# 审查补充实验 1 + 3（简洁原则，合计约 30 分钟）
#   1) 滑坡/泥石流 空间斜率敏感性：U 型是否成分效应
#   3) 地震无 jump 对照：jump 与时间样条是否可分
# 运行:  Rscript runs/run_audit_experiments.R
# =====================================================================
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1) stop("请使用 Rscript runs/<脚本名>.R 运行")
script_file <- normalizePath(sub("^--file=", "", script_arg))
source(file.path(dirname(script_file), "..", "R", "00_init.R"))
log_line <- function(...) cat(format(Sys.time(), "[%H:%M:%S]"), ..., "\n"); flush.console()

fmt3 <- function(d, p) if (p %in% names(d))
  sprintf("%.4f [%.4f, %.4f]", mean(d[[p]]), quantile(d[[p]], .025), quantile(d[[p]], .975)) else "无"

# ================= 实验 1：LS/DF 斜率敏感性 =================
for (hz in c("landslide", "debris_flow")) {
  log_line(hz, "斜率敏感性拟合")
  dat <- readRDS(file.path(paths$legacy_rds, paste0(hz, "_st_data.rds")))
  hazard_configs[[hz]]$has_slope <- 1L
  res <- fit_hazard(hz, variant = "baseline", use_weights = 1, dat = dat)
  hazard_configs[[hz]]$has_slope <- NULL
  d <- res$draws
  saveRDS(d, file.path(paths$out_dir, paste0(hz, "_slope_w1_draws.rds")))

  d_base <- readRDS(file.path(paths$out_dir, paste0(hz, "_baseline_w1_draws.rds")))
  cfg <- hazard_configs[[hz]]
  sb <- spatial_decay(d_base, cfg); ss <- spatial_decay(d, cfg)
  eb <- colMeans(sb$logit_e); es <- colMeans(ss$logit_e)
  log_line(sprintf("  theta1: 基线 %s -> 斜率 %s", fmt3(d_base, "theta1"), fmt3(d, "theta1")))
  log_line(sprintf("  tau_slope_e: %s  tau_slope_b: %s", fmt3(d, "tau_slope_e"), fmt3(d, "tau_slope_b")))
  log_line(sprintf("  情绪曲线回升(末-谷): 基线 %.3f -> 斜率 %.3f",
                   eb[cfg$K] - min(eb), es[cfg$K] - min(es)))
  log_line(sprintf("  基线曲线: %s", paste(sprintf("%.3f", eb), collapse = " ")))
  log_line(sprintf("  斜率曲线: %s", paste(sprintf("%.3f", es), collapse = " ")))
}

# ================= 实验 3：地震无 jump 对照 =================
log_line("earthquake 无 jump 对照拟合")
dat <- readRDS(file.path(paths$legacy_rds, "earthquake_st_data.rds"))
hazard_configs$earthquake$has_jump <- 0L
res_nj <- fit_hazard("earthquake", variant = "baseline", use_weights = 1, dat = dat)
hazard_configs$earthquake$has_jump <- 1L
d_nj <- res_nj$draws
saveRDS(d_nj, file.path(paths$out_dir, "earthquake_nojump_w1_draws.rds"))

d_b <- readRDS(file.path(paths$out_dir, "earthquake_baseline_w1_draws.rds"))
log_line(sprintf("  theta1: 有jump %s -> 无jump %s", fmt3(d_b, "theta1"), fmt3(d_nj, "theta1")))
log_line(sprintf("  jump_e(有jump版): %s", fmt3(d_b, "jump_e")))

# 时间曲线对照：无 jump 版曲线是否仍出现事件后尖峰
cfg0 <- hazard_configs$earthquake; cfg0$has_jump <- 0L
tt_nj <- temporal_trajectory(d_nj, cfg0)
tt_b <- temporal_trajectory(d_b, hazard_configs$earthquake)
th <- time_bin_centers()
log_line(sprintf("  有jump曲线: %s", paste(sprintf("%.3f", colMeans(tt_b$prob_e)), collapse = " ")))
log_line(sprintf("  无jump曲线: %s", paste(sprintf("%.3f", colMeans(tt_nj$prob_e)), collapse = " ")))
log_line(sprintf("  有jump: 峰 %+dh (%.3f) | 无jump: 峰 %+dh (%.3f)",
                 th[which.max(colMeans(tt_b$prob_e))], max(colMeans(tt_b$prob_e)),
                 th[which.max(colMeans(tt_nj$prob_e))], max(colMeans(tt_nj$prob_e))))
# 样条能否自造尖峰：无jump曲线在 t=7 处的跳变幅度
pe_nj <- colMeans(tt_nj$prob_e)
log_line(sprintf("  无jump曲线 t7 前后变化: %.3f -> %.3f (+%.3f)",
                 pe_nj[6], pe_nj[7], pe_nj[7] - pe_nj[6]))
log_line("全部完成")
