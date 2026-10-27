# =====================================================================
# 跨灾种后验差检验（"置信区间相减"的正确实现）
#   四灾种基线定稿 draws 两两相减（6 组），输出差的后验均值、95% CrI、
#   P(diff>0)。四灾种数据不相交、独立拟合，后验独立，逐行配对相减合法。
#   合法性仅限跨灾种；同数据模型对照（有/无耦合、有/无 jump、规则A/B）
#   后验不独立，不可用本方法。
# 输入: output/rds/{earthquake,landslide,debris_flow}_baseline_w1_draws.rds
#       + 台风基线 draws（去重定稿）
# 输出: output/cross_hazard_cri_diff.csv
# 运行: Rscript runs/cross_hazard_cri_diff.R
# =====================================================================
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1) stop("请使用 Rscript runs/<脚本名>.R 运行")
script_file <- normalizePath(sub("^--file=", "", script_arg))
source(file.path(dirname(script_file), "..", "R", "00_init.R"))

dr_dir <- file.path(paths$out_dir, "rds")
if (!dir.exists(dr_dir)) dr_dir <- paths$out_dir   # 兼容旧布局（draws 在 output 根目录）

draws_file <- function(hz) file.path(dr_dir, if (hz == "typhoon")
  paste0(hz, "_baseline_w1_draws.rds"))

hz_order <- c("earthquake", "typhoon", "landslide", "debris_flow")
draws_all <- setNames(lapply(hz_order, function(h) readRDS(draws_file(h))), hz_order)

params <- c("theta1", "theta2", "beta_e", "beta_b")

# ---- 0) 各灾种自身后验（对账用，应与表1一致）----
cat("\n[对账] 各灾种参数后验（draws 直接汇总）:\n")
for (h in hz_order) {
  d <- draws_all[[h]]
  cat(sprintf("%-11s theta1 %.4f [%.4f, %.4f] | theta2 %+.4f [%.4f, %.4f]\n", h,
              mean(d$theta1), quantile(d$theta1, .025), quantile(d$theta1, .975),
              mean(d$theta2), quantile(d$theta2, .025), quantile(d$theta2, .975)))
}
cat(sprintf("抽样数: %s\n", paste(sapply(draws_all, nrow), collapse = "/")))

# ---- 1) 两两后验差 ----
pairs <- utils::combn(hz_order, 2, simplify = FALSE)
rows <- list()
for (pr in pairs) {
  a <- pr[1]; b <- pr[2]
  for (p in params) {
    diff_draws <- draws_all[[a]][[p]] - draws_all[[b]][[p]]
    lo <- unname(quantile(diff_draws, .025)); hi <- unname(quantile(diff_draws, .975))
    rows[[length(rows) + 1]] <- data.frame(
      comparison = paste0(a, " - ", b),
      higher = a, lower = b,
      parameter = p,
      mean_higher = mean(draws_all[[a]][[p]]),
      mean_lower = mean(draws_all[[b]][[p]]),
      diff_mean = mean(diff_draws),
      diff_Q2.5 = lo, diff_Q97.5 = hi,
      p_diff_gt0 = mean(diff_draws > 0),
      sig_diff = lo > 0 | hi < 0)
  }
}
out <- do.call(rbind, rows)
write.csv(out, file.path(paths$out_dir, "cross_hazard_cri_diff.csv"), row.names = FALSE)

cat("\n[theta1 梯度检验]（差的 95% CrI 不含 0 = 差异显著）:\n")
t1 <- out[out$parameter == "theta1", ]
for (i in seq_len(nrow(t1))) {
  r <- t1[i, ]
  cat(sprintf("  %-28s %+.4f [%+.4f, %+.4f]  P(>0)=%.3f  %s\n",
              r$comparison, r$diff_mean, r$diff_Q2.5, r$diff_Q97.5,
              r$p_diff_gt0, ifelse(r$sig_diff, "显著", "不显著")))
}
cat("\n[theta2 / beta 检验]:\n")
oth <- out[out$parameter != "theta1", ]
for (i in seq_len(nrow(oth))) {
  r <- oth[i, ]
  cat(sprintf("  %-28s %-6s %+.4f [%+.4f, %+.4f]  P(>0)=%.3f  %s\n",
              r$comparison, r$parameter, r$diff_mean, r$diff_Q2.5, r$diff_Q97.5,
              r$p_diff_gt0, ifelse(r$sig_diff, "显著", "不显著")))
}
cat(sprintf("\n已保存 %s\n", file.path(paths$out_dir, "cross_hazard_cri_diff.csv")))
