# =====================================================================
# Stage 3 - typhoon subset test: validate on a small sample before any
# full run. Three arms isolate the contribution of each fix:
#   A legacy data (output_test, duplicates) + flat model -- reproduces
#     the pseudo-replication state
#   B deduplicated data + flat model (has_group=0) -- dedup effect only
#   C deduplicated data + nested model (has_group=1) -- both (official)
# Subset: typhoons 401(6 pts) + 402(5) + 405(1) + 409(4) = 16 points, 4 groups
# =====================================================================
source("/Users/rogerbqw/Downloads/DBNs/refactor/R/00_init.R")

SUBSET <- c("401", "402", "405", "409")

cat("\n========== 1. 去重 ==========\n")
st_dedup <- dedup_typhoon(typhoons = SUBSET)
saveRDS(st_dedup, file.path(paths$out_dir, "typhoon_subset_dedup_st_data.rds"))

cat("\n========== 2. 结构校验 ==========\n")
stopifnot(
  all(st_dedup$event_id %in% c(
    sprintf("401-%d", 1:6), sprintf("402-%d", 1:5), "405-1", sprintf("409-%d", 1:4))),
  all(as.character(st_dedup$typhoon_id) %in% SUBSET)
)
cat("路径点数:", length(unique(st_dedup$event_id)),
    " 组数:", length(unique(st_dedup$typhoon_id)),
    " 维度:", nrow(st_dedup), "\n")
cat("每事件 400 格:", all(table(st_dedup$event_id) == 400), "\n")
# post totals vs. the legacy RDS on the same subset
old_sub <- readRDS(file.path(paths$legacy_rds, "typhoon_st_data.rds")) %>%
  filter(typhoon_of(as.character(event_id)) %in% SUBSET)
cat(sprintf("帖子总量: 旧(重复计入) %d -> 去重后 %d\n",
            sum(old_sub$N, na.rm = TRUE), sum(st_dedup$N, na.rm = TRUE)))

cat("\n========== 3. 打包校验（嵌套开关）=========\n")
packed <- build_stan_data(st_dedup, hazard_configs$typhoon)
sd_n <- packed$stan_data
stopifnot(sd_n$has_group == 1, sd_n$G == 4,
          length(sd_n$group_of_event) == sd_n$J,
          all(sd_n$group_of_event >= 1), all(sd_n$group_of_event <= 4))
cat("has_group:", sd_n$has_group, " G:", sd_n$G,
    " J:", sd_n$J, " N:", sd_n$N,
    " group_of_event:", paste(sd_n$group_of_event, collapse = ""), "\n")
# flat packing (arm B): drop group_col temporarily
cfg_flat <- hazard_configs$typhoon; cfg_flat$group_col <- NULL
sd_f <- build_stan_data(st_dedup, cfg_flat)$stan_data
stopifnot(sd_f$has_group == 0, sd_f$G == 1)
cat("平模型: has_group =", sd_f$has_group, "\n")

cat("\n========== 4. 三组 quick 拟合 ==========\n")
t1 <- function(d, p) sprintf("%.4f [%.4f, %.4f]",
  mean(d[[p]]), quantile(d[[p]], .025), quantile(d[[p]], .975))

# A: legacy (pseudo-replicated) data, flat model (no typhoon_id column -> no nesting)
cat("[A] 旧数据 + 平模型...\n")
resA <- fit_hazard("typhoon", dat = old_sub, quick = TRUE)
# B: deduplicated + flat (nesting switch off)
cat("[B] 去重 + 平模型...\n")
saved_gc <- hazard_configs$typhoon$group_col
hazard_configs$typhoon$group_col <- NULL
resB <- fit_hazard("typhoon", dat = st_dedup, quick = TRUE)
hazard_configs$typhoon$group_col <- saved_gc
# C: deduplicated + nested (official setting; group_col on by default)
cat("[C] 去重 + 嵌套...\n")
resC <- fit_hazard("typhoon", dat = st_dedup, quick = TRUE)

for (nm in c("theta1", "theta2", "beta_e", "beta_b", "phi_b_beta")) {
  cat(sprintf("%-10s A: %-26s B: %-26s C: %s\n", nm,
    t1(resA$draws, nm), t1(resB$draws, nm), t1(resC$draws, nm)))
}
cat("tau_v_e (C):", t1(resC$draws, "tau_v_e"), " tau_w_e (C):", t1(resC$draws, "tau_w_e"), "\n")
cat("tau_v_b (C):", t1(resC$draws, "tau_v_b"), " tau_w_b (C):", t1(resC$draws, "tau_w_b"), "\n")

cat("\n========== 5. 僵尸参数检查（第一要义）=========\n")
# free-parameter blocks: conditional parameters of size 0 never appear in
# draws; transformed quantities like jump_e/tau_v_e that are identically 0
# are deterministic constants, not zombie parameters
cat("C 含 jump_raw(自由参数):", any(grepl("^jump_raw", names(resC$draws))), "（应为 FALSE）\n")
cat("B 含 z_v/tau_v 自由参数:", any(grepl("z_v_|tau_v_.*arr", names(resB$draws))), "（应为 FALSE）\n")
cat("C jump_e 恒为0:", all(resC$draws$jump_e == 0), "（transformed 常量，应为 TRUE）\n")

cat("\n子集测试完成\n")
