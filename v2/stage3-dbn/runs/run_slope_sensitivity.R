# =====================================================================
# 敏感性分析：事件级空间随机斜率（检验"远环回升"是否成分效应）
# 逻辑：若共享样条的远环回升来自"某些事件的远环恰好覆盖大城市"（成分效应），
#       允许每个事件有自己的衰减斜率后，共享曲线的回升应收缩（tau_slope 吸收）。
# 对照组：同数据基线（earthquake_baseline_v2）
# 运行:  Rscript runs/run_slope_sensitivity.R
# =====================================================================
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1) stop("请使用 Rscript runs/<脚本名>.R 运行")
script_file <- normalizePath(sub("^--file=", "", script_arg))
source(file.path(dirname(script_file), "..", "R", "00_init.R"))
log_line <- function(...) cat(format(Sys.time(), "[%H:%M:%S]"), ..., "\n"); flush.console()

dat <- readRDS(file.path(paths$legacy_rds, "earthquake_st_data.rds"))

# ---- 1) 斜率模型拟合 ----
hazard_configs$earthquake$has_slope <- 1L
res_s <- fit_hazard("earthquake", variant = "baseline", use_weights = 1,
                    dat = dat, output_dir = paths$out_dir)
hazard_configs$earthquake$has_slope <- NULL
d <- res_s$draws

summ <- summarize_draws(d, hazard = "earthquake", variant = "slope_full_w1")
write_summary(summ, file.path(paths$out_dir, "earthquake_slope_summary.csv"))

t1 <- function(p) sprintf("%.4f [%.4f, %.4f]",
  mean(d[[p]]), quantile(d[[p]], .025), quantile(d[[p]], .975))
log_line("theta1(斜率模型):", t1("theta1"))
log_line("tau_slope_e:", t1("tau_slope_e"), " tau_slope_b:", t1("tau_slope_b"))

# ---- 2) 共享空间曲线：基线 vs 斜率模型（回升是否收缩） ----
d_base <- readRDS(file.path(paths$out_dir, "earthquake_baseline_w1_draws.rds"))
cfg <- hazard_configs$earthquake
sd_base <- spatial_decay(d_base, cfg)
sd_slope <- spatial_decay(d, cfg)
dc <- ring_centers(cfg)
cmp_curve <- data.frame(
  dist_km = dc,
  base_logit_e = colMeans(sd_base$logit_e),
  slope_logit_e = colMeans(sd_slope$logit_e),
  base_logit_b = colMeans(sd_base$logit_b),
  slope_logit_b = colMeans(sd_slope$logit_b))
cmp_curve$base_rebound_e <- cmp_curve$base_logit_e - min(cmp_curve$base_logit_e)
cmp_curve$slope_rebound_e <- cmp_curve$slope_logit_e - min(cmp_curve$slope_logit_e)
print(round(cmp_curve, 3))
write.csv(cmp_curve, file.path(paths$out_dir, "slope_sensitivity_curve.csv"), row.names = FALSE)
log_line(sprintf("情绪回升幅度(末环-最低环): 基线 %.3f -> 斜率模型 %.3f",
                 tail(cmp_curve$base_rebound_e, 1), tail(cmp_curve$slope_rebound_e, 1)))

# ---- 3) 成分效应描述性检验：事件级远环比 vs 远环人口占比 ----
pop <- readRDS(file.path(paths$out_dir, "modring_pop_earthquake.rds"))
ev_levels <- levels(as.factor(as.character(dat$event_id)))   # 与打包事件序一致
obs_spatial_ev <- dat %>%
  dplyr::filter(!is.na(E_obs)) %>%
  dplyr::group_by(event_id, dist_bin) %>%
  dplyr::summarise(m = mean(E_obs), .groups = "drop") %>%
  tidyr::pivot_wider(names_from = dist_bin, values_from = m)
near <- rowMeans(obs_spatial_ev[, c("1", "2")], na.rm = TRUE)
far <- rowMeans(obs_spatial_ev[, c("9", "10")], na.rm = TRUE)
ids <- obs_spatial_ev$event_id
pop_near <- pop[match(ids, ev_levels), 1:2] |> (\(m) rowMeans(m, na.rm=TRUE))()
pop_far <- pop[match(ids, ev_levels), 9:10] |> (\(m) rowMeans(m, na.rm=TRUE))()
desc <- data.frame(event_id = ids, far_near_ratio = far / near,
                   pop_far_share = pop_far / (pop_far + pop_near))
ok <- is.finite(desc$far_near_ratio) & is.finite(desc$pop_far_share)
ct <- cor.test(desc$far_near_ratio[ok], desc$pop_far_share[ok])
log_line(sprintf("成分检验: 事件远/近观测比 与 远环人口占比的相关 r=%.3f (p=%.4f, n=%d)",
                 ct$estimate, ct$p.value, sum(ok)))
log_line("完成")
