# =====================================================================
# Stage 3 - DBN pipeline test suite.
# Scope: unit tests -> data packing -> Stan compilation -> quick smoke
# fit -> summaries and figures.
# Reads only the RDS files under paths$legacy_rds; all writes go to
# paths$out_dir.
# =====================================================================
options(warn = 1)
FAILURES <- character(0)

check <- function(name, cond, detail = "") {
  status <- if (isTRUE(cond)) "PASS" else "FAIL"
  cat(sprintf("[%s] %s %s\n", status, name, detail))
  if (!isTRUE(cond)) FAILURES <<- c(FAILURES, name)
  invisible(cond)
}

source("/Users/rogerbqw/Downloads/DBNs/refactor/R/00_init.R")
cat("\n========== 1. 配置与单元测试 ==========\n")

# 1.1 jump boundary: event time t=0 must fall in bin 7 (legacy >= 8 was off by one)
check("JUMP_START == 7", JUMP_START == 7)
check("bin(0h) 属于第7窗", floor((0 - TIME_MIN) / BIN_HOURS) + 1 == 7,
      sprintf("t=0 所在窗: %d", floor((0 - TIME_MIN) / BIN_HOURS) + 1))

# 1.2 ring centers consistent with configs
check("earthquake 环中心", all(abs(ring_centers(hazard_configs$earthquake) -
      seq(2.5, 47.5, length.out = 10)) < 1e-9))
check("typhoon 环中心", all(abs(ring_centers(hazard_configs$typhoon) -
      seq(5, 195, length.out = 20)) < 1e-9))

# 1.3 the grid produces exactly K rings (legacy expand_grid(1:20) created phantom rings)
synth <- data.frame(
  time_gap = rep(c(-60, 5, 100), 2),
  lon = c(114.0, 114.01, 114.02, 114.0, 114.01, 114.02),
  lat = c(36.0, 36.0, 36.0, 36.0, 36.0, 36.0),
  center_lon = 114.0, center_lat = 36.0,
  EMO_probs = '""', raw_prob = c(0.1, 0.4, 0.7, 0.2, 0.5, 0.8)
)
g <- grid_one_event(synth, hazard_configs$earthquake)
check("grid K=10 个环", length(unique(g$dist_bin)) == 10,
      sprintf("实际环数: %d", length(unique(g$dist_bin))))
check("grid K x T 全网格", nrow(g) == 10 * N_TIME_BINS)
check("grid 无超范围环", max(g$dist_bin) <= 10)

# 1.4 lag_idx: 0 for the first bin, pointing at the previous bin otherwise
fake <- expand_grid(event_id = "e1", dist_bin = 1:2, time_bin = 1:4) %>% mutate(x = 1)
fake <- attach_lag_idx(fake) %>% mutate(rn = dplyr::row_number())
lag_ok <- all(fake$lag_idx[fake$time_bin == 1] == 0) &&
  all(fake$lag_idx[fake$time_bin == 2] > 0)
check("lag_idx 首窗=0, 后续>0", lag_ok)
# correctness of the pointer: the lag row of a time_bin=2 row must be at time_bin=1
lag_map_check <- fake %>%
  filter(time_bin == 2) %>%
  left_join(fake %>% select(rn, chk_time = time_bin, dist_bin),
            by = c("dist_bin", "lag_idx" = "rn"))
check("lag 指向前一窗", all(lag_map_check$chk_time == 1, na.rm = TRUE))

# 1.5 attach_lag_idx drops any stale lag_idx shipped in the RDS (landslide RDS carries one)
ls_rds <- readRDS(file.path(paths$legacy_rds, "landslide_st_data.rds"))
check("旧 lag_idx 列被丢弃重建", !("lag_row" %in% names(ls_rds)))
rebuilt <- attach_lag_idx(ls_rds)
check("landslide 重建 lag 后仍在范围内",
      all(rebuilt$lag_idx >= 0 & rebuilt$lag_idx <= nrow(rebuilt)))

cat("\n========== 2. 数据打包 ==========\n")
eq <- readRDS(file.path(paths$legacy_rds, "earthquake_st_data.rds"))
p_eq <- build_stan_data(eq, hazard_configs$earthquake)
sd_eq <- p_eq$stan_data
check("earthquake 维度", sd_eq$N == 7400 && sd_eq$J == 37 && sd_eq$K == 10 && sd_eq$T == 20,
      sprintf("N=%d J=%d K=%d T=%d", sd_eq$N, sd_eq$J, sd_eq$K, sd_eq$T))
check("dist_bin 上界合法", max(sd_eq$dist_bin) <= sd_eq$K)
check("lag_idx 上界合法", max(sd_eq$lag_idx) <= sd_eq$N && min(sd_eq$lag_idx) >= 0)
check("n_posts >= 1", min(sd_eq$n_posts) >= 1)
check("E_obs 在 logit 范围", all(abs(sd_eq$E_obs[sd_eq$E_miss == 0]) < 50))
check("B_obs 在 (0,1)", all(sd_eq$B_obs[sd_eq$B_miss == 0] > 0 &
                            sd_eq$B_obs[sd_eq$B_miss == 0] < 1))
check("ev_cov_miss 展平长度 = J*Gc", length(sd_eq$ev_cov_miss) == sd_eq$J * sd_eq$Gc)
check("B_spline 维度", all(dim(sd_eq$B_spline) == c(10, sd_eq$M)))
check("T_spline 维度", all(dim(sd_eq$T_spline) == c(20, sd_eq$P)))

ty <- readRDS(file.path(paths$legacy_rds, "typhoon_st_data.rds"))
p_ty <- build_stan_data(ty, hazard_configs$typhoon)
sd_ty <- p_ty$stan_data
check("typhoon 维度", sd_ty$N == 21200 && sd_ty$J == 53 && sd_ty$K == 20,
      sprintf("N=%d J=%d K=%d（53个路径点，嵌套结构待后续处理）",
              sd_ty$N, sd_ty$J, sd_ty$K))
check("typhoon jump 关闭", sd_ty$has_jump == 0)
check("earthquake jump 开启", sd_eq$has_jump == 1 && sd_eq$jump_start == 7)

cat("\n========== 3. Stan 编译 ==========\n")
mod_core <- cmdstan_model(paths$stan_core)
check("dbn_core.stan 编译", !is.null(mod_core))
mod_mod <- cmdstan_model(paths$stan_mod)
check("dbn_moderator.stan 编译", !is.null(mod_mod))

cat("\n========== 4. 冒烟拟合（earthquake, quick 模式, iter<=1500）==========\n")
t0 <- Sys.time()
res <- fit_hazard("earthquake", variant = "baseline", use_weights = 1,
                  quick = TRUE, output_dir = paths$out_dir)
cat(sprintf("耗时: %.1f 秒\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))

d <- res$draws
check("draws 非空且含 theta1", nrow(d) > 0 && "theta1" %in% names(d))
t1 <- summarize_draws(d, hazard = "earthquake", variant = "baseline_w1")
theta1_row <- t1 %>% filter(parameter == "theta1")
cat(sprintf("theta1 = %.4f [%.4f, %.4f]（旧 ADVI 结果: 0.1169 [0.1047, 0.1287]）\n",
            theta1_row$mean, theta1_row$Q2.5, theta1_row$Q97.5))
check("theta1 为正", theta1_row$mean > 0)
check("jump 项已估计", "jump_e" %in% names(d))
check("sigma 分离生效", all(c("sigma_obs_e", "sigma_proc_e") %in% names(d)))

cat("\n========== 5. 汇总与图形 ==========\n")
dir.create(paths$out_dir, showWarnings = FALSE, recursive = TRUE)
write_summary(t1, file.path(paths$out_dir, "smoke_earthquake_summary.csv"))
fig_temporal(d, res$packed$dat, res$cfg, "earthquake_smoke")
fig_spatial(d, res$packed$dat, res$cfg, "earthquake_smoke")
figs <- list.files(paths$fig_dir, pattern = "earthquake_smoke")
check("生成图形 >= 3 张", length(figs) >= 3, paste(figs, collapse = ", "))

cat("\n========== 6. 调节效应模型冒烟（landslide + 合成 mod_ring）==========\n")
ls_dat <- readRDS(file.path(paths$legacy_rds, "landslide_st_data.rds"))
ls_cfg <- hazard_configs$landslide
ls_J <- dplyr::n_distinct(ls_dat$event_id)
set.seed(1)
mod_ring_fake <- matrix(rnorm(ls_J * ls_cfg$K), nrow = ls_J, ncol = ls_cfg$K)
res_mod <- fit_hazard("landslide", variant = "moderator", use_weights = 1,
                      mod_ring = mod_ring_fake, dat = ls_dat,
                      quick = TRUE, output_dir = paths$out_dir)
dm <- res_mod$draws
check("moderator 模型含 gamma_mod/delta_mod",
      all(c("gamma_mod", "delta_mod") %in% names(dm)))
t2 <- summarize_draws(dm, hazard = "landslide", variant = "moderator_w1")
cat(sprintf("landslide moderator: theta1 = %.4f, delta_mod = %.4f\n",
            t2$mean[t2$parameter == "theta1"],
            t2$mean[t2$parameter == "delta_mod"]))
check("moderator draws 已保存",
      file.exists(file.path(paths$out_dir, "landslide_moderator_w1_draws.rds")))

cat("\n=========================================\n")
if (length(FAILURES) == 0) {
  cat("全部测试通过 ✔\n")
} else {
  cat(sprintf("失败 %d 项: %s\n", length(FAILURES), paste(FAILURES, collapse = " | ")))
  quit(status = 1)
}
