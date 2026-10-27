# =====================================================================
# 模型-观测一致性评估（取代旧 精确度证明/process_file_save.R 的 RMSE/MAE 部分）
# 修复点：
#   1) 旧表台风行 RMSE=MAE（数学上仅当所有|残差|相等才成立，系笔误/复制错误）
#   2) 明确比较对象：条件边际轨迹（随机效应置零）vs 格子观测均值——
#      这是拟合一致性指标而非预测精度（模型无独立测试集），
#      论文表述需相应从 prediction accuracy 改为 model-observation consistency
#   3) 断言 RMSE >= MAE（恒等式 sqrt(mean(r^2)) >= mean(|r|) 的数值校验）
# =====================================================================

# 对单个灾种：时间维度（20窗）与空间维度（K环）上 模型曲线 vs 观测均值
eval_hazard <- function(draws, dat, cfg) {
  # ---- 时间维度 ----
  tt <- temporal_trajectory(draws, cfg)
  obs_t <- obs_temporal(dat)
  cmp_t <- dplyr::left_join(
    data.frame(time_hour = tt$time_hour,
               E_fit = colMeans(tt$prob_e),
               B_fit = colMeans(tt$prob_b)),
    obs_t, by = "time_hour")

  metrics <- function(o, f, dim) {
    r <- o - f
    data.frame(
      dimension = dim,
      RMSE = sqrt(mean(r^2, na.rm = TRUE)),
      MAE = mean(abs(r), na.rm = TRUE),
      n = sum(!is.na(r)))
  }
  out <- rbind(
    metrics(cmp_t$mean_E, cmp_t$E_fit, "temporal_emotion"),
    metrics(cmp_t$mean_B, cmp_t$B_fit, "temporal_behavior"))

  # ---- 空间维度 ----
  sd_curve <- spatial_decay(draws, cfg)
  obs_e <- obs_spatial(dat, "E_obs"); obs_b <- obs_spatial(dat, "B_obs")
  e_fit <- plogis(colMeans(sd_curve$logit_e))   # 概率尺度
  b_fit <- plogis(colMeans(sd_curve$logit_b))
  out <- rbind(out,
    metrics(obs_e, e_fit, "spatial_emotion"),
    metrics(obs_b, b_fit, "spatial_behavior"))

  # ---- 恒等式校验（第一要义：数学不变式必须成立）----
  stopifnot(all(out$RMSE + 1e-12 >= out$MAE))
  out
}

# fits: named list，名字为灾种键（hazard_configs 的键），
#        每项 list(draws = <draws 或 rds 路径>, dat = <st_data>)
eval_all <- function(fits) {
  do.call(rbind, lapply(names(fits), function(hz) {
    f <- fits[[hz]]
    d <- if (is.character(f$draws)) readRDS(f$draws) else f$draws
    res <- eval_hazard(d, f$dat, hazard_configs[[hz]])
    res$hazard <- hz
    res
  }))
}
