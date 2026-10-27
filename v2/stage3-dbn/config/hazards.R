# =====================================================================
# 单一事实来源：灾种配置 + 拟合设置
# 所有脚本从这里读参数，禁止在代码里散落 magic number
# =====================================================================

TIME_MIN   <- -72   # 事件前小时数
TIME_MAX   <- 168   # 事件后小时数
BIN_HOURS  <- 12    # 时间窗宽度
N_TIME_BINS <- (TIME_MAX - TIME_MIN) %/% BIN_HOURS   # 20

# 第一个"事件后"时间窗的索引：time_gap=0 落在 bin (0 - TIME_MIN)/BIN_HOURS + 1
# 修正旧代码 op.stan 中 t >= 8 的 off-by-one（旧注释误以为 t=8 是 [0,12) 窗）
JUMP_START <- (0 - TIME_MIN) %/% BIN_HOURS + 1          # = 7，即 [0, +12) 窗
stopifnot(JUMP_START == 7)

hazard_configs <- list(
  earthquake = list(
    label = "earthquake", K = 10, ring_km = 5, max_dist_km = 50,
    df_dist = 6, df_time = 8, has_jump = 1,
    ev_covs = c("mag", "depth")
  ),
  landslide = list(
    label = "landslide", K = 10, ring_km = 5, max_dist_km = 50,
    df_dist = 6, df_time = 8, has_jump = 0,
    ev_covs = c("volume", "max_rain")
  ),
  debris_flow = list(
    label = "debris_flow", K = 10, ring_km = 5, max_dist_km = 50,
    df_dist = 6, df_time = 8, has_jump = 0,
    ev_covs = c("volume", "max_rain")
  ),
  typhoon = list(
    label = "typhoon", K = 20, ring_km = 10, max_dist_km = 200,
    df_dist = 5, df_time = 8, has_jump = 0,
    ev_covs = c("typhoon_level"),
    group_col = "typhoon_id"   # 嵌套层：路径点(事件) 嵌套于 台风(组)
  )
)

# 拟合设置（定稿）：tol_rel_obj=0.01（ELBO 相对改善<1% 的平台标准），
# iter 上限 30000（用户定稿；20k 时多数拟合未达 0.01 平台）
fit_settings <- list(
  iter          = 30000,
  grad_samples  = 5,
  elbo_samples  = 200,
  output_samples = 1000,
  eta           = 0.1,
  tol_rel_obj   = 0.01,
  seed          = 123
)

# 路径集中管理
# 原始微博与外部资源位于本机归档目录；模型输出一律写入本目录
resource_dir <- "/Users/rogerbqw/Downloads/毕设数据/Thesis-ZJU/DBNs"
paths <- list(
  raw_input   = "/Users/rogerbqw/Downloads/毕设数据/Thesis-ZJU/weibo_dbn/input_v2_v1layout",
  legacy_rds  = file.path(stage3_root, "data"),
  event_feat  = file.path(resource_dir, "event_features.csv"),
  pop_raster  = file.path(resource_dir, "chn_pd_2020_1km.tif"),
  gdp_raster  = file.path(resource_dir, "gdp2020.tif"),
  stan_core   = file.path(stage3_root, "stan", "dbn_core.stan"),
  stan_mod    = file.path(stage3_root, "stan", "dbn_moderator.stan"),
  out_dir     = file.path(stage3_root, "output"),
  fig_dir     = file.path(stage3_root, "output", "figures")
)

# 距离环中心：seq(ring_km/2, max - ring_km/2, length.out = K)
# K=10, 5km 环 -> 2.5, 7.5, ..., 47.5；K=20, 10km 环 -> 5, 15, ..., 195
ring_centers <- function(cfg) {
  seq(cfg$ring_km / 2, cfg$max_dist_km - cfg$ring_km / 2, length.out = cfg$K)
}

time_bin_centers <- function() {
  TIME_MIN + (seq_len(N_TIME_BINS) - 0.5) * BIN_HOURS   # -66, -54, ..., 162
}
