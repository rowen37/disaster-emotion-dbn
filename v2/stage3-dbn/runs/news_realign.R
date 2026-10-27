library(readxl); library(dplyr)
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1) stop("请使用 Rscript runs/<脚本名>.R 运行")
script_file <- normalizePath(sub("^--file=", "", script_arg))
source(file.path(dirname(script_file), "..", "R", "00_init.R"))

# ---- 1) 模型行为窗口（30k 曲线，>=90% 峰值的连续区间）----
win <- list()
for (h in c("earthquake","typhoon","landslide","debris_flow")) {
  d <- readRDS(file.path(paths$out_dir, if (h=="typhoon") "typhoon_baseline_w1_draws.rds" else paste0(h,"_baseline_w1_draws.rds")))
  tt <- temporal_trajectory(d, hazard_configs[[h]])
  pb <- colMeans(tt$prob_b); th <- time_bin_centers()
  idx <- which(pb >= 0.9 * max(pb))
  # 取包含峰值的连续段
  pk <- which.max(pb); run <- pk
  while (run[1]-1 %in% idx) run <- c(run[1]-1, run)
  while (tail(run,1)+1 %in% idx) run <- c(run, tail(run,1)+1)
  win[[h]] <- c(lo = th[min(run)] - 6, hi = th[max(run)] + 6)  # 窗端各扩半格
  cat(sprintf("%-11s 行为窗口: [%+.0fh, %+.0fh] (峰 %+dh)\n", h,
              win[[h]]["lo"], win[[h]]["hi"], th[pk]))
}

# ---- 2) 新闻事件级去重 (URL × 绝对行为时间) ----
d <- read_excel("/Users/rogerbqw/Downloads/毕设数据/数据文件夹/灾害数据表格版/外部新闻验证.xlsx",
                sheet = "all_events_with_groups")
d$type3 <- ifelse(grepl("^4", as.character(d$event_id)), "typhoon",
           ifelse(grepl("^1", as.character(d$event_id)), "earthquake",
           ifelse(grepl("^2", as.character(d$event_id)), "landslide", "debris_flow")))
rec <- list()
for (i in seq_len(nrow(d))) {
  if (d$has_news[i] != 1) next
  urls <- as.character(unlist(d[i, grep("^链接", names(d))]))
  urls <- urls[!is.na(urls) & urls != "NA"]
  rts <- suppressWarnings(as.numeric(unlist(d[i, grep("^相对时间", names(d))])))
  abs1 <- d$`行为1时间`[i]; abs2 <- d$`行为2时间`[i]
  for (k in seq_along(rts)) {
    if (!is.finite(rts[k])) next
    at <- if (k == 1) abs1 else if (k == 2) abs2 else NA
    rec[[length(rec)+1]] <- data.frame(
      event_id = as.character(d$event_id[i]), type = d$type3[i],
      url = if (length(urls) >= k) urls[k] else if (length(urls) > 0) urls[1] else NA,
      abs_time = at, rel = rts[k])
  }
}
rec <- bind_rows(rec)
rec <- rec[!is.na(rec$url) & is.finite(rec$rel), ]
cat("\n事件×行为记录:", nrow(rec), "\n")
# 去重键: (url, abs_time)；同键多路径点取 |rel| 最小（最近通过点）
inc <- rec %>%
  mutate(abs_key = ifelse(is.na(abs_time), paste("r", round(rel, 1)), as.character(abs_time))) %>%
  group_by(url, abs_key) %>%
  arrange(abs(rel), .by_group = TRUE) %>%
  slice(1) %>% ungroup()
cat("去重后新闻行为事件:", nrow(inc), "\n")
saveRDS(inc, file.path(paths$out_dir, "news_incidents_dedup.rds"))

# ---- 3) 与新模型窗口对齐 ----
cat("\n=== 对齐结果（新窗口）===\n")
for (h in c("earthquake","typhoon","landslide","debris_flow")) {
  s <- inc[inc$type == h, ]
  if (nrow(s) == 0) { cat(sprintf("%-11s 无新闻\n", h)); next }
  inwin <- s$rel >= win[[h]]["lo"] & s$rel <= win[[h]]["hi"]
  cat(sprintf("%-11s n=%d 中位 %+.1fh | 窗内 %d/%d (%.0f%%) | rel范围 [%.0f, %.0f]\n",
      h, nrow(s), median(s$rel), sum(inwin), nrow(s), 100*mean(inwin),
      min(s$rel), max(s$rel)))
}
