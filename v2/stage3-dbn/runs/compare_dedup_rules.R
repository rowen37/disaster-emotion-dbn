script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1) stop("请使用 Rscript runs/<脚本名>.R 运行")
script_file <- normalizePath(sub("^--file=", "", script_arg))
source(file.path(dirname(script_file), "..", "R", "00_init.R"))
# 重放解析 + 两种规则
files <- list.files(file.path(paths$raw_input, "typhoon"), pattern = "[.]csv$", full.names = TRUE)
reads <- lapply(files, read_hazard_csv)
allp <- data.table::rbindlist(reads, idcol = "fidx")
fmap <- data.frame(fidx = seq_along(files),
                   path_id = tools::file_path_sans_ext(basename(files)))
allp <- merge(allp, fmap, by = "fidx", all.x = TRUE)
allp[, key := paste(created_at, text, lon, lat, sep = "|")]
allp[, ty := sub("-[0-9]+$", "", path_id)]

# 规则A（现行代码）：全局最近路径点
setorder(allp, key, dist_to_center)
ruleA <- allp[!duplicated(key)]
# 规则B（文档声称）：台风内最近路径点（跨台风保留于各台风）
setorder(allp, ty, key, dist_to_center)
ruleB <- allp[!duplicated(paste(ty, key))]

cat(sprintf("规则A(全局): %d 帖 | 规则B(台风内): %d 帖 | 差 = 跨台风重复计入 %d\n",
            nrow(ruleA), nrow(ruleB), nrow(ruleB) - nrow(ruleA)))
# 401 与 410 在两规则下的帖子数
for (t in c("401", "410", "412", "404")) {
  cat(sprintf("  台风%s: A=%d -> B=%d (%+.0f%%)\n", t,
              ruleA[ty == t, .N], ruleB[ty == t, .N],
              100 * (ruleB[ty == t, .N] / ruleA[ty == t, .N] - 1)))
}
