script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1) stop("请使用 Rscript runs/<脚本名>.R 运行")
script_file <- normalizePath(sub("^--file=", "", script_arg))
source(file.path(dirname(script_file), "..", "R", "00_init.R"))
for (hz in names(hazard_configs)) {
  for (v in c("pop", "gdp")) {
    rp <- if (v == "pop") paths$pop_raster else paths$gdp_raster
    cat(sprintf("[%s][%s] 提取中...\n", hz, v))
    flush.console()
    t0 <- Sys.time()
    m <- tryCatch(build_mod_ring(hz, rp), error = function(e) {
      cat("  FAILED:", conditionMessage(e), "\n"); NULL })
    if (is.null(m)) next
    saveRDS(m, file.path(paths$out_dir, sprintf("modring_%s_%s.rds", v, hz)))
    cat(sprintf("[%s][%s] 完成 %dx%d, %.1f min\n", hz, v, nrow(m), ncol(m),
                as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  }
}
cat("全部 mod_ring 提取完成\n")
