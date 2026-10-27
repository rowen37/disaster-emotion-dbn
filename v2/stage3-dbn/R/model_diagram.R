# =====================================================================
# 新旧 DBN 模型结构对比图
# 输出: stage3-dbn/output/figures/dbn_structure_comparison.png
# 骨架（自由潜变量 + 结构化先验均值）不变；差异在噪声结构/观测加权/jump 边界
# =====================================================================
dir.create(paths$fig_dir, showWarnings = FALSE, recursive = TRUE)
outfile <- file.path(paths$fig_dir, "dbn_structure_comparison.png")

FAM <- "PingFangSC-Regular"
grDevices::png(outfile, width = 2400, height = 1400, res = 160, type = "quartz",
               family = FAM)
par(mar = rep(0.2, 4), family = FAM)
plot(NA, xlim = c(0, 22), ylim = c(0, 11.4), xlab = "", ylab = "", axes = FALSE)

COL_CUR  <- "#D6EAF8"   # 当前窗潜变量
COL_PREV <- "#E5E8E8"   # t-1 窗（弱化）
COL_MU   <- "#FEF9E7"   # 结构均值
COL_OBS  <- "#FFFFFF"
RED  <- "#C0392B"
GRN  <- "#1E8449"
BLU  <- "#1F618D"
GRAY <- "#7F8C8D"

box <- function(cx, cy, w, h, label, fill = "white", border = "gray40",
                lwd = 1.2, cex = 1.0, col = "black", font = 1) {
  graphics::rect(cx - w/2, cy - h/2, cx + w/2, cy + h/2,
                 col = fill, border = border, lwd = lwd)
  graphics::text(cx, cy, label, cex = cex, col = col, font = font)
}

arrow <- function(x0, y0, x1, y1, col = "gray30", lwd = 1.3, lty = 1, len = 0.14) {
  graphics::arrows(x0, y0, x1, y1, length = len, col = col, lwd = lwd, lty = lty)
}

draw_panel <- function(ox, title, subtitle, old = TRUE) {
  hl <- if (old) RED else GRN
  graphics::text(ox + 5, 10.55, title, cex = 1.35, font = 2)
  graphics::text(ox + 5, 10.12, subtitle, cex = 0.92, col = GRAY)

  # ---- 外生结构项 ----
  graphics::rect(ox + 0.15, 8.7, ox + 9.85, 9.62, col = "#FBFCFC", border = GRAY, lwd = 0.9)
  exo <- c("α_e, α_b", "时间样条 s(t)", "空间样条 f(k)", "事件截距 u_j", "事件协变量 γ·X",
           if (old) "jump 项" else "jump 项")
  cx_exo <- ox + c(1.05, 2.7, 4.35, 6.0, 7.65, 9.15)
  for (i in seq_along(exo)) {
    jump_i <- i == length(exo)
    box(cx_exo[i], 9.16, 1.5, 0.62, exo[i],
        fill = if (jump_i) (if (old) "#FDEDEC" else "#E9F7EF") else "white",
        border = if (jump_i) hl else "gray40",
        lwd = if (jump_i) 1.8 else 1.0, cex = 0.82)
  }
  graphics::text(ox + 9.75, 8.84, "外生结构项（同时进入 μ_e 与 μ_b）", cex = 0.78, col = GRAY, adj = 1)

  # ---- 外生 -> μ ----
  arrow(ox + 4.6, 8.7, ox + 4.6, 7.85, col = GRAY, lwd = 1.1)
  arrow(ox + 8.3, 8.7, ox + 8.3, 5.1,  col = GRAY, lwd = 1.1, lty = 2)

  # ---- μ 框 ----
  box(ox + 4.6, 7.5, 2.2, 0.68, "μ_e 结构先验均值", fill = COL_MU, cex = 0.95)
  box(ox + 8.3, 4.65, 2.0, 0.68, "μ_b 结构先验均值", fill = COL_MU, cex = 0.95)

  # ---- 潜变量框 ----
  box(ox + 1.3, 5.3, 1.7, 0.8, "e_{t-1}", fill = COL_PREV, cex = 1.0)
  box(ox + 4.6, 5.3, 1.7, 0.8, "e_t",     fill = COL_CUR,  cex = 1.05, font = 2)
  box(ox + 1.3, 3.1, 1.7, 0.8, "b_{t-1}", fill = COL_PREV, cex = 1.0)
  box(ox + 4.6, 3.1, 1.7, 0.8, "b_t",     fill = COL_CUR,  cex = 1.05, font = 2)
  graphics::text(ox + 2.95, 6.35, "自由潜变量（每格子一个参数）", cex = 0.8, col = GRAY)
  graphics::text(ox + 9.85, 6.9, "i = 1…N 时空格子", cex = 0.8, col = GRAY, adj = 1)

  # ---- 潜变量间箭头 ----
  arrow(ox + 2.15, 5.3, ox + 3.75, 5.3); graphics::text(ox + 2.95, 5.52, "β_e", cex = 0.95, font = 2)
  arrow(ox + 2.15, 3.1, ox + 3.75, 3.1); graphics::text(ox + 2.95, 2.88, "β_b", cex = 0.95, font = 2)
  arrow(ox + 4.6, 4.9, ox + 4.6, 3.5, col = BLU, lwd = 2.2)
  graphics::text(ox + 4.92, 4.2, "θ₁", cex = 1.1, font = 2, col = BLU)
  graphics::text(ox + 4.92, 3.95, "同期耦合", cex = 0.72, col = BLU)
  arrow(ox + 2.0, 4.85, ox + 3.85, 3.45, col = BLU, lwd = 1.4, lty = 3)
  graphics::text(ox + 2.45, 4.32, "θ₂ 滞后", cex = 0.85, col = BLU)

  # ---- μ -> 潜变量（虚线：只作先验均值） ----
  arrow(ox + 4.6, 7.16, ox + 4.6, 5.72, col = GRAY, lty = 2, lwd = 1.1)
  arrow(ox + 7.4, 4.5, ox + 5.5, 3.35, col = GRAY, lty = 2, lwd = 1.1)

  # ---- 观测 ----
  obs_e_lab <- if (old) "E_obs_i" else "E_obs_i"
  obs_b_lab <- if (old) "B_obs_i" else "B_obs_i"
  box(ox + 1.5, 1.55, 1.7, 0.7, obs_e_lab, fill = COL_OBS, border = "gray20", lwd = 1.6)
  box(ox + 7.6, 1.55, 1.7, 0.7, obs_b_lab, fill = COL_OBS, border = "gray20", lwd = 1.6)
  arrow(ox + 4.15, 5.0, ox + 2.1, 1.95, col = "gray30", lwd = 1.3)
  arrow(ox + 5.05, 2.95, ox + 6.85, 1.95, col = "gray30", lwd = 1.3)

  # 观测模型标注（新旧差异核心）
  if (old) {
    graphics::text(ox + 1.5, 2.35, "N(e_t, σ_e)", cex = 0.95, col = RED, font = 2)
    graphics::text(ox + 7.6, 2.35, "Beta(μ, φ)  等权", cex = 0.95, col = RED, font = 2)
    graphics::text(ox + 1.5, 0.95, "状态方程也用 σ_e（双重身份）", cex = 0.82, col = RED, adj = 0.5)
    graphics::text(ox + 7.6, 0.95, "n 帖格与 1 帖格同票", cex = 0.82, col = RED, adj = 0.5)
    graphics::text(cx_exo[6], 8.55, "自 t≥8 生效：错过 [0,+12h) 首窗", cex = 0.82,
                   col = RED, adj = 0.5, font = 2)
  } else {
    graphics::text(ox + 1.5, 2.35, "N(e_t, σ_obs/√n_i)", cex = 0.95, col = GRN, font = 2)
    graphics::text(ox + 7.6, 2.35, "Beta(μ, φ·n_i)", cex = 0.95, col = GRN, font = 2)
    graphics::text(ox + 1.5, 0.95, "状态方程独立用 σ_proc", cex = 0.82, col = GRN, adj = 0.5)
    graphics::text(ox + 7.6, 0.95, "精度随帖子数增长", cex = 0.82, col = GRN, adj = 0.5)
    graphics::text(cx_exo[6], 8.55, "自 t≥7：正确覆盖 [0,+12h)", cex = 0.82,
                   col = GRN, adj = 0.5, font = 2)
  }
}

graphics::text(11, 11.15, "DBN 模型结构：重构前 vs 重构后（骨架不变，噪声结构与观测加权修正）",
               cex = 1.5, font = 2)
draw_panel(0, "旧模型（op.stan + 4 份灾种文件）", "论文现行结果所依据的设定", old = TRUE)
draw_panel(11.5, "新模型（dbn_core.stan 单文件统一四灾种）", "重构版：同一 DAG + 修正项（绿）", old = FALSE)

graphics::text(11, 0.25,
               "骨架相同：e_t、b_t 为自由潜变量，AR（β）与耦合（θ₁/θ₂）通过结构化先验均值进入；变化只发生在观测模型、噪声分离与 jump 边界",
               cex = 0.95, col = GRAY)
dev.off()
cat("已生成:", normalizePath(outfile), "\n")
