# =====================================================================
# 论文图统一重绘
#   统一: NPG 科研分类色 + viridis 连续色 / Arial 全套字号 / 线宽规范
#   Fig1  θ1 后验密度 + 森林图（细边线 + 数值标注）
#   Fig2  时间轨迹 2x4（上情绪 / 下行为，列 = 地震/滑坡/泥石流/台风）
#   Fig3  空间衰减 2x4（同排列）
#   Fig4  分环模拟轨迹 2x4（细模型线）
#   FigS1 / FigS2  环-人口地图（同字体同风格）
# 运行: python3 runs/make_figures.py（依赖见 requirements-figures.txt）
# =====================================================================
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output/figure_2/_data"
OUT = ROOT / "output/figure_2"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- 统一风格常量 ----------
HZ = ["earthquake", "landslide", "debris_flow", "typhoon"]   # 统一列序
COL = {"earthquake": "#E64B35", "landslide": "#00A087",
       "debris_flow": "#7E6148", "typhoon": "#3C5488"}       # NPG 配色
LBL = {"earthquake": "Earthquake", "landslide": "Landslide",
       "debris_flow": "Debris flow", "typhoon": "Typhoon"}
OBS_C = "#1A1A1A"          # 观测点
GRID_C = "#9AA0A6"         # 参考线
YLABEL = {"E": "Negative emotion probability", "B": "Behavior probability"}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9.0,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 110,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("saved:", name)


def row_tags(axes):
    """每行第一面板内左上角的 (a)/(b) 行标签"""
    for r, tag in ((0, "(a)"), (1, "(b)")):
        axes[r, 0].text(0.03, 0.96, tag, transform=axes[r, 0].transAxes,
                        fontsize=10.5, fontweight="bold", va="top", ha="left")


def ring_ramp(hexc, n=5):
    c0, c1 = np.array(to_rgb(hexc)), np.array(to_rgb("#D2D2D2"))
    return [tuple(c0 + (c1 - c0) * i / (n - 1)) for i in range(n)]


# =====================================================================
# Fig 1：θ1 后验密度 + 森林图
# =====================================================================
def fig1():
    dens = pd.read_csv(DATA / "fig1_theta1_density.csv")
    forest = pd.read_csv(DATA / "fig1_forest.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 4.4))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.13,
                        wspace=0.28)

    # ---- (a) 后验密度（细边线 + 中位数线数值）----
    ymax = dens.y.max()
    med = {r.hazard: r["median"] for _, r in
           forest[forest.param == "theta1"].iterrows()}
    xr = (dens.x.min(), dens.x.max())
    pad = 0.04 * (xr[1] - xr[0])
    ax1.set_xlim(xr[0] - pad, xr[1] + pad)
    ax1.set_ylim(0, ymax * 1.12)
    stagger = [1.10, 0.95, 0.80, 0.65]
    for i, h in enumerate(HZ):
        d = dens[dens.hazard == h]
        ax1.fill(d.x, d.y, facecolor=COL[h], alpha=0.28, edgecolor=COL[h],
                 linewidth=0.9, zorder=2 + i)
        m = med[h]
        ax1.axvline(m, color=COL[h], lw=0.9, ls=(0, (4, 2.5)), alpha=0.75)
        ax1.text(m, ymax * stagger[i], f"{m:.3f}", color=COL[h],
                 fontsize=7.6, ha="center", va="top", fontweight="bold")
    ax1.set_xlabel(r"$\theta_1$")
    ax1.set_ylabel("Posterior density")
    ax1.set_title("(a) Contemporaneous coupling")
    ax1.legend(handles=[Patch(facecolor=COL[h], alpha=0.28, edgecolor=COL[h],
                              linewidth=0.9, label=LBL[h]) for h in HZ],
               loc="upper right", frameon=False, fontsize=7.6,
               handlelength=1.2, borderaxespad=0.2)

    # ---- (b) 森林图（细线 + 每点数值）----
    plist = ["theta1", "theta2", "beta_e", "beta_b"]
    plab = [r"$\theta_1$", r"$\theta_2$", r"$\beta_e$", r"$\beta_b$"]
    yloc = {"theta1": 4, "theta2": 3, "beta_e": 2, "beta_b": 1}
    off = {"earthquake": -0.15, "landslide": -0.05,
           "debris_flow": 0.05, "typhoon": 0.15}
    lab_dy = {"earthquake": -0.33, "landslide": -0.11,
              "debris_flow": 0.11, "typhoon": 0.33}
    xs = forest[forest.param.isin(plist)]
    xr2 = (xs.lo.min(), xs.hi.max())
    pad2 = 0.12 * (xr2[1] - xr2[0])
    ax2.set_xlim(xr2[0] - pad2, xr2[1] + pad2)
    ax2.set_ylim(0.5, 4.5)
    ax2.axvline(0, color=GRID_C, lw=0.8, ls=(0, (4, 3)))
    for h in HZ:
        for _, r in forest[(forest.hazard == h) & forest.param.isin(plist)].iterrows():
            y = yloc[r.param] + off[h]
            ax2.plot([r.lo, r.hi], [y, y], color=COL[h], lw=1.1,
                     solid_capstyle="butt", zorder=2)
            ax2.plot(r["mean"], y, "o", ms=3.8, color=COL[h], zorder=3)
            ax2.text(r["mean"], yloc[r.param] + lab_dy[h],
                     f'{r["mean"]:.3f}', color=COL[h],
                     fontsize=6.6, ha="center", va="center")
    ax2.set_yticks([yloc[p] for p in plist], plab)
    ax2.set_xlabel("Posterior mean (95% CrI)")
    ax2.set_title("(b) Key parameters by hazard")
    save(fig, "Fig1_theta1_posteriors")


# =====================================================================
# Fig 2 / Fig 3：2x4 网格（上情绪 / 下行为）
# =====================================================================
def grid_axes(figsize=(13.0, 7.0), top=0.90):
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    fig.subplots_adjust(left=0.055, right=0.99, top=top, bottom=0.085,
                        wspace=0.24, hspace=0.34)
    for ax in axes[0]:
        ax.tick_params(labelbottom=False)
    return fig, axes


def fig2():
    tm = pd.read_csv(DATA / "fig2_temporal.csv")
    ob = pd.read_csv(DATA / "fig2_obs.csv")
    fig, axes = grid_axes()
    # 行共享 y 范围（四灾种量级接近，直接由数据决定）
    ylim = {}
    for ch in ("E", "B"):
        top = max(max(tm[(tm.hazard == h) & (tm.channel == ch)].hi.max(),
                      ob[ob.hazard == h][f"obs_{ch}"].max()) for h in HZ)
        ylim[ch] = (0, top * 1.08)
    for r, ch in ((0, "E"), (1, "B")):
        for c, h in enumerate(HZ):
            ax = axes[r, c]
            d = tm[(tm.hazard == h) & (tm.channel == ch)]
            o = ob[ob.hazard == h].dropna(subset=[f"obs_{ch}"])
            ax.fill_between(d.time_hour, d.lo, d.hi, color=COL[h],
                            alpha=0.12, lw=0, zorder=1)
            ax.axvline(0, color=GRID_C, lw=0.8, ls=(0, (4, 3)), zorder=0)
            ax.plot(d.time_hour, d["mean"], color=COL[h], lw=2.0, zorder=3)
            ax.plot(o.time_hour, o[f"obs_{ch}"], "o", ms=3.0, color=OBS_C,
                    zorder=4)
            ax.set_ylim(*ylim[ch])
            ax.set_xlim(-72, 168)
            ax.set_xticks([-72, -24, 24, 72, 120, 168])
            if r == 0:
                ax.set_title(LBL[h])
            if c == 0:
                ax.set_ylabel(YLABEL[ch])
            if r == 1:
                ax.set_xlabel("Time since event (h)")
    row_tags(axes)
    handles = [Line2D([0], [0], color="#444444", lw=2.0,
                      label="Model mean (95% CrI)"),
               Line2D([0], [0], marker="o", ls="none", ms=3.5, color=OBS_C,
                      label="Observed cell mean")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.985))
    save(fig, "Fig2_temporal_split")


def fig3():
    sp = pd.read_csv(DATA / "fig3_spatial.csv")
    ob = pd.read_csv(DATA / "fig3_obs.csv")
    fig, axes = grid_axes()
    for r, ch in ((0, "E"), (1, "B")):
        for c, h in enumerate(HZ):
            ax = axes[r, c]
            d = sp[(sp.hazard == h) & (sp.channel == ch)]
            o = ob[ob.hazard == h].dropna(subset=[f"obs_{ch}"])
            ax.fill_between(d.dist_km, d.lo, d.hi, color=COL[h],
                            alpha=0.12, lw=0, zorder=1)
            ax.plot(d.dist_km, d["mean"], color=COL[h], lw=2.0, zorder=3)
            ax.plot(o.dist_km, o[f"obs_{ch}"], "o", ms=3.0, color=OBS_C,
                    zorder=4)
            ax.set_ylim(0, np.nanmax([d.hi.max(), o[f"obs_{ch}"].max()]) * 1.06)
            if r == 0:
                ax.set_title(LBL[h])
            if c == 0:
                ax.set_ylabel(YLABEL[ch])
            if r == 1:
                ax.set_xlabel("Distance from event (km)")
    row_tags(axes)
    handles = [Line2D([0], [0], color="#444444", lw=2.0,
                      label="Model mean (95% CrI)"),
               Line2D([0], [0], marker="o", ls="none", ms=3.5, color=OBS_C,
                      label="Observed ring mean")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.985))
    save(fig, "Fig3_spatial_split")


# =====================================================================
# Fig 4：分距离环模拟轨迹（2x4，细模型线）
# =====================================================================
def fig4():
    rg = pd.read_csv(DATA / "fig4_rings.csv")
    fig, axes = grid_axes()
    for r, ch in ((0, "E"), (1, "B")):
        for c, h in enumerate(HZ):
            ax = axes[r, c]
            d = rg[(rg.hazard == h) & (rg.channel == ch)]
            rings = d.ring.drop_duplicates().tolist()
            ramp = ring_ramp(COL[h], len(rings))
            ax.axvline(0, color=GRID_C, lw=0.8, ls=(0, (4, 3)), zorder=0)
            for i, ring in enumerate(rings):
                dr = d[d.ring == ring]
                ax.fill_between(dr.time_hour, dr.lo, dr.hi, color=ramp[i],
                                alpha=0.10, lw=0, zorder=1)
                ax.plot(dr.time_hour, dr["mean"], color=ramp[i], lw=1.1,
                        zorder=3)
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_xlim(-72, 168)
            ax.set_xticks([-72, -24, 24, 72, 120, 168])
            if r == 0:
                ax.set_title(LBL[h])
            if c == 0:
                ax.set_ylabel(YLABEL[ch])
            if r == 1:
                ax.set_xlabel("Time since event (h)")
    # 环图例：第 1 列（50 km 域）与第 4 列（200 km 域）
    d50 = rg[(rg.hazard == "earthquake") & (rg.channel == "E")]
    ramp50 = ring_ramp(COL["earthquake"], 5)
    axes[0, 0].legend(
        handles=[Line2D([0], [0], color=ramp50[i], lw=1.1, label=ring)
                 for i, ring in enumerate(d50.ring.drop_duplicates())],
        loc="upper right", frameon=False, fontsize=7.0, handlelength=1.6,
        borderaxespad=0.3, labelspacing=0.25)
    d200 = rg[(rg.hazard == "typhoon") & (rg.channel == "E")]
    ramp200 = ring_ramp(COL["typhoon"], 5)
    axes[0, 3].legend(
        handles=[Line2D([0], [0], color=ramp200[i], lw=1.1, label=ring)
                 for i, ring in enumerate(d200.ring.drop_duplicates())],
        loc="upper right", frameon=False, fontsize=7.0, handlelength=1.6,
        borderaxespad=0.3, labelspacing=0.25)
    row_tags(axes)
    save(fig, "Fig4_ring_trajectories")


# =====================================================================
# FigS1 / FigS2：环-人口地图
# =====================================================================
def load_grid(name):
    g = pd.read_csv(DATA / f"{name}.csv", header=None).values.astype(float)
    return np.flipud(g)


def map_style(ax, midlat):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect(1.0 / np.cos(np.deg2rad(midlat)))
    for s in ax.spines.values():
        s.set_linewidth(0.8); s.set_color("#555555")


def fig_s1(meta):
    ov = load_grid(meta["s1"]["overview"]["file"].replace(".csv", ""))
    e = meta["s1"]["overview"]["ext"]
    rings = pd.read_csv(DATA / "s1_rings_50km.csv")
    centers = pd.read_csv(DATA / "s1_centers.csv")
    zr = pd.read_csv(DATA / "s1_zoom_rings.csv")
    ann = pd.read_csv(DATA / "s1_zoom_annulus.csv")
    zooms = meta["s1"]["zoom"]
    vmax = max(np.nanmax(np.log10(ov + 1)),
               *(np.nanmax(np.log10(load_grid(z["grid"]["file"].replace(".csv", "")) + 1))
                 for z in zooms))
    vmax = np.ceil(vmax * 2) / 2

    fig = plt.figure(figsize=(10.5, 9.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.22, 1])
    fig.subplots_adjust(left=0.045, right=0.885, top=0.935, bottom=0.045,
                        wspace=0.10, hspace=0.16)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(np.log10(ov + 1), extent=e, origin="lower", cmap="viridis",
               vmin=0, vmax=vmax, interpolation="nearest")
    for eid, g in rings.groupby("event_id", sort=False):
        ax0.add_patch(Polygon(g[["x", "y"]].values, closed=True, fc="none",
                              ec=COL["earthquake"], lw=0.9, alpha=0.9))
    ax0.plot(centers.center_lon, centers.center_lat, "o", ms=2.2,
             color="#1A1A1A")
    map_style(ax0, 35)
    ax0.set_title("Earthquake events: 50-km collection rings on 2020 "
                  "population density")

    for k, z in enumerate(zooms):
        ax = fig.add_subplot(gs[1, k])
        zg = load_grid(z["grid"]["file"].replace(".csv", ""))
        ze = z["grid"]["ext"]
        ax.imshow(np.log10(zg + 1), extent=ze, origin="lower", cmap="viridis",
                  vmin=0, vmax=vmax, interpolation="nearest")
        zz = zr[zr.zoom == z["id"]]
        for km, g in zz.groupby("km"):
            ax.add_patch(Polygon(g[["x", "y"]].values, closed=True, fc="none",
                                 ec="white", lw=0.8))
        ga = ann[ann.zoom == z["id"]]
        ax.add_patch(Polygon(ga[["x", "y"]].values, closed=True,
                             fc=COL["earthquake"], alpha=0.35,
                             ec=COL["earthquake"], lw=1.1))
        ax.plot(z["lon"], z["lat"], "o", ms=4.5, color=COL["earthquake"])
        map_style(ax, z["lat"])
        ax.set_title(f'{z["event_id"]}: 45\u201350 km ring = '
                     f'{z["far_dens"]:.0f} pers./km\u00b2')
    cax = fig.add_axes([0.905, 0.08, 0.018, 0.80])
    cb = fig.colorbar(ax0.images[0], cax=cax)
    cb.set_label(r"$\log_{10}$(persons/km$^2$ + 1)", fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)
    cb.outline.set_linewidth(0.8)
    save(fig, "FigS1_eq_ring_population")


def fig_s2(meta):
    m = meta["s2"]
    ov = load_grid(m["overview"]["file"].replace(".csv", ""))
    e = m["overview"]["ext"]
    mask = load_grid(m["corridor_mask"]["file"].replace(".csv", ""))
    tracks = pd.read_csv(DATA / "s2_tracks.csv")
    points = pd.read_csv(DATA / "s2_points.csv")
    zr = pd.read_csv(DATA / "s2_zoom_rings.csv")
    z = m["zoom"]
    zg = load_grid(z["grid"]["file"].replace(".csv", ""))
    ze = z["grid"]["ext"]
    vmax = np.nanmax([np.nanmax(np.log10(ov + 1)), np.nanmax(np.log10(zg + 1))])
    vmax = np.ceil(vmax * 2) / 2
    midlat = float(np.nanmean([e[2], e[3]]))

    fig = plt.figure(figsize=(10.5, 9.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.22, 1])
    fig.subplots_adjust(left=0.045, right=0.885, top=0.935, bottom=0.045,
                        wspace=0.10, hspace=0.16)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(np.log10(ov + 1), extent=e, origin="lower", cmap="viridis",
               vmin=0, vmax=vmax, interpolation="nearest")
    ax0.imshow(mask, extent=e, origin="lower", vmin=0, vmax=1,
               cmap=ListedColormap([(0, 0, 0, 0), (*to_rgb(COL["typhoon"]), 0.12)]),
               interpolation="nearest")
    xs = np.linspace(e[0], e[1], mask.shape[1])
    ys = np.linspace(e[2], e[3], mask.shape[0])
    ax0.contour(xs, ys, mask, levels=[0.5], colors=[COL["typhoon"]],
                linewidths=1.0, alpha=0.85)
    for tid, g in tracks.groupby("typhoon_id", sort=False):
        g = g.sort_values("pt")
        ax0.plot(g.x, g.y, color=COL["typhoon"], lw=1.2)
    ax0.plot(points.center_lon, points.center_lat, "o", ms=2.2,
             color=COL["typhoon"])
    map_style(ax0, midlat)
    ax0.set_title("Typhoon tracks and 200-km collection corridor on 2020 "
                  "population density")

    axz = fig.add_subplot(gs[1, 0])
    axz.imshow(np.log10(zg + 1), extent=ze, origin="lower", cmap="viridis",
               vmin=0, vmax=vmax, interpolation="nearest")
    for km, g in zr.groupby("km"):
        axz.add_patch(Polygon(g[["x", "y"]].values, closed=True, fc="none",
                              ec="white", lw=0.8))
    axz.plot(z["lon"], z["lat"], "o", ms=4.5, color=COL["typhoon"])
    map_style(axz, z["lat"])
    axz.set_title(f'Track point {z["event_id"]}: 0\u201310 km = '
                  f'{z["near10"]:.0f}, 90\u2013100 km = {z["mid100"]:.0f} '
                  f'pers./km\u00b2')

    axb = fig.add_subplot(gs[1, 1])
    dens = np.asarray(z["ring_dens"], dtype=float)
    axb.bar(np.arange(1, 21), dens, color=COL["typhoon"], alpha=0.85, width=0.8)
    axb.set_xticks(np.arange(1, 21))
    axb.tick_params(labelsize=7.5)
    axb.set_xlabel("Distance ring (10-km bins)")
    axb.set_ylabel("Persons/km\u00b2")
    axb.set_title(f'Ring density, track point {z["event_id"]}')
    axb.spines["top"].set_visible(False); axb.spines["right"].set_visible(False)

    cax = fig.add_axes([0.905, 0.08, 0.018, 0.80])
    cb = fig.colorbar(ax0.images[0], cax=cax)
    cb.set_label(r"$\log_{10}$(persons/km$^2$ + 1)", fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)
    cb.outline.set_linewidth(0.8)
    save(fig, "FigS2_ty_ring_population")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    meta = json.loads((DATA / "meta.json").read_text())
    fig_s1(meta); fig_s2(meta)
    print("全部完成 ->", OUT)
