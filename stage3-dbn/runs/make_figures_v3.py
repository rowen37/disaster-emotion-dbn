# =====================================================================
# Stage 3 - unified paper figure renderer (v3), consuming the CSVs
# exported by export_figure2_data.R.
#   1) Fig1: densities via scipy gaussian_kde (1000 draws, 1024 grid
#      points), 600-dpi PNG; all numeric annotations in black
#   2) Fig2/Fig3: model peak annotated per panel (black)
#   3) Fig4: ring legend in all four columns, y axis fixed to 0-0.8
#   4) FigS1/S2: colorbar attached to the top map only; nodata in light
#      gray (white rings keep land/sea visible); zoom titles carry event
#      details (date/magnitude/typhoon level, from the raw data); every
#      event id labeled in the overview; "ring density" spelled as
#      "ring population density"
# Run: uv run --with numpy --with pandas --with scipy --with matplotlib \
#         python3 make_figures_v3.py
# =====================================================================
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

ROOT = Path("/Users/rogerbqw/Downloads/DBNs/refactor")
DATA = ROOT / "output/figure_2/_data"
OUT = ROOT / "output/figure_3"
OUT.mkdir(parents=True, exist_ok=True)
RAW = Path("/Users/rogerbqw/Downloads/weibo_dbn/input")
FEAT = pd.read_csv("/Users/rogerbqw/Downloads/DBNs/event_features.csv",
                   dtype={"event_id": str})

HZ = ["earthquake", "landslide", "debris_flow", "typhoon"]
COL = {"earthquake": "#E64B35", "landslide": "#00A087",
       "debris_flow": "#7E6148", "typhoon": "#3C5488"}
LBL = {"earthquake": "Earthquake", "landslide": "Landslide",
       "debris_flow": "Debris flow", "typhoon": "Typhoon"}
OBS_C = "#1A1A1A"
GRID_C = "#9AA0A6"
NODATA = "#E8E8E8"           # raster nodata (sea / outside coverage)
YLABEL = {"E": "Negative emotion probability", "B": "Behavior probability"}
STROKE = [pe.withStroke(linewidth=1.6, foreground="#1A1A1A")]

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


def save(fig, name, dpi=600):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=dpi)
    plt.close(fig)
    print("saved:", name)


def row_tags(axes):
    for r, tag in ((0, "(a)"), (1, "(b)")):
        axes[r, 0].text(0.03, 0.96, tag, transform=axes[r, 0].transAxes,
                        fontsize=10.5, fontweight="bold", va="top", ha="left")


def ring_ramp(hexc, n=5):
    c0, c1 = np.array(to_rgb(hexc)), np.array(to_rgb("#D2D2D2"))
    return [tuple(c0 + (c1 - c0) * i / (n - 1)) for i in range(n)]


def annotate_peak(ax, xs, ys, color, x_range, fmt="{:.2f}"):
    """Peak of the model curve: highlighted dot + black numeric label."""
    i = int(np.argmax(ys))
    px, pv = xs[i], ys[i]
    ax.plot(px, pv, "o", ms=4.2, mfc=color, mec="white", mew=0.9, zorder=6)
    near_left = (px - x_range[0]) < 0.18 * (x_range[1] - x_range[0])
    if near_left:
        ax.annotate(fmt.format(pv), (px, pv), textcoords="offset points",
                    xytext=(8, 3), ha="left", va="center", fontsize=7.4,
                    color="black", fontweight="bold", zorder=7)
    else:
        ax.annotate(fmt.format(pv), (px, pv), textcoords="offset points",
                    xytext=(0, 6), ha="center", va="bottom", fontsize=7.4,
                    color="black", fontweight="bold", zorder=7)


# =====================================================================
# Fig 1
# =====================================================================
def fig1():
    draws = pd.read_csv(DATA / "fig1_theta1_draws.csv")
    forest = pd.read_csv(DATA / "fig1_forest.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 4.4))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.13,
                        wspace=0.28)

    # ---- (a) posterior density: gaussian_kde (1024 points, Scott bandwidth) ----
    dens, meds = {}, {}
    for h in HZ:
        v = draws[draws.hazard == h].theta1.values
        k = gaussian_kde(v)
        pad = 3 * k.factor * v.std()
        xs = np.linspace(v.min() - pad, v.max() + pad, 1024)
        dens[h] = (xs, k(xs))
        meds[h] = float(np.median(v))
    ymax = max(y.max() for _, y in dens.values())
    xall = np.concatenate([x for x, _ in dens.values()])
    pad_x = 0.04 * (xall.max() - xall.min())
    ax1.set_xlim(xall.min() - pad_x, xall.max() + pad_x)
    ax1.set_ylim(0, ymax * 1.12)
    stagger = [1.10, 0.95, 0.80, 0.65]
    for i, h in enumerate(HZ):
        x, y = dens[h]
        ax1.fill(x, y, facecolor=COL[h], alpha=0.28, edgecolor=COL[h],
                 linewidth=0.9, zorder=2 + i)
        m = meds[h]
        ax1.axvline(m, color=COL[h], lw=0.9, ls=(0, (4, 2.5)), alpha=0.75)
        ax1.text(m, ymax * stagger[i], f"{m:.3f}", color="black",
                 fontsize=6.8, ha="center", va="top")
    ax1.set_xlabel(r"$\theta_1$")
    ax1.set_ylabel("Posterior density")
    ax1.set_title("(a) Contemporaneous coupling")
    ax1.legend(handles=[Patch(facecolor=COL[h], alpha=0.28, edgecolor=COL[h],
                              linewidth=0.9, label=LBL[h]) for h in HZ],
               loc="upper right", frameon=False, fontsize=7.6,
               handlelength=1.2, borderaxespad=0.2)

    # ---- (b) forest plot (thin lines + black numeric labels) ----
    plist = ["theta1", "theta2", "beta_e", "beta_b"]
    plab = [r"$\theta_1$", r"$\theta_2$", r"$\beta_e$", r"$\beta_b$"]
    yloc = {"theta1": 4, "theta2": 3, "beta_e": 2, "beta_b": 1}
    off = {"earthquake": -0.15, "landslide": -0.05,
           "debris_flow": 0.05, "typhoon": 0.15}
    lab_dy = {"earthquake": -0.33, "landslide": -0.11,
              "debris_flow": 0.11, "typhoon": 0.33}
    xs_ = forest[forest.param.isin(plist)]
    xr2 = (xs_.lo.min(), xs_.hi.max())
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
                     f'{r["mean"]:.3f}', color="black",
                     fontsize=6.8, ha="center", va="center")
    ax2.set_yticks([yloc[p] for p in plist], plab)
    ax2.set_xlabel("Posterior mean (95% CrI)")
    ax2.set_title("(b) Key parameters by hazard")
    save(fig, "Fig1_theta1_posteriors", dpi=600)


# =====================================================================
# Fig 2 / Fig 3
# =====================================================================
def grid_axes(figsize=(13.0, 7.0)):
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.085,
                        wspace=0.24, hspace=0.34)
    for ax in axes[0]:
        ax.tick_params(labelbottom=False)
    return fig, axes


def fig2():
    tm = pd.read_csv(DATA / "fig2_temporal.csv")
    ob = pd.read_csv(DATA / "fig2_obs.csv")
    fig, axes = grid_axes()
    ylim = {}
    for ch in ("E", "B"):
        top = max(max(tm[(tm.hazard == h) & (tm.channel == ch)].hi.max(),
                      ob[ob.hazard == h][f"obs_{ch}"].max()) for h in HZ)
        ylim[ch] = (0, top * 1.10)
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
            annotate_peak(ax, d.time_hour.values, d["mean"].values, COL[h],
                          (-72, 168))
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
    save(fig, "Fig2_temporal_split", dpi=600)


def fig3():
    sp = pd.read_csv(DATA / "fig3_spatial.csv")
    ob = pd.read_csv(DATA / "fig3_obs.csv")
    fig, axes = grid_axes()
    for r, ch in ((0, "E"), (1, "B")):
        for c, h in enumerate(HZ):
            ax = axes[r, c]
            d = sp[(sp.hazard == h) & (sp.channel == ch)]
            o = ob[ob.hazard == h].dropna(subset=[f"obs_{ch}"])
            xmax = float(d.dist_km.max())
            ax.fill_between(d.dist_km, d.lo, d.hi, color=COL[h],
                            alpha=0.12, lw=0, zorder=1)
            ax.plot(d.dist_km, d["mean"], color=COL[h], lw=2.0, zorder=3)
            ax.plot(o.dist_km, o[f"obs_{ch}"], "o", ms=3.0, color=OBS_C,
                    zorder=4)
            annotate_peak(ax, d.dist_km.values, d["mean"].values, COL[h],
                          (0, xmax))
            ax.set_ylim(0, np.nanmax([d.hi.max(), o[f"obs_{ch}"].max()]) * 1.10)
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
    save(fig, "Fig3_spatial_split", dpi=600)


# =====================================================================
# Fig 4: ring legend in all columns, y axis 0-0.8
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
            ax.set_ylim(0, 0.8)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
            ax.set_xlim(-72, 168)
            ax.set_xticks([-72, -24, 24, 72, 120, 168])
            if r == 0:
                ax.set_title(LBL[h])
                ramp_e = ring_ramp(COL[h], len(rings))
                ax.legend(handles=[Line2D([0], [0], color=ramp_e[i], lw=1.1,
                                          label=ring)
                                   for i, ring in enumerate(rings)],
                          loc="upper right", frameon=False, fontsize=6.8,
                          handlelength=1.5, borderaxespad=0.25,
                          labelspacing=0.22)
            if c == 0:
                ax.set_ylabel(YLABEL[ch])
            if r == 1:
                ax.set_xlabel("Time since event (h)")
    row_tags(axes)
    save(fig, "Fig4_ring_trajectories", dpi=600)


# =====================================================================
# FigS1 / FigS2
# =====================================================================
def load_grid(name):
    g = pd.read_csv(DATA / f"{name}.csv", header=None).values.astype(float)
    return np.flipud(g)


def viridis_bad():
    cm = plt.get_cmap("viridis").copy()
    cm.set_bad(NODATA)
    return cm


def map_style(ax, midlat):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect(1.0 / np.cos(np.deg2rad(midlat)))
    for s in ax.spines.values():
        s.set_linewidth(0.8); s.set_color("#555555")


def event_date(eid, hz):
    p = RAW / hz / f"{eid}.csv"
    dt = pd.read_csv(p, usecols=["happened_at"], nrows=1).iloc[0, 0]
    return pd.to_datetime(dt).strftime("%Y-%m-%d")


def side_colorbar(fig, ax, im):
    """Colorbar flush with the top map's right edge, small gap."""
    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.014, pos.y0, 0.013, pos.height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\log_{10}$(persons/km$^2$ + 1)", fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)
    cb.outline.set_linewidth(0.8)


def fig_s1(meta):
    ov = load_grid(meta["s1"]["overview"]["file"].replace(".csv", ""))
    e = meta["s1"]["overview"]["ext"]
    rings = pd.read_csv(DATA / "s1_rings_50km.csv")
    centers = pd.read_csv(DATA / "s1_centers.csv", dtype={"event_id": str})
    zr = pd.read_csv(DATA / "s1_zoom_rings.csv")
    ann = pd.read_csv(DATA / "s1_zoom_annulus.csv")
    zooms = meta["s1"]["zoom"]
    vmax = max(np.nanmax(np.log10(ov + 1)),
               *(np.nanmax(np.log10(load_grid(z["grid"]["file"].replace(".csv", "")) + 1))
                 for z in zooms))
    vmax = np.ceil(vmax * 2) / 2
    cmap = viridis_bad()

    fig = plt.figure(figsize=(10.8, 9.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.22, 1])
    fig.subplots_adjust(left=0.03, right=0.965, top=0.935, bottom=0.04,
                        wspace=0.08, hspace=0.16)
    ax0 = fig.add_subplot(gs[0, :])
    im0 = ax0.imshow(np.ma.masked_invalid(np.log10(ov + 1)), extent=e,
                     origin="lower", cmap=cmap, vmin=0, vmax=vmax,
                     interpolation="nearest")
    for eid, g in rings.groupby("event_id", sort=False):
        ax0.add_patch(Polygon(g[["x", "y"]].values, closed=True, fc="none",
                              ec=COL["earthquake"], lw=0.9, alpha=0.9))
    ax0.plot(centers.center_lon, centers.center_lat, "o", ms=2.2,
             color="#1A1A1A")
    for _, q in centers.iterrows():   # label every event id
        ax0.text(q.center_lon, q.center_lat + 0.45, q.event_id, color="white",
                 fontsize=5.2, ha="center", va="bottom", path_effects=STROKE,
                 zorder=6)
    map_style(ax0, 35)
    ax0.set_title(f"Earthquake events (n = {len(centers)}): 50-km collection "
                  "rings on 2020 population density")
    side_colorbar(fig, ax0, im0)

    for k, z in enumerate(zooms):
        ax = fig.add_subplot(gs[1, k])
        zg = load_grid(z["grid"]["file"].replace(".csv", ""))
        ze = z["grid"]["ext"]
        ax.imshow(np.ma.masked_invalid(np.log10(zg + 1)), extent=ze,
                  origin="lower", cmap=cmap, vmin=0, vmax=vmax,
                  interpolation="nearest")
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
        mag = FEAT.loc[FEAT.event_id == z["event_id"], "mag"].iloc[0]
        ax.set_title(f'Event {z["event_id"]} \u00b7 M {mag:.1f} \u00b7 '
                     f'{event_date(z["event_id"], "earthquake")}\n'
                     f'45\u201350 km ring population density = '
                     f'{z["far_dens"]:,.0f} pers./km\u00b2', fontsize=9.0)
    save(fig, "FigS1_eq_ring_population", dpi=300)


def fig_s2(meta):
    m = meta["s2"]
    ov = load_grid(m["overview"]["file"].replace(".csv", ""))
    e = m["overview"]["ext"]
    mask = load_grid(m["corridor_mask"]["file"].replace(".csv", ""))
    tracks = pd.read_csv(DATA / "s2_tracks.csv")
    points = pd.read_csv(DATA / "s2_points.csv", dtype={"event_id": str,
                                                        "typhoon_id": str})
    zr = pd.read_csv(DATA / "s2_zoom_rings.csv")
    z = m["zoom"]
    zg = load_grid(z["grid"]["file"].replace(".csv", ""))
    ze = z["grid"]["ext"]
    vmax = np.nanmax([np.nanmax(np.log10(ov + 1)), np.nanmax(np.log10(zg + 1))])
    vmax = np.ceil(vmax * 2) / 2
    midlat = float(np.nanmean([e[2], e[3]]))
    cmap = viridis_bad()

    fig = plt.figure(figsize=(10.8, 9.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.22, 1])
    fig.subplots_adjust(left=0.03, right=0.965, top=0.935, bottom=0.04,
                        wspace=0.08, hspace=0.16)
    ax0 = fig.add_subplot(gs[0, :])
    im0 = ax0.imshow(np.ma.masked_invalid(np.log10(ov + 1)), extent=e,
                     origin="lower", cmap=cmap, vmin=0, vmax=vmax,
                     interpolation="nearest")
    ax0.imshow(mask, extent=e, origin="lower", vmin=0, vmax=1,
               cmap=ListedColormap([(0, 0, 0, 0),
                                    (*to_rgb(COL["typhoon"]), 0.12)]),
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
    for _, q in points.iterrows():   # label every track point id
        ax0.text(q.center_lon, q.center_lat + 0.22, q.event_id, color="white",
                 fontsize=4.8, ha="center", va="bottom", path_effects=STROKE,
                 zorder=6)
    map_style(ax0, midlat)
    nty = points.typhoon_id.nunique()
    ax0.set_title(f"Typhoon tracks ({nty} typhoons, {len(points)} track "
                  "points): 200-km collection corridor on 2020 population "
                  "density")
    side_colorbar(fig, ax0, im0)

    axz = fig.add_subplot(gs[1, 0])
    axz.imshow(np.ma.masked_invalid(np.log10(zg + 1)), extent=ze,
               origin="lower", cmap=cmap, vmin=0, vmax=vmax,
               interpolation="nearest")
    for km, g in zr.groupby("km"):
        axz.add_patch(Polygon(g[["x", "y"]].values, closed=True, fc="none",
                              ec="white", lw=0.8))
    ann2 = pd.read_csv(DATA / "s2_zoom_annulus.csv")   # highlighted 90-100 km annulus
    axz.add_patch(Polygon(ann2[["x", "y"]].values, closed=True,
                          fc=COL["typhoon"], alpha=0.35,
                          ec=COL["typhoon"], lw=1.1))
    axz.plot(z["lon"], z["lat"], "o", ms=4.5, color=COL["typhoon"])
    map_style(axz, z["lat"])
    lvl = FEAT.loc[FEAT.event_id == z["event_id"], "typhoon_level"].iloc[0]
    axz.set_title(f'Track point {z["event_id"]} \u00b7 Typhoon '
                  f'{str(z["event_id"]).split("-")[0]} \u00b7 Level {lvl:.0f}'
                  f' \u00b7 {event_date(z["event_id"], "typhoon")}\n'
                  f'0\u201310 km = {z["near10"]:,.0f}, '
                  f'90\u2013100 km = {z["mid100"]:,.0f} pers./km\u00b2',
                  fontsize=9.0)

    axb = fig.add_subplot(gs[1, 1])
    dens = np.asarray(z["ring_dens"], dtype=float)
    axb.bar(np.arange(1, 21), dens, color=COL["typhoon"], alpha=0.85, width=0.8)
    axb.set_xticks(np.arange(1, 21))
    axb.tick_params(labelsize=7.5)
    axb.set_xlabel("Distance ring (10-km bins)")
    axb.set_ylabel("Persons/km\u00b2")
    axb.set_title(f'Ring population density,\ntrack point {z["event_id"]}',
                  fontsize=9.0)
    axb.spines["top"].set_visible(False); axb.spines["right"].set_visible(False)
    save(fig, "FigS2_ty_ring_population", dpi=300)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    meta = json.loads((DATA / "meta.json").read_text())
    fig_s1(meta); fig_s2(meta)
    print("全部完成 ->", OUT)
