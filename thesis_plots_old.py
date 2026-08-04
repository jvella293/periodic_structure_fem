"""Thesis-quality figures, exported individually as vector PDFs.

Design decisions
----------------
* PALETTE is Okabe-Ito, the standard colourblind-safe qualitative set
  (deuteranopia, protanopia and tritanopia safe). Colours are assigned
  SEMANTIC names here and used consistently across every figure, so
  "time-domain" is the same colour in every plot in the chapter.
* No titles are baked into the figures — captions belong in LaTeX. Set
  DEBUG_HEADER = True to get the parameter banner back for working notes.
* Figures are sized to the text width so they need no scaling in LaTeX;
  scaling a figure changes its effective font size and is the usual reason
  thesis figures look inconsistent.
* Dense traces are rasterised inside an otherwise-vector PDF: axes, text
  and fitted lines stay sharp, but a 100k-point trace does not bloat the
  file to tens of megabytes.

Usage
-----
    import thesis_plots as tp
    tp.use_thesis_style()
    tp.fig_growth(t, F, g, stem="2T_growth")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Match your LaTeX body font. "stix" ~ Times (most TU Delft templates);
# "cm" ~ Computer Modern (default LaTeX report class).
FONT = "stix"

# TU Delft report class text width is ~150 mm. Measure yours with
# \the\textwidth in LaTeX and put the value here in inches.
TEXT_WIDTH = 5.91          # in — full text width
HALF_WIDTH = 2.87          # in — two figures side by side
GOLDEN = 0.618

DEBUG_HEADER = False       # True => parameter banner across the top
FIG_DIR = Path("figures")
SAVE_PNG = True            # PNG alongside PDF for quick browsing
DPI_RASTER = 600           # for rasterised dense traces


# ----------------------------------------------------------------------
# Okabe-Ito colourblind-safe palette
# ----------------------------------------------------------------------

OKABE_ITO = {
    "black":     "#000000",
    "orange":    "#E69F00",
    "sky":       "#56B4E9",
    "green":     "#009E73",
    "yellow":    "#F0E442",
    "blue":      "#0072B2",
    "vermilion": "#D55E00",
    "purple":    "#CC79A7",
}

# Semantic assignment — use these, not the raw colours, so a change here
# propagates through every figure in the chapter.
C = {
    "string":     OKABE_ITO["sky"],        # w_c(t), string at contact
    "mass1":      OKABE_ITO["vermilion"],  # z1(t), contact mass
    "mass2":      OKABE_ITO["green"],      # z2(t), secondary mass
    "force":      OKABE_ITO["blue"],       # contact force
    "envelope":   OKABE_ITO["orange"],     # upper hull
    "fit":        OKABE_ITO["black"],      # fitted exponential
    "timedomain": OKABE_ITO["vermilion"],  # time-domain FEM results
    "floquet":    OKABE_ITO["blue"],       # Floquet/Hill reference
    "tongue":     OKABE_ITO["purple"],     # tongue boundaries
    "neutral":    "#999999",               # zero lines, guides
}


def use_thesis_style() -> None:
    """Apply the thesis rcParams. Call once before making figures."""
    serif = {
        "stix": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "cm": ["CMU Serif", "cmr10", "DejaVu Serif"],
    }[FONT]
    mathfont = {"stix": "stix", "cm": "cm"}[FONT]

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": serif,
        "mathtext.fontset": mathfont,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,

        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "grid.color": C["neutral"],

        "lines.linewidth": 1.0,
        "lines.markersize": 4,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,

        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "legend.borderpad": 0.4,

        "figure.dpi": 120,
        "savefig.dpi": DPI_RASTER,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,        # embed TrueType — required by most
        "ps.fonttype": 42,         # thesis submission systems
    })


def save_figure(fig, stem: str, header: str | None = None) -> Path:
    """Export one figure as a vector PDF (and optionally PNG).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to export.
    stem : str
        Filename stem — no extension, no spaces.
    header : str or None
        Parameter banner, drawn only when DEBUG_HEADER is True.

    Returns
    -------
    pathlib.Path
        Path to the written PDF.
    """
    FIG_DIR.mkdir(exist_ok=True)
    if DEBUG_HEADER and header:
        fig.suptitle(header, fontsize=6, family="monospace")
        fig.subplots_adjust(top=0.86)

    pdf_path = FIG_DIR / f"{stem}.pdf"
    fig.savefig(pdf_path)
    if SAVE_PNG:
        fig.savefig(FIG_DIR / f"{stem}.png")
    return pdf_path


def _new(width: float = TEXT_WIDTH, ratio: float = GOLDEN):
    fig, ax = plt.subplots(figsize=(width, width * ratio))
    return fig, ax


RASTER_THRESHOLD = 5000


def _raster(n: int) -> bool:
    """Rasterise only dense traces; keep sparse ones fully vector."""
    return n > RASTER_THRESHOLD


# ----------------------------------------------------------------------
# Tier 2 — representative time-domain response (one figure per panel)
# ----------------------------------------------------------------------

def fig_displacement(t, w_c, z1, z2=None, *, stem="displacement",
                     header=None, t_window=None, width=TEXT_WIDTH):
    """Displacement time histories at the contact point.

    Parameters
    ----------
    t, w_c, z1 : array_like
        Time, string displacement at contact, contact-mass displacement.
    z2 : array_like or None
        Secondary-mass displacement. Omit unless the figure is making a
        point about the second DOF — two traces read more clearly.
    t_window : tuple of float or None
        ``(t0, t1)`` to zoom on a few cycles. Full history hides the
        waveform once there are hundreds of oscillations; a zoomed inset
        of the late-time response is usually the more informative figure.
    """
    fig, ax = _new(width)
    sl = slice(None)
    if t_window is not None:
        sl = (t >= t_window[0]) & (t <= t_window[1])

    scale = 1e3  # metres -> millimetres
    n = np.asarray(t)[sl].size
    ax.plot(t[sl], np.asarray(w_c)[sl] * scale, color=C["string"],
            label=r"$w_c(t)$  string at contact", rasterized=_raster(n))
    ax.plot(t[sl], np.asarray(z1)[sl] * scale, color=C["mass1"],
            label=r"$z_1(t)$  contact mass", rasterized=_raster(n))
    if z2 is not None:
        ax.plot(t[sl], np.asarray(z2)[sl] * scale, color=C["mass2"],
                label=r"$z_2(t)$  secondary mass", rasterized=_raster(n))

    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(r"Displacement [mm]")
    ax.legend(loc="upper left")
    ax.margins(x=0.01)
    return fig, save_figure(fig, stem, header)


def fig_contact_force(t, F, *, stem="contact_force", header=None,
                      t_window=None, width=TEXT_WIDTH):
    """Contact-force perturbation time history (linear scale)."""
    fig, ax = _new(width)
    sl = slice(None)
    if t_window is not None:
        sl = (t >= t_window[0]) & (t <= t_window[1])

    ax.plot(np.asarray(t)[sl], np.asarray(F)[sl], color=C["force"],
            lw=0.8, rasterized=_raster(np.asarray(t)[sl].size))
    ax.axhline(0.0, color=C["neutral"], lw=0.5, zorder=0)
    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(r"Contact force perturbation $F_{\mathrm{tr}}$ [N]")
    ax.margins(x=0.01)
    return fig, save_figure(fig, stem, header)


def fig_growth(t, F, g, *, stem="growth_rate", header=None,
               width=TEXT_WIDTH, annotate=True):
    """Log-envelope growth figure: |F_tr|, upper hull, fitted exponential.

    This is the evidence figure — it is what demonstrates the instability
    and where Re(lambda) comes from.

    Parameters
    ----------
    g : dict
        Output of ``main.growth_rate``: needs ``env``, ``hull``, and (if a
        fit succeeded) ``slope``, ``intercept``, ``t_hull``.
    """
    fig, ax = _new(width, ratio=0.55)

    n = np.asarray(t).size
    ax.semilogy(t, g["env"], color=C["string"], lw=0.4, alpha=0.6,
                rasterized=_raster(n), label=r"$|F_{\mathrm{tr}}(t)|$")
    ax.semilogy(t, g["hull"], color=C["envelope"], lw=1.1,
                rasterized=_raster(n), label="Upper envelope")

    if g.get("slope") is not None:
        t_h = g["t_hull"]
        ax.semilogy(t_h, np.exp(g["intercept"] + g["slope"] * t_h),
                    color=C["fit"], ls="--", lw=1.0,
                    label=r"Fit, $\mathrm{Re}(\lambda)$ = "
                          rf"${g['slope']:+.4f}$ s$^{{-1}}$")
        if annotate:
            err = g.get("slope_err", 0.0)
            ax.text(0.03, 0.94,
                    rf"$\mathrm{{Re}}(\lambda) = {g['slope']:+.4f}"
                    rf" \pm {err:.4f}$ s$^{{-1}}$",
                    transform=ax.transAxes, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="0.8", lw=0.5))

    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(r"$|F_{\mathrm{tr}}|$ [N]")
    ax.legend(loc="lower right")
    ax.margins(x=0.01)
    return fig, save_figure(fig, stem, header)


# ----------------------------------------------------------------------
# Tier 1 — stability-map validation (the money plot)
# ----------------------------------------------------------------------

def fig_stability_sweep(td_m1, td_rate, td_err=None, *,
                        floquet_m1=None, floquet_rate=None,
                        tongue=None, stem="stability_sweep",
                        header=None, width=TEXT_WIDTH,
                        xlabel=r"Contact mass $m_1$ [kg]"):
    """Re(lambda) vs swept parameter: Floquet curve + time-domain points.

    This is the figure that actually demonstrates validation. A single
    matching point is necessary but not sufficient; a reader wants
    agreement across a sweep, ideally through a tongue.

    Parameters
    ----------
    td_m1, td_rate : array_like
        Time-domain sweep coordinate and measured growth rates.
    td_err : array_like or None
        Fit uncertainties, drawn as error bars.
    floquet_m1, floquet_rate : array_like or None
        Reference curve from the Hill/Floquet code. Omit to plot the
        time-domain points alone.
    tongue : tuple of float or None
        ``(m1_lo, m1_hi)`` — shaded as the predicted unstable band.
    """
    fig, ax = _new(width, ratio=0.55)

    if tongue is not None:
        lo, hi = tongue
        ax.axvspan(lo, hi, color=C["tongue"], alpha=0.12, lw=0,
                   label="Floquet unstable band", zorder=0)
        for edge in (lo, hi):
            ax.axvline(edge, color=C["tongue"], lw=0.7, ls=":", zorder=1)

    ax.axhline(0.0, color=C["neutral"], lw=0.6, zorder=1)

    if floquet_m1 is not None and floquet_rate is not None:
        ax.plot(floquet_m1, floquet_rate, color=C["floquet"], lw=1.2,
                label="Floquet (frequency domain)", zorder=2)

    if td_err is not None:
        ax.errorbar(td_m1, td_rate, yerr=td_err, fmt="o",
                    color=C["timedomain"], mfc="white", mew=1.0,
                    capsize=2.5, elinewidth=0.8, lw=0,
                    label="Time-domain FEM", zorder=3)
    else:
        ax.plot(td_m1, td_rate, "o", color=C["timedomain"], mfc="white",
                mew=1.0, lw=0, label="Time-domain FEM", zorder=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Growth rate $\mathrm{Re}(\lambda)$ [s$^{-1}$]")
    ax.legend(loc="best")
    ax.margins(x=0.03)
    return fig, save_figure(fig, stem, header)


def fig_sweep_error(td_m1, td_rate, floquet_rate_at_td, *,
                    stem="sweep_error", header=None, width=TEXT_WIDTH):
    """Residual between time-domain and Floquet rates across the sweep.

    A companion to fig_stability_sweep for when the two curves overlap so
    closely that the discrepancy is invisible on the main axes.
    """
    fig, ax = _new(width, ratio=0.45)
    resid = np.asarray(td_rate) - np.asarray(floquet_rate_at_td)
    ax.axhline(0.0, color=C["neutral"], lw=0.6, zorder=1)
    ax.plot(td_m1, resid, "o-", color=C["timedomain"], mfc="white",
            mew=1.0, lw=0.8, zorder=2)
    ax.set_xlabel(r"Contact mass $m_1$ [kg]")
    ax.set_ylabel(r"$\mathrm{Re}(\lambda)_{\mathrm{FEM}}"
                  r" - \mathrm{Re}(\lambda)_{\mathrm{Floquet}}$ [s$^{-1}$]")
    ax.margins(x=0.03)
    return fig, save_figure(fig, stem, header)
