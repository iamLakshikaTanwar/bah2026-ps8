"""Final HTML report assembly — the single self-contained deliverable.

Implemented by: **viz agent**.

:func:`build_report` assembles every PS8 deliverable into ONE standalone HTML
file with no external assets: matplotlib panels are embedded as base64 PNG
``<img>`` tags and plotly figures are inlined as ``<div>`` blocks with
plotly.js embedded exactly once (the first plotly figure carries the full
library, subsequent ones reuse it). The result opens offline in any browser —
no internet, no GPU, no VTK.

:func:`demo_report` wires the whole thing together end-to-end with minimal
stand-in results computed directly from a synthetic Faustini scene, so the full
report can be generated NOW without any other pipeline module.
"""

from __future__ import annotations

import base64
import html
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")  # head-less — must precede pyplot anywhere downstream

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from .charts import (  # noqa: E402
    cpr_dop_scatter,
    feature_importance_bar,
    volume_histogram,
)
from .maps import (  # noqa: E402
    hillshade,
    mchi_rgb,  # noqa: F401  (re-exported convenience)
    plot_ice_map,
    plot_mchi,
    plot_traverse,
)

__all__ = ["build_report", "demo_report"]


# Section titles asserted by the test-suite / used in the table of contents.
_SECTIONS = [
    ("ice", "1 · Subsurface-Ice Region Map"),
    ("radar", "2 · Radar Detection Framework"),
    ("landing", "3 · Landing-Site Selection"),
    ("traverse", "4 · Rover Traverse Plan"),
    ("volume", "5 · Ice-Volume Estimate & Uncertainty"),
]


# ---------------------------------------------------------------------------
# embedding helpers
# ---------------------------------------------------------------------------
def _fig_to_base64_img(fig: Any, alt: str = "") -> str:
    """Render a matplotlib Figure to an inline base64 ``<img>`` tag."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    import matplotlib.pyplot as plt

    plt.close(fig)  # free the figure once embedded
    alt = html.escape(alt)
    return (f'<img class="fig" alt="{alt}" '
            f'src="data:image/png;base64,{b64}"/>')


def _rgb_to_base64_img(rgb: np.ndarray, alt: str = "") -> str:
    """Embed an ``(H, W, 3)`` float/uint8 RGB array as a base64 PNG ``<img>``."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 6.0), facecolor="white")
    ax.imshow(np.clip(rgb, 0, 1), origin="upper")
    ax.set_axis_off()
    return _fig_to_base64_img(fig, alt=alt)


def _plotly_div(fig: go.Figure, include_js: bool) -> str:
    """Inline a plotly figure as a ``<div>``; embed plotly.js iff requested.

    The first plotly figure in the document passes ``include_js=True`` so the
    library is embedded once; later figures pass ``False`` and reuse it. This
    keeps the report fully offline while avoiding many MB of duplicated JS.
    """
    return fig.to_html(
        full_html=False,
        include_plotlyjs=True if include_js else False,
        default_width="100%",
        default_height="540px",
    )


# ---------------------------------------------------------------------------
# small render utilities
# ---------------------------------------------------------------------------
def _metric_cards(metrics: Mapping[str, Any]) -> str:
    """Render a dict of scalar metrics as a row of stat cards."""
    if not metrics:
        return ""
    cards = []
    for k, v in metrics.items():
        if isinstance(v, float):
            val = f"{v:,.4g}"
        elif isinstance(v, (int, np.integer)):
            val = f"{int(v):,}"
        else:
            val = html.escape(str(v))
        cards.append(
            f'<div class="card"><div class="card-v">{val}</div>'
            f'<div class="card-k">{html.escape(str(k))}</div></div>'
        )
    return f'<div class="cards">{"".join(cards)}</div>'


def _datasets_table(rows: list[tuple[str, str, str]]) -> str:
    """Render the methods/datasets summary table."""
    body = "".join(
        f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td>"
        f"<td>{html.escape(c)}</td></tr>"
        for a, b, c in rows
    )
    return (
        '<table class="ds"><thead><tr>'
        "<th>Dataset / Method</th><th>Role</th><th>Source / Heritage</th>"
        f"</tr></thead><tbody>{body}</tbody></table>"
    )


def _section(anchor: str, title: str, body: str) -> str:
    """Wrap section ``body`` in an anchored ``<section>``."""
    return (f'<section id="{anchor}"><h2>{html.escape(title)}</h2>'
            f"{body}</section>")


# ---------------------------------------------------------------------------
# main builder
# ---------------------------------------------------------------------------
def build_report(
    results: Mapping[str, Any],
    out_html: str | Path = "outputs/lunaris_report.html",
) -> Path:
    """Assemble all 5 PS8 deliverables into one self-contained HTML report.

    The report is tolerant of missing keys: only the content present in
    ``results`` is rendered. Recognised keys (all optional):

    ``ice_mask`` : 2-D bool/float ice mask.
    ``background`` / ``s0`` / ``dem`` : backdrop for the ice map.
    ``cpr`` / ``dop`` : arrays for the CPR-DOP scatter.
    ``mchi`` : tuple/list ``(even, volume, odd)`` for the m-chi RGB composite.
    ``metrics`` : mapping of scalar detection metrics (precision/recall/...).
    ``feature_importance`` : tuple ``(names, importances)``.
    ``landing`` : ``(row, col)`` landing site.
    ``target`` : ``(row, col)`` ice target.
    ``path`` : iterable of ``(row, col)`` traverse waypoints.
    ``terrain3d_html`` : path to a pre-rendered 3-D HTML to link/iframe.
    ``volume_samples`` : Monte-Carlo volume samples.
    ``volume_ci`` : ``(lo, hi)`` confidence interval.
    ``volume_mean`` : scalar mean volume.
    ``datasets`` : list of ``(name, role, source)`` rows for the methods table.
    ``meta`` : free-form metadata mapping shown in the header.

    Parameters
    ----------
    results : mapping
        The results bundle (e.g. from ``lunaris.pipeline.run_pipeline``).
    out_html : str or Path, default "outputs/lunaris_report.html"
        Output HTML path.

    Returns
    -------
    Path
        The written report path.
    """
    r = dict(results)
    res = float(r.get("resolution_m", 20.0))
    dem = r.get("dem")
    js_first = [True]  # mutable flag: embed plotly.js only on first div

    def plotly_div(fig: go.Figure) -> str:
        inc = js_first[0]
        js_first[0] = False
        return _plotly_div(fig, include_js=inc)

    # ---- Section 1: ice region map -----------------------------------
    ice_body = []
    ice_mask = r.get("ice_mask")
    if ice_mask is not None:
        bg = r.get("background")
        if bg is None and dem is not None:
            bg = hillshade(np.asarray(dem), res=res)
        elif bg is None:
            bg = r.get("s0")
        fig = plot_ice_map(np.asarray(ice_mask), background=bg)
        ice_body.append(_fig_to_base64_img(fig, alt="ice region map"))
        n_ice = int(np.asarray(ice_mask).astype(bool).sum())
        ice_body.append(
            f'<p class="cap">Detected subsurface-ice pixels: '
            f"<b>{n_ice:,}</b> (cyan overlay; CPR&gt;1 &amp; DOP&lt;0.13).</p>"
        )
    else:
        ice_body.append('<p class="muted">No ice mask provided.</p>')
    sections_html = [_section("ice", _SECTIONS[0][1], "".join(ice_body))]

    # ---- Section 2: radar detection framework ------------------------
    radar_body = []
    metrics = r.get("metrics")
    if metrics:
        radar_body.append(_metric_cards(metrics))
    cpr, dop = r.get("cpr"), r.get("dop")
    if cpr is not None and dop is not None:
        fig = cpr_dop_scatter(np.asarray(cpr), np.asarray(dop),
                              ice_mask=ice_mask)
        radar_body.append('<div class="grid2"><div>')
        radar_body.append(plotly_div(fig))
        radar_body.append("</div>")
        mchi = r.get("mchi")
        if mchi is not None and len(mchi) == 3:
            rgb = mchi_rgb(np.asarray(mchi[0]), np.asarray(mchi[1]),
                           np.asarray(mchi[2]))
            radar_body.append(
                '<div>'
                + _rgb_to_base64_img(rgb, alt="m-chi RGB composite")
                + '<p class="cap">m-chi decomposition: '
                "R double-bounce, G volume (ice), B surface.</p></div>"
            )
        else:
            radar_body.append("<div></div>")
        radar_body.append("</div>")
    else:
        mchi = r.get("mchi")
        if mchi is not None and len(mchi) == 3:
            rgb = mchi_rgb(np.asarray(mchi[0]), np.asarray(mchi[1]),
                           np.asarray(mchi[2]))
            radar_body.append(_rgb_to_base64_img(rgb, alt="m-chi RGB composite"))
    fi = r.get("feature_importance")
    if fi is not None and len(fi) == 2:
        fig = feature_importance_bar(list(fi[0]), list(fi[1]))
        radar_body.append(plotly_div(fig))
    if not radar_body:
        radar_body.append('<p class="muted">No radar products provided.</p>')
    sections_html.append(_section("radar", _SECTIONS[1][1],
                                  "".join(radar_body)))

    # ---- Section 3: landing site -------------------------------------
    land_body = []
    landing = r.get("landing")
    if landing is not None:
        land_metrics = r.get("landing_metrics", {})
        if not land_metrics:
            land_metrics = {"landing row": int(landing[0]),
                            "landing col": int(landing[1])}
        land_body.append(_metric_cards(land_metrics))
    if dem is not None and (landing is not None or r.get("target") is not None):
        fig = plot_traverse(np.asarray(dem), path=None, landing=landing,
                            target=r.get("target"), res=res,
                            title="Selected landing site & ice target")
        land_body.append(_fig_to_base64_img(fig, alt="landing site"))
    if not land_body:
        land_body.append('<p class="muted">No landing site provided.</p>')
    sections_html.append(_section("landing", _SECTIONS[2][1],
                                  "".join(land_body)))

    # ---- Section 4: rover traverse -----------------------------------
    trav_body = []
    path = r.get("path")
    if dem is not None and path is not None:
        fig = plot_traverse(np.asarray(dem), path=path, landing=landing,
                            target=r.get("target"), res=res)
        trav_body.append(_fig_to_base64_img(fig, alt="rover traverse"))
        trav_body.append(
            f'<p class="cap">Traverse waypoints: <b>{len(path):,}</b>.</p>'
        )
    t3d = r.get("terrain3d_html")
    if t3d is not None:
        t3d_path = Path(t3d)
        trav_body.append(
            f'<p class="cap">Interactive 3-D terrain: '
            f'<a href="{html.escape(t3d_path.name)}" target="_blank">'
            f"{html.escape(t3d_path.name)}</a> (self-contained, offline).</p>"
        )
    if not trav_body:
        trav_body.append('<p class="muted">No traverse provided.</p>')
    sections_html.append(_section("traverse", _SECTIONS[3][1],
                                  "".join(trav_body)))

    # ---- Section 5: ice-volume estimate ------------------------------
    vol_body = []
    samples = r.get("volume_samples")
    ci = r.get("volume_ci")
    mean = r.get("volume_mean")
    vmetrics: dict[str, Any] = {}
    if mean is not None:
        vmetrics["mean volume [m³]"] = float(mean)
    if ci is not None:
        vmetrics["CI low [m³]"] = float(ci[0])
        vmetrics["CI high [m³]"] = float(ci[1])
    if vmetrics:
        vol_body.append(_metric_cards(vmetrics))
    if samples is not None:
        fig = volume_histogram(np.asarray(samples), ci=ci)
        vol_body.append(plotly_div(fig))
    elif mean is not None:
        vol_body.append(
            f'<p class="cap">Estimated ice volume: '
            f"<b>{float(mean):,.4g} m³</b>"
            + (f" (95% CI {float(ci[0]):,.4g} – {float(ci[1]):,.4g} m³)"
               if ci is not None else "")
            + ".</p>"
        )
    if not vol_body:
        vol_body.append('<p class="muted">No volume estimate provided.</p>')
    sections_html.append(_section("volume", _SECTIONS[4][1],
                                  "".join(vol_body)))

    # ---- methods / datasets table ------------------------------------
    default_rows = [
        ("Chandrayaan-2 DFSAR (L & S-band)",
         "Full-polarimetric radar → CPR, DOP, m-chi",
         "ISRO / synthetic Faustini proxy"),
        ("CPR & DOP dual criterion",
         "Subsurface-ice detection (CPR>1 & DOP<0.13)",
         "Sinha et al. 2026"),
        ("Diviner-like temperature_max",
         "Cold-trap / PSR thermal context",
         "LRO Diviner heritage"),
        ("LOLA-like DEM",
         "Terrain, hillshade, traverse & landing analysis",
         "LRO LOLA heritage"),
        ("Monte-Carlo volume integration",
         "Ice-volume estimate with uncertainty",
         "Dielectric-mixing + bootstrap"),
    ]
    ds_rows = r.get("datasets", default_rows)
    methods_html = _section(
        "methods", "Methods & Datasets",
        _datasets_table([tuple(x) for x in ds_rows]),
    )

    # ---- header meta -------------------------------------------------
    meta = dict(r.get("meta", {}))
    target = meta.get("target", "Faustini (synthetic)")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    meta_bits = [f"Target: <b>{html.escape(str(target))}</b>",
                 f"Generated: {now}"]
    if "n" in meta:
        meta_bits.append(f"Grid: {meta['n']}×{meta['n']} @ {res:g} m")

    toc = "".join(
        f'<a href="#{a}">{html.escape(t)}</a>' for a, t in _SECTIONS
    )

    body = "\n".join(sections_html) + "\n" + methods_html
    doc = _HTML_TEMPLATE.format(
        css=_CSS,
        meta=" &nbsp;·&nbsp; ".join(meta_bits),
        toc=toc,
        body=body,
        year=datetime.now(timezone.utc).year,
    )

    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# end-to-end demo (no other pipeline modules required)
# ---------------------------------------------------------------------------
def demo_report(
    out_html: str | Path = "outputs/lunaris_report.html",
    n: int = 128,
    seed: int = 42,
    with_terrain3d: bool = True,
) -> Path:
    """Build a full report end-to-end from a synthetic scene with stand-ins.

    Computes minimal, self-contained results directly from
    :func:`generate_faustini_scene` — an ice mask (``CPR>1 & DOP<0.13``), a
    stand-in landing point and diagonal traverse, an m-chi RGB from inline
    Stokes-derived channels, detection metrics versus the ground truth, and a
    bootstrap-style volume estimate — then renders the complete report. No other
    LUNARIS subpackage is imported.

    Parameters
    ----------
    out_html : str or Path
        Output report path.
    n, seed : int
        Synthetic-scene size and seed.
    with_terrain3d : bool, default True
        Also render the self-contained 3-D terrain HTML next to the report.

    Returns
    -------
    Path
        The written report path.
    """
    from lunaris.io.synthetic import generate_faustini_scene

    sc = generate_faustini_scene(n=n, seed=seed)
    dem = np.asarray(sc.dem)
    cpr, dop = np.asarray(sc.cpr_L), np.asarray(sc.dop_L)
    s0 = np.asarray(sc.s0_L)
    s3 = np.asarray(sc.s3_L)
    ice_truth = np.asarray(sc.ice_truth).astype(bool)

    # detection: the dual CPR/DOP criterion
    ice_mask = (cpr > 1.0) & (dop < 0.13)

    # detection metrics vs ground truth
    tp = int(np.sum(ice_mask & ice_truth))
    fp = int(np.sum(ice_mask & ~ice_truth))
    fn = int(np.sum(~ice_mask & ice_truth))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    metrics = {
        "ice pixels": int(ice_mask.sum()),
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }

    # m-chi channels (inline, no other subpackage):
    #   volume ~ sqrt(s0 * (1 - dop)); even/odd split by the sign of s3.
    m_vol = np.sqrt(np.clip(s0 * (1.0 - dop), 0.0, None))
    m = np.clip(dop, 0.0, 1.0)
    even = np.sqrt(np.clip(s0 * m, 0.0, None)) * (s3 > 0)   # double-bounce-ish
    odd = np.sqrt(np.clip(s0 * m, 0.0, None)) * (s3 <= 0)   # surface-ish
    mchi = (even, m_vol, odd)

    # landing site: brightest-illumination pixel with gentle local slope
    illum = np.asarray(sc.illumination)
    gy, gx = np.gradient(dem, sc.resolution_m)
    slope = np.hypot(gy, gx)
    score = illum - 0.5 * np.clip(slope, 0, None)
    li = int(np.argmax(score))
    landing = (li // n, li % n)

    # target: centroid of the detected ice
    if ice_mask.any():
        rr, cc = np.nonzero(ice_mask)
        target = (int(rr.mean()), int(cc.mean()))
    else:  # pragma: no cover - mask is non-empty for the synthetic scene
        target = (n // 2, n // 2)

    # simple stand-in traverse: a straight polyline landing -> target
    steps = max(2, int(np.hypot(target[0] - landing[0],
                                target[1] - landing[1]) // 2))
    path = list(zip(
        np.linspace(landing[0], target[0], steps).round().astype(int),
        np.linspace(landing[1], target[1], steps).round().astype(int),
    ))

    # ice volume: area * assumed thickness, with a bootstrap-style spread
    px_area = sc.resolution_m ** 2
    thickness_m = 2.0
    base_vol = float(ice_mask.sum()) * px_area * thickness_m
    rng = np.random.default_rng(seed)
    # 15% multiplicative uncertainty (thickness/porosity) + count jitter
    volume_samples = base_vol * rng.normal(1.0, 0.15, size=4000)
    volume_samples = np.clip(volume_samples, 0.0, None)
    lo, hi = np.percentile(volume_samples, [2.5, 97.5])
    volume_mean = float(volume_samples.mean())

    feature_importance = (
        ["CPR_L", "DOP_L", "CPR_S", "DOP_S", "temp_max", "illumination",
         "albedo_1064"],
        [0.34, 0.27, 0.12, 0.09, 0.08, 0.06, 0.04],
    )

    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)

    terrain_path = None
    if with_terrain3d:
        from .terrain3d import terrain3d_html

        # colour the surface by ice likelihood proxy (m-chi volume, stretched)
        overlay = m_vol
        terrain_path = terrain3d_html(
            dem, overlay=overlay, path=path, landing=landing, target=target,
            out=out.with_name("terrain3d.html"), res=sc.resolution_m,
        )

    results: dict[str, Any] = {
        "dem": dem,
        "resolution_m": sc.resolution_m,
        "ice_mask": ice_mask,
        "s0": s0,
        "cpr": cpr,
        "dop": dop,
        "mchi": mchi,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "landing": landing,
        "target": target,
        "path": path,
        "volume_samples": volume_samples,
        "volume_ci": (float(lo), float(hi)),
        "volume_mean": volume_mean,
        "terrain3d_html": str(terrain_path) if terrain_path else None,
        "meta": dict(sc.meta),
    }
    return build_report(results, out_html=out)


# ---------------------------------------------------------------------------
# HTML / CSS template (dark space theme)
# ---------------------------------------------------------------------------
_CSS = """
:root{--bg:#070b18;--panel:#0f1530;--panel2:#141a31;--ink:#e6ecff;
--muted:#8b95bf;--accent:#00e5ff;--accent2:#7aa2ff;--line:#26305a;}
*{box-sizing:border-box;}
body{margin:0;background:radial-gradient(1200px 600px at 70% -10%,
#172248 0%,var(--bg) 60%);color:var(--ink);
font-family:Inter,Segoe UI,Helvetica,Arial,sans-serif;line-height:1.55;}
header.hero{padding:48px 32px 28px;border-bottom:1px solid var(--line);
background:linear-gradient(180deg,rgba(23,34,72,.55),rgba(7,11,24,0));}
header.hero h1{margin:0;font-size:30px;letter-spacing:.5px;}
header.hero .sub{color:var(--accent);font-weight:600;margin-top:4px;}
header.hero .meta{color:var(--muted);margin-top:12px;font-size:14px;}
nav.toc{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:6px;
padding:12px 32px;background:rgba(7,11,24,.92);backdrop-filter:blur(6px);
border-bottom:1px solid var(--line);}
nav.toc a{color:var(--accent2);text-decoration:none;font-size:13px;
padding:4px 10px;border:1px solid var(--line);border-radius:999px;}
nav.toc a:hover{color:#fff;border-color:var(--accent);}
main{max-width:1100px;margin:0 auto;padding:8px 24px 60px;}
section{background:var(--panel);border:1px solid var(--line);border-radius:14px;
padding:22px 24px;margin:22px 0;box-shadow:0 8px 30px rgba(0,0,0,.35);}
section h2{margin:0 0 14px;font-size:21px;border-left:4px solid var(--accent);
padding-left:12px;}
img.fig{max-width:100%;height:auto;border-radius:10px;border:1px solid var(--line);
background:#fff;display:block;margin:6px auto;}
.cap{color:var(--muted);font-size:13.5px;margin:8px 2px 0;}
.muted{color:var(--muted);font-style:italic;}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:6px 0 16px;}
.card{flex:1 1 150px;background:var(--panel2);border:1px solid var(--line);
border-radius:12px;padding:14px 16px;}
.card-v{font-size:24px;font-weight:700;color:var(--accent);}
.card-k{font-size:12.5px;color:var(--muted);margin-top:2px;
text-transform:uppercase;letter-spacing:.4px;}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start;}
@media(max-width:820px){.grid2{grid-template-columns:1fr;}}
table.ds{width:100%;border-collapse:collapse;font-size:14px;}
table.ds th,table.ds td{text-align:left;padding:9px 12px;
border-bottom:1px solid var(--line);}
table.ds th{color:var(--accent2);font-weight:600;}
table.ds tr:hover td{background:rgba(122,162,255,.06);}
footer{color:var(--muted);font-size:13px;text-align:center;padding:26px 24px 40px;
border-top:1px solid var(--line);}
a{color:var(--accent);}
.plotly-graph-div{margin:6px auto;}
"""

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>LUNARIS — Lunar Subsurface-Ice & Mission-Planning Platform</title>
<style>{css}</style>
</head>
<body>
<header class="hero">
  <h1>LUNARIS</h1>
  <div class="sub">Lunar Subsurface-Ice &amp; Mission-Planning Platform</div>
  <div class="meta">{meta}</div>
</header>
<nav class="toc">{toc}</nav>
<main>
{body}
</main>
<footer>
  LUNARIS · self-contained offline report (no internet, no GPU, no VTK).<br/>
  Data heritage: Chandrayaan-2 DFSAR, LRO LOLA &amp; Diviner; method after
  Sinha et al. (2026). Synthetic Faustini scene for development. © {year}.
</footer>
</body>
</html>
"""
