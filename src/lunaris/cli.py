"""``lunaris`` command-line interface (Typer).

Exposes the console-script entry point ``lunaris`` (see ``pyproject.toml``).
``demo`` runs the full nine-stage pipeline on the deterministic synthetic
Faustini scene and prints a rich mission summary; ``run`` does the same from a
YAML config; ``version`` prints the package version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import Settings, load_config

app = typer.Typer(
    add_completion=False,
    help="Lunar Subsurface-Ice Detection & Mission-Planning Platform (BAH 2026 PS8).",
)
console = Console()


def _fmt(x: Any, nd: int = 3) -> str:
    """Format a number compactly, or ``-`` for ``None``/NaN."""
    if x is None:
        return "-"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if xf != xf:  # NaN
        return "-"
    if abs(xf) >= 1e6 or (0 < abs(xf) < 1e-3):
        return f"{xf:.3e}"
    return f"{xf:,.{nd}f}"


def _print_summary(s: dict[str, Any]) -> None:
    """Render the headline mission-summary table from a pipeline summary dict."""
    console.print(Panel.fit(
        f"[bold cyan]LUNARIS[/] — {s['target']} "
        f"([{'green' if s['ingest_source'] == 'real-data' else 'yellow'}]"
        f"{s['ingest_source']}[/], grid {s['grid'][0]}×{s['grid'][1]} @ "
        f"{_fmt(s['resolution_m'], 1)} m)",
        border_style="cyan",
    ))

    t = Table(title="Mission summary", show_lines=False, header_style="bold")
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")

    t.add_row("[bold]— Ice detection —", "")
    t.add_row("O(1) LUT == direct rule", "yes" if s["lut_matches_rule"] else "NO")
    t.add_row("Precision", _fmt(s["precision"]))
    t.add_row("Recall", _fmt(s["recall"]))
    t.add_row("F1", _fmt(s["f1"]))
    t.add_row("Detected ice pixels", _fmt(s["ice_pixels"], 0))
    t.add_row("RF fusion CV-F1", f"{_fmt(s['rf_cv_f1_mean'])} ± {_fmt(s['rf_cv_f1_std'])}")
    t.add_row("Top fusion feature", str(s["top_feature"]))

    t.add_row("[bold]— Terrain —", "")
    t.add_row("Hurst exponent", _fmt(s["hurst_exponent"]))
    t.add_row("Cold-trap pixels (<110 K)", _fmt(s["cold_trap_pixels"], 0))
    t.add_row("Median slope [deg]", _fmt(s["median_slope_deg"]))

    t.add_row("[bold]— Landing & traverse —", "")
    t.add_row("Landing site (r, c)", f"{tuple(s['landing_site'])}")
    t.add_row("Landing suitability", _fmt(s["landing_score"]))
    t.add_row("AHP consistency ratio", _fmt(s["ahp_consistency_ratio"]))
    t.add_row("Ice access (r, c)", f"{tuple(s['ice_access'])}")
    t.add_row("Traverse length", f"{_fmt(s['traverse_length_m'], 1)} m "
                                  f"({_fmt(s['traverse_length_km'])} km)")
    t.add_row("Energy feasible", "[green]yes[/]" if s["energy_feasible"]
                                 else "[red]no[/]")
    t.add_row("Dark dwell [h] (≤70)", _fmt(s["dark_hours"]))
    t.add_row("Min SOC", _fmt(s["min_soc"]))

    t.add_row("[bold]— Ice inventory (top 5 m) —", "")
    t.add_row("Ice area [m²]", _fmt(s["ice_area_m2"], 0))
    t.add_row("Ice volume fraction", _fmt(s["ice_fraction"]))
    t.add_row("Ice volume [m³]", _fmt(s["ice_volume_m3"], 0))
    ci = s["ice_mass_ci_t"]
    t.add_row("Ice mass [t]", f"{_fmt(s['ice_mass_t'], 1)} "
                              f"(95% CI {_fmt(ci[0], 1)}–{_fmt(ci[1], 1)})")
    t.add_row("LCROSS cross-check [t]",
              _fmt(s["lcross_cross_check_kg"] / 1000.0, 1))
    console.print(t)


def _run_and_report(settings: Settings) -> None:
    """Execute the pipeline and print outputs + the summary table."""
    from .pipeline import run_pipeline  # local import: heavy deps

    console.print(f"[bold cyan]lunaris[/] v{__version__} — target "
                  f"[bold]{settings.target_name}[/] → "
                  f"[dim]{settings.outputs_dir}[/]")
    with console.status("[cyan]running 9-stage pipeline…", spinner="dots"):
        results = run_pipeline(settings)

    _print_summary(results["summary"])

    out = Path(settings.outputs_dir)
    console.print(
        f"\n[green]✓ outputs written[/] → "
        f"[bold]{out / 'lunaris_report.html'}[/]\n"
        f"  • report HTML, 3-D terrain, figures/*.png, *.tif (COG), "
        f"results_summary.json"
    )


@app.command()
def run(
    config: str = typer.Option(
        "configs/faustini.yaml", "--config", "-c", help="Path to a YAML config."
    ),
) -> None:
    """Run the full ice-detection & mission-planning pipeline from a config."""
    settings = load_config(config)
    _run_and_report(settings)


@app.command()
def demo(
    n: int = typer.Option(256, "--n", help="Grid edge length [px]."),
    seed: int = typer.Option(42, "--seed", help="RNG seed."),
    out: str = typer.Option("outputs/faustini", "--out", help="Output directory."),
) -> None:
    """Run the full pipeline on the synthetic Faustini scene and report results.

    A self-contained, deterministic end-to-end demo: detection → terrain →
    fusion → landing → traverse → ice inventory → HTML report. No external data
    required.
    """
    settings = Settings(grid_size=n, seed=seed, outputs_dir=Path(out))
    _run_and_report(settings)


@app.command()
def version() -> None:
    """Print the lunaris version."""
    console.print(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
