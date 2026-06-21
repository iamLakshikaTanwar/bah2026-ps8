"""``lunaris`` command-line interface (Typer).

Exposes the console-script entry point ``lunaris`` (see ``pyproject.toml``).
The ``demo`` command is a real, working smoke test that generates the
deterministic synthetic Faustini scene and prints its ice-criterion summary;
``run`` wires up the full pipeline (finished by the integration agent).
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import load_config

app = typer.Typer(
    add_completion=False,
    help="Lunar Subsurface-Ice Detection & Mission-Planning Platform (BAH 2026 PS8).",
)
console = Console()


@app.command()
def run(
    config: str = typer.Option(
        "configs/faustini.yaml", "--config", "-c", help="Path to a YAML config."
    ),
) -> None:
    """Run the full ice-detection & mission-planning pipeline.

    (The pipeline body is completed by the integration agent; this command
    loads the config and invokes :func:`lunaris.pipeline.run_pipeline`.)
    """
    from .pipeline import run_pipeline  # local import: heavy deps / stub

    settings = load_config(config)
    console.print(f"[bold cyan]lunaris[/] v{__version__} — target "
                  f"[bold]{settings.target_name}[/]")
    try:
        results = run_pipeline(settings)
    except NotImplementedError as exc:
        console.print(f"[yellow]pipeline not yet wired:[/] {exc}")
        raise typer.Exit(code=0) from None
    console.print(f"[green]done[/] — {len(results)} result keys")


@app.command()
def demo(
    n: int = typer.Option(256, help="Grid edge length [px]."),
    seed: int = typer.Option(42, help="RNG seed."),
) -> None:
    """Generate the synthetic Faustini scene and print its ice-criterion summary.

    A self-contained smoke test that exercises the data backbone end-to-end
    without any external data.
    """
    import numpy as np

    from .io.synthetic import generate_faustini_scene

    console.print(f"[bold cyan]lunaris demo[/] — synthesising Faustini "
                  f"(n={n}, seed={seed})")
    scene = generate_faustini_scene(n=n, seed=seed)
    ice = scene.ice_truth
    bg = ~ice

    table = Table(title="Synthetic Faustini ice-criterion summary")
    table.add_column("region", style="bold")
    table.add_column("median CPR (L)", justify="right")
    table.add_column("median DOP (L)", justify="right")
    table.add_column("pixels", justify="right")
    table.add_row(
        "ICE truth",
        f"{np.median(scene.cpr_L[ice]):.3f}",
        f"{np.median(scene.dop_L[ice]):.3f}",
        f"{int(ice.sum())}",
    )
    table.add_row(
        "background",
        f"{np.median(scene.cpr_L[bg]):.3f}",
        f"{np.median(scene.dop_L[bg]):.3f}",
        f"{int(bg.sum())}",
    )
    console.print(table)
    console.print(
        f"shape={scene.shape}  resolution={scene.resolution_m} m  "
        f"crs set: {bool(scene.crs)}"
    )


@app.command()
def version() -> None:
    """Print the lunaris version."""
    console.print(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
