from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal, Sequence

import numpy as np
import torch
from pydantic import Field, PrivateAttr

from qseek.corrections.base import TravelTimeCorrections
from qseek.models.catalog import EventCatalog
from qseek.utils import NSL, PhaseDescription, _NSL, weighted_median

if TYPE_CHECKING:
    from qseek.models.station import StationInventory
    from qseek.octree import Node, Octree

logger = logging.getLogger(__name__)

# Auto-detect GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Station corrections device: %s", DEVICE)

WeightingMethod = Literal[
    "none",
    "confidence",
    "semblance",
    "add-confidence-semblance",
    "mul-confidence-semblance",
]
StatisticMethod = Literal["median", "average"]
InterpolationMethod = Literal["nearest", "linear", "cubic"]


@dataclass
class DelayRecord:
    """Single traveltime delay observation."""

    nsl: _NSL
    phase: str
    delay: float  # seconds (observed - modelled)
    confidence: float = 1.0
    semblance: float = 0.0
    event_east: float = 0.0
    event_north: float = 0.0
    event_depth: float = 0.0


def compute_weight(
    weighting: WeightingMethod,
    confidence: float,
    semblance: float,
) -> float:
    """Compute weight for a single delay record."""
    if weighting == "none":
        return 1.0
    elif weighting == "confidence":
        return confidence
    elif weighting == "semblance":
        return semblance
    elif weighting == "add-confidence-semblance":
        return confidence + semblance
    elif weighting == "mul-confidence-semblance":
        return confidence * semblance
    return 1.0


def extract_delays_from_rundir(
    rundir: Path,
    min_distance_border: float,
    min_num_picks: int,
) -> list[DelayRecord]:
    """Extract traveltime delays from a rundir's event catalog."""
    logger.info("extracting delays from rundir: %s", rundir)
    catalog = EventCatalog.load_rundir(rundir)
    records: list[DelayRecord] = []

    for event in catalog:
        # Filter by border distance
        if event.distance_border < min_distance_border:
            continue

        # Filter by minimum number of picks
        if event.n_picks < min_num_picks:
            continue

        for receiver in event.receivers:
            for phase, phase_det in receiver.phase_arrivals.items():
                if phase_det.observed is None:
                    continue
                if phase_det.traveltime_delay is None:
                    continue

                delay_sec = phase_det.traveltime_delay.total_seconds()
                confidence = phase_det.observed.detection_value

                records.append(
                    DelayRecord(
                        nsl=receiver.nsl,
                        phase=phase,
                        delay=delay_sec,
                        confidence=confidence,
                        semblance=event.semblance,
                        event_east=event.east_shift,
                        event_north=event.north_shift,
                        event_depth=event.effective_depth,
                    )
                )

    logger.info("extracted %d delay records from %s", len(records), rundir)
    return records


def _plot_station_delay_bars(
    station_delays: dict,
    phases: list[str],
    output_dir: Path,
) -> None:
    """Plot station correction delay values as a bar chart per phase."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for phase in phases:
        stations_with_phase = []
        delays_for_phase = []

        for nsl_str, phase_delays in station_delays.items():
            if phase in phase_delays:
                stations_with_phase.append(nsl_str)
                delays_for_phase.append(phase_delays[phase])

        if not stations_with_phase:
            continue

        fig, ax = plt.subplots(figsize=(max(12, len(stations_with_phase) * 0.4), 6))
        x = np.arange(len(stations_with_phase))
        colors = ["#e74c3c" if d > 0 else "#3498db" for d in delays_for_phase]
        ax.bar(x, delays_for_phase, color=colors, alpha=0.8, edgecolor="black",
               linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(stations_with_phase, rotation=90, fontsize=7)
        ax.set_ylabel("Delay (s)")
        ax.set_title(f"Station Corrections - Phase: {phase}")
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        safe_phase = phase.replace(":", "_")
        fig.savefig(output_dir / f"delays_{safe_phase}.png", dpi=150)
        plt.close(fig)
        logger.info("saved delay bar chart: %s", output_dir / f"delays_{safe_phase}.png")


def _plot_station_residual_histogram(
    records: list[DelayRecord],
    weighting: WeightingMethod,
    output_dir: Path,
) -> None:
    """Plot travel time residual histograms per station per phase.

    Creates a histogram for each (station, phase) combination showing the
    distribution of travel time residuals, with vertical lines for:
    - Mean (unweighted)
    - Mean weighted
    - Median (unweighted)
    - Median weighted
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group records by (nsl, phase)
    grouped: dict[tuple[str, str], list[DelayRecord]] = defaultdict(list)
    for rec in records:
        grouped[(rec.nsl.pretty, rec.phase)].append(rec)

    # Create diverging colormap: red → white → blue
    cmap_colors = [
        (0.7, 0.1, 0.1),   # dark red
        (0.9, 0.3, 0.3),   # red
        (1.0, 0.6, 0.5),   # light red/salmon
        (1.0, 0.85, 0.8),  # very light red
        (0.95, 0.95, 0.95), # near white
        (0.8, 0.85, 1.0),  # very light blue
        (0.5, 0.65, 1.0),  # light blue
        (0.3, 0.45, 0.9),  # blue
        (0.1, 0.2, 0.6),   # dark blue
    ]
    cmap = LinearSegmentedColormap.from_list("residual", cmap_colors, N=256)

    for (nsl_str, phase), recs in grouped.items():
        if len(recs) < 10:
            continue

        delays = np.array([r.delay for r in recs])
        weights = np.array(
            [compute_weight(weighting, r.confidence, r.semblance) for r in recs]
        )

        # Compute statistics
        mean_val = float(np.mean(delays))
        mean_wt = float(np.average(delays, weights=weights))
        median_val = float(np.median(delays))
        try:
            median_wt = float(weighted_median(delays, weights))
        except (ValueError, IndexError):
            median_wt = median_val

        # Create histogram
        fig, ax = plt.subplots(figsize=(10, 6))

        # Determine bin range symmetrically around 0
        max_abs = max(abs(delays.min()), abs(delays.max()))
        max_abs = min(max_abs, 0.5)  # cap at 0.5s for readability
        n_bins = 40
        bins = np.linspace(-max_abs, max_abs, n_bins + 1)

        # Color each bar based on its bin center value
        n_vals, bin_edges, patches = ax.hist(
            delays, bins=bins, edgecolor="black", linewidth=0.4, alpha=0.85,
        )
        # Color bars: red for negative, blue for positive
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        norm_vals = (bin_centers - bin_centers.min()) / (
            bin_centers.max() - bin_centers.min() + 1e-10
        )
        for val, patch in zip(norm_vals, patches):
            patch.set_facecolor(cmap(val))

        # Add statistic lines
        line_styles = [
            (mean_val, "--", 1.5, "0.5", f"{mean_val:.2f} s  Mean"),
            (mean_wt, "-.", 2.0, "0.2", f"{mean_wt:.2f} s  Mean wt."),
            (median_val, ":", 1.5, "0.5", f"{median_val:.2f} s  Median"),
            (median_wt, ":", 2.0, "0.2", f"{median_wt:.2f} s  Median wt."),
        ]
        for val, ls, lw, color, label in line_styles:
            ax.axvline(x=val, linestyle=ls, linewidth=lw, color=color, label=label)

        # Zero line
        ax.axvline(x=0, color="black", linewidth=0.8)

        # Labels and styling
        phase_short = phase.split(":")[-1] if ":" in phase else phase
        ax.text(
            0.02, 0.08,
            f"Phase: {phase_short} ({len(recs)} picks)\nStation: {nsl_str}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        ax.set_xlabel("Travel time residual [s]", fontsize=11)
        ax.set_ylabel("Observations #", fontsize=11)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.tick_params(labelsize=9)
        plt.tight_layout()

        safe_nsl = nsl_str.replace(".", "_")
        safe_phase = phase.replace(":", "_")
        filename = output_dir / f"residuals_{safe_nsl}_{safe_phase}.png"
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        logger.info("saved residual histogram: %s", filename)


def _plot_ssst_3d_volume(
    grid_coords: np.ndarray,
    grid_delays: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    """Plot SSST delay volume for selected stations.

    Uses plotly for interactive 3D volume rendering if available,
    otherwise falls back to matplotlib multi-slice approach.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract regular grid axes
    east_ax = np.unique(grid_coords[:, 0])
    north_ax = np.unique(grid_coords[:, 1])
    depth_ax = np.unique(grid_coords[:, 2])
    ne, nn, nd = len(east_ax), len(north_ax), len(depth_ax)

    if ne < 2 or nn < 2 or nd < 2:
        logger.warning("grid too small for volume plot")
        return

    # Build index lookup
    e_idx = np.searchsorted(east_ax, grid_coords[:, 0])
    n_idx = np.searchsorted(north_ax, grid_coords[:, 1])
    d_idx = np.searchsorted(depth_ax, grid_coords[:, 2])

    for phase, nsl_delays in grid_delays.items():
        for nsl_str, delays_flat in nsl_delays.items():
            if np.all(delays_flat == 0):
                continue

            vol = np.zeros((ne, nn, nd), dtype=np.float32)
            vol[e_idx, n_idx, d_idx] = delays_flat

            vmax = max(abs(np.nanmin(vol)), abs(np.nanmax(vol)))
            if vmax < 1e-8:
                continue

            phase_short = phase.split(":")[-1] if ":" in phase else phase
            safe_nsl = nsl_str.replace(".", "_")
            safe_phase = phase.replace(":", "_")
            title = (
                f"Delay volume — Phase: {phase_short}, "
                f"Station: {nsl_str}"
            )

            # Try plotly first for true volume rendering
            try:
                _plot_volume_plotly(
                    east_ax, north_ax, depth_ax, vol,
                    vmax, title, safe_nsl, safe_phase, output_dir,
                )
            except (ImportError, Exception) as e:
                logger.info("plotly unavailable (%s), using matplotlib", e)
                _plot_volume_matplotlib(
                    east_ax, north_ax, depth_ax, vol,
                    vmax, title, safe_nsl, safe_phase, output_dir,
                )

        # 2D slice plots
        _plot_ssst_slices(
            east_ax, north_ax, depth_ax, (ne, nn, nd),
            nsl_delays, e_idx, n_idx, d_idx,
            phase, output_dir,
        )


def _plot_volume_plotly(
    east_ax: np.ndarray,
    north_ax: np.ndarray,
    depth_ax: np.ndarray,
    vol: np.ndarray,
    vmax: float,
    title: str,
    safe_nsl: str,
    safe_phase: str,
    output_dir: Path,
) -> None:
    """Plot delay volume using plotly's Volume trace."""
    import plotly.graph_objects as go

    ne, nn, nd = vol.shape

    # Create meshgrid for plotly (needs flattened X, Y, Z, values)
    E, N, D = np.meshgrid(east_ax, north_ax, -depth_ax, indexing="ij")

    fig = go.Figure(data=go.Volume(
        x=E.flatten(),
        y=N.flatten(),
        z=D.flatten(),
        value=vol.flatten(),
        isomin=-vmax,
        isomax=vmax,
        opacity=0.15,
        surface_count=25,
        colorscale="RdBu_r",
        colorbar=dict(
            title="Delay (s)",
            orientation="h",
            y=-0.1,
            thickness=15,
        ),
        caps=dict(x_show=True, y_show=True, z_show=True),
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        scene=dict(
            xaxis_title="Easting (m)",
            yaxis_title="Northing (m)",
            zaxis_title="Depth (m)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        width=1000,
        height=800,
        margin=dict(l=20, r=20, t=60, b=80),
    )

    # Save as interactive HTML
    html_file = output_dir / f"ssst_3d_{safe_nsl}_{safe_phase}.html"
    fig.write_html(str(html_file))
    logger.info("saved SSST 3D volume (interactive): %s", html_file)

    # Try to save as static PNG too
    try:
        png_file = output_dir / f"ssst_3d_{safe_nsl}_{safe_phase}.png"
        fig.write_image(str(png_file), scale=2)
        logger.info("saved SSST 3D volume (PNG): %s", png_file)
    except (ImportError, ValueError) as e:
        logger.info("static PNG export unavailable (%s), HTML saved", e)


def _plot_volume_matplotlib(
    east_ax: np.ndarray,
    north_ax: np.ndarray,
    depth_ax: np.ndarray,
    vol: np.ndarray,
    vmax: float,
    title: str,
    safe_nsl: str,
    safe_phase: str,
    output_dir: Path,
) -> None:
    """Fallback: plot delay volume as multi-slice surfaces in matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    ne, nn, nd = vol.shape
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu_r
    neg_depth = -depth_ax

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot multiple depth slices as semi-transparent surfaces
    n_slices = min(nd, 8)
    slice_indices = np.linspace(0, nd - 1, n_slices, dtype=int)

    for di in slice_indices:
        E, N = np.meshgrid(east_ax, north_ax, indexing="ij")
        Z = np.full_like(E, neg_depth[di])
        colors = cmap(norm(vol[:, :, di]))
        ax.plot_surface(
            E, N, Z, facecolors=colors,
            shade=False, alpha=0.4, rstride=1, cstride=1,
        )

    # Plot front face (north = max)
    E_f, D_f = np.meshgrid(east_ax, neg_depth, indexing="ij")
    N_f = np.full_like(E_f, north_ax[-1])
    ax.plot_surface(
        E_f, N_f, D_f, facecolors=cmap(norm(vol[:, -1, :])),
        shade=False, alpha=0.6, rstride=1, cstride=1,
    )

    # Plot right face (east = max)
    N_s, D_s = np.meshgrid(north_ax, neg_depth, indexing="ij")
    E_s = np.full_like(N_s, east_ax[-1])
    ax.plot_surface(
        E_s, N_s, D_s, facecolors=cmap(norm(vol[-1, :, :])),
        shade=False, alpha=0.6, rstride=1, cstride=1,
    )

    ax.set_xlabel("Easting (m)", fontsize=10, labelpad=10)
    ax.set_ylabel("Northing (m)", fontsize=10, labelpad=10)
    ax.set_zlabel("Depth (m)", fontsize=10, labelpad=10)
    ax.set_title(title, fontsize=12, pad=15)
    ax.tick_params(labelsize=8)
    ax.view_init(elev=25, azim=-60)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=ax, orientation="horizontal",
        shrink=0.6, pad=0.08, aspect=30,
    )
    cbar.set_label("Delay (s)", fontsize=10)

    filename = output_dir / f"ssst_3d_{safe_nsl}_{safe_phase}.png"
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved SSST 3D plot: %s", filename)


def _plot_ssst_slices(
    east_ax: np.ndarray,
    north_ax: np.ndarray,
    depth_ax: np.ndarray,
    grid_shape: tuple[int, int, int],
    nsl_delays: dict[str, np.ndarray],
    e_idx: np.ndarray,
    n_idx: np.ndarray,
    d_idx: np.ndarray,
    phase: str,
    output_dir: Path,
) -> None:
    """Plot 2D slice views using smooth contourf instead of scatter."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    ne, nn, nd = grid_shape

    # Average delays across all stations for overview
    all_vols = []
    for delays_flat in nsl_delays.values():
        if np.all(delays_flat == 0):
            continue
        vol = np.zeros((ne, nn, nd), dtype=np.float32)
        vol[e_idx, n_idx, d_idx] = delays_flat
        all_vols.append(vol)

    if not all_vols:
        return
    avg_vol = np.mean(all_vols, axis=0)

    vmax = max(abs(np.nanmin(avg_vol)), abs(np.nanmax(avg_vol)))
    if vmax < 1e-8:
        return
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    phase_short = phase.split(":")[-1] if ":" in phase else phase
    safe_phase = phase.replace(":", "_")

    # Top view: average over depth
    slices = [
        ("top_view", "NE",
         east_ax, north_ax,
         avg_vol.mean(axis=2).T,  # (nn, ne)
         "Easting (m)", "Northing (m)"),
        ("front_view", "ED",
         east_ax, -depth_ax,
         avg_vol.mean(axis=1).T,  # (nd, ne)
         "Easting (m)", "-Depth (m)"),
        ("side_view", "ND",
         north_ax, -depth_ax,
         avg_vol.mean(axis=0).T,  # (nd, nn)
         "Northing (m)", "-Depth (m)"),
    ]

    for suffix, name, x_ax, y_ax, data_2d, xlabel, ylabel in slices:
        fig, ax = plt.subplots(figsize=(10, 8))

        cf = ax.contourf(
            x_ax, y_ax, data_2d,
            levels=50,
            cmap="RdBu_r",
            norm=norm,
        )

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(
            f"SSST Average Delay — {name} view, Phase: {phase_short}",
            fontsize=12,
        )

        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label("Delay (s)", fontsize=10)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=9)
        plt.tight_layout()

        filename = output_dir / f"ssst_{suffix}_{safe_phase}.png"
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        logger.info("saved SSST slice plot: %s", filename)


class StationCorrections(TravelTimeCorrections):
    """Station corrections extracted from previous runs.

    Station corrections can be extracted from previous runs to refine the
    localisation accuracy. The corrections can also help to improve the
    semblance and find more events in a dataset.
    """

    corrections: Literal["StationCorrections"] = "StationCorrections"

    import_rundirs: list[Path] = Field(
        default_factory=list,
        description="Path to rundir, to extract the station corrections from.",
    )
    plot_corrections: bool = Field(
        default=False,
        description="Plot the station corrections statistics.",
    )
    statistic: StatisticMethod = Field(
        default="median",
        description="Arithmetic measure for the traveltime delays. "
        "Choose from median and average.",
    )
    weighting: WeightingMethod = Field(
        default="mul-confidence-semblance",
        description="Weighting of the traveltime delays. Choose from none, "
        "confidence, semblance, add-confidence-semblance and "
        "mul-confidence-semblance.",
    )
    min_num_station_picks: int = Field(
        default=50,
        description="Minimum number of picks at a station required to "
        "calculate station corrections.",
    )
    min_distance_border: float = Field(
        default=500.0,
        description="Minimum event distance from the border of the octree grid.",
    )
    min_num_picks: int = Field(
        default=3,
        description="Minimum number of picks per event to be included "
        "in the statistics.",
    )

    _station_delays: dict[str, dict[str, float]] = PrivateAttr(default_factory=dict)

    @property
    def n_stations(self) -> int:
        return len(self._station_delays)

    def get_delay(
        self,
        station_nsl: NSL,
        phase: PhaseDescription,
        node: Node | None = None,
    ) -> float:
        nsl_str = station_nsl.pretty
        if nsl_str not in self._station_delays:
            return 0.0
        return self._station_delays[nsl_str].get(phase, 0.0)

    async def get_delays(
        self,
        station_nsls: Sequence[NSL],
        phase: PhaseDescription,
        nodes: Sequence[Node],
    ) -> np.ndarray:
        delays = np.array(
            [self.get_delay(nsl, phase) for nsl in station_nsls],
            dtype=np.float32,
        )
        # Shape (1, n_stations) broadcasts to (n_nodes, n_stations)
        return delays[np.newaxis, :]

    async def prepare(
        self,
        stations: StationInventory,
        octree: Octree,
        phases: Iterable[PhaseDescription],
        rundir: Path,
    ) -> None:
        logger.info("preparing StationCorrections")
        phases_list = list(phases)

        # Collect all delay records from import rundirs
        all_records: list[DelayRecord] = []
        for rd in self.import_rundirs:
            rd_path = Path(rd).resolve()
            if not rd_path.is_dir():
                logger.warning("rundir not found: %s", rd_path)
                continue
            records = await asyncio.to_thread(
                extract_delays_from_rundir,
                rd_path,
                self.min_distance_border,
                self.min_num_picks,
            )
            all_records.extend(records)

        if not all_records:
            logger.warning("no delay records found, station corrections will be zero")
            return

        # Group records by (NSL, phase)
        grouped: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for rec in all_records:
            w = compute_weight(self.weighting, rec.confidence, rec.semblance)
            grouped[rec.nsl.pretty][rec.phase].append((rec.delay, w))

        # Compute statistics
        self._station_delays = {}
        for nsl_str, phase_data in grouped.items():
            self._station_delays[nsl_str] = {}
            for phase, delay_weights in phase_data.items():
                delays = np.array([d for d, _ in delay_weights])
                weights = np.array([w for _, w in delay_weights])

                if len(delays) < self.min_num_station_picks:
                    logger.debug(
                        "skipping %s/%s: only %d picks (min %d)",
                        nsl_str, phase, len(delays), self.min_num_station_picks,
                    )
                    continue

                if self.statistic == "median":
                    delay_val = float(weighted_median(delays, weights))
                else:  # average
                    delay_val = float(np.average(delays, weights=weights))

                self._station_delays[nsl_str][phase] = delay_val

        logger.info(
            "computed station corrections for %d stations", len(self._station_delays)
        )

        # Save corrections
        corrections_dir = rundir / "station_corrections"
        corrections_dir.mkdir(exist_ok=True)
        corrections_file = corrections_dir / "corrections.json"
        corrections_file.write_text(json.dumps(self._station_delays, indent=2))
        logger.info("saved station corrections to %s", corrections_file)

        # Plot residual histograms + bar charts
        if self.plot_corrections:
            await asyncio.to_thread(
                _plot_station_delay_bars,
                self._station_delays,
                phases_list,
                corrections_dir,
            )
            await asyncio.to_thread(
                _plot_station_residual_histogram,
                all_records,
                self.weighting,
                corrections_dir,
            )


class SourceSpecificStationCorrections(TravelTimeCorrections):
    """Source-Specific Station Corrections (SSST).

    Computes spatially-varying station corrections on the octree grid.
    Uses GPU CUDA acceleration via PyTorch when available.

    The delay grid is stored as a regular 3D array and interpolated using
    scipy's RegularGridInterpolator for instant build and O(1) queries.
    Results are cached to disk so that subsequent runs skip GPU computation.
    """

    corrections: Literal["SourceSpecificStationCorrections"] = (
        "SourceSpecificStationCorrections"
    )

    import_rundirs: list[Path] = Field(
        default_factory=list,
        description="Path to rundir, to extract the station corrections from.",
    )
    weighting: WeightingMethod = Field(
        default="mul-confidence-semblance",
        description="Weighting of the traveltime delays.",
    )
    min_confidence: float = Field(
        default=5.0,
        description="Minimum cumulative pick confidence within the sphere "
        "surrounding the node. If the cumulative pick confidence inside the "
        "sphere is below this value, the sphere radius is increased until "
        "enough picks are inside the sphere.",
    )
    min_distance_border: float = Field(
        default=500.0,
        description="Minimum event distance from the border of the octree grid.",
    )
    min_num_picks: int = Field(
        default=3,
        description="Minimum number of picks per event to be included in "
        "the statistics. Higher values will result in fewer events.",
    )
    spatial_weighting_exponent: float = Field(
        default=3.0,
        description="The exponent of the spatial weighting function "
        "around the sphere.",
    )
    resolution_octree_level: int = Field(
        default=0,
        description="The octree level (resolution) to use for the station "
        "corrections. This is the SSST grid spacing.",
    )
    delay_interpolation_method: InterpolationMethod = Field(
        default="linear",
        description="The interpolation method to use for interpolating "
        "delays between nodes.",
    )

    # Internal state
    _grid_axes: tuple[np.ndarray, ...] = PrivateAttr(default=())
    _delay_grids: dict[str, dict[str, np.ndarray]] = PrivateAttr(
        default_factory=dict
    )
    _interpolators: dict[str, dict[str, object]] = PrivateAttr(
        default_factory=dict
    )

    @property
    def n_stations(self) -> int:
        if not self._delay_grids:
            return 0
        first_phase = next(iter(self._delay_grids))
        return len(self._delay_grids[first_phase])

    # ── GPU Computation ───────────────────────────────────────────────

    def _compute_ssst_grid_gpu(
        self,
        records: list[DelayRecord],
        grid_coords: np.ndarray,
        phases: list[str],
        station_nsls: list[str],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Compute SSST grid delays using GPU CUDA acceleration.

        Returns flat delay arrays per (phase, station).
        """
        n_grid = grid_coords.shape[0]
        device = DEVICE
        logger.info("computing SSST grid on %s (%d grid nodes)", device, n_grid)

        # Group records by (phase, nsl)
        phase_nsl_records: dict[str, dict[str, list[DelayRecord]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for rec in records:
            phase_nsl_records[rec.phase][rec.nsl.pretty].append(rec)

        grid_tensor = torch.tensor(grid_coords, dtype=torch.float32, device=device)
        result: dict[str, dict[str, np.ndarray]] = {}

        for phase in phases:
            result[phase] = {}
            if phase not in phase_nsl_records:
                continue

            for nsl_str in station_nsls:
                recs = phase_nsl_records[phase].get(nsl_str, [])
                if not recs:
                    result[phase][nsl_str] = np.zeros(n_grid, dtype=np.float32)
                    continue

                # Build tensors on GPU
                event_coords = torch.tensor(
                    [[r.event_east, r.event_north, r.event_depth] for r in recs],
                    dtype=torch.float32, device=device,
                )
                delays_t = torch.tensor(
                    [r.delay for r in recs], dtype=torch.float32, device=device,
                )
                base_weights = torch.tensor(
                    [compute_weight(self.weighting, r.confidence, r.semblance)
                     for r in recs],
                    dtype=torch.float32, device=device,
                )
                confidences = torch.tensor(
                    [r.confidence for r in recs], dtype=torch.float32, device=device,
                )

                # Distances and spatial weights: (n_grid, n_events)
                distances = torch.cdist(grid_tensor, event_coords)
                eps = torch.tensor(1.0, device=device)
                spatial_w = 1.0 / torch.pow(
                    torch.maximum(distances, eps), self.spatial_weighting_exponent,
                )

                # Combined weights and weighted average
                combined_w = spatial_w * base_weights.unsqueeze(0)
                w_sums = combined_w.sum(dim=1)
                w_delays = (combined_w * delays_t.unsqueeze(0)).sum(dim=1)
                cum_conf = (spatial_w * confidences.unsqueeze(0)).sum(dim=1)

                node_delays = torch.zeros(n_grid, dtype=torch.float32, device=device)

                # Where confidence is sufficient → weighted average
                valid = (w_sums > 0) & (cum_conf >= self.min_confidence)
                node_delays[valid] = w_delays[valid] / w_sums[valid]

                # Fallback for insufficient confidence → global weighted average
                insufficient = ~valid & (w_sums > 0)
                if insufficient.any():
                    total_w = base_weights.sum()
                    if total_w > 0:
                        node_delays[insufficient] = (
                            (delays_t * base_weights).sum() / total_w
                        )

                result[phase][nsl_str] = node_delays.cpu().numpy()

        return result

    # ── Regular Grid Interpolator ─────────────────────────────────────

    def _flat_to_3d(
        self,
        grid_coords: np.ndarray,
        flat_delays: dict[str, dict[str, np.ndarray]],
    ) -> tuple[
        tuple[np.ndarray, ...],
        dict[str, dict[str, np.ndarray]],
    ]:
        """Convert flat delay arrays to 3D regular grid arrays.

        Returns (grid_axes, delay_grids_3d).
        """
        east_ax = np.unique(grid_coords[:, 0])
        north_ax = np.unique(grid_coords[:, 1])
        depth_ax = np.unique(grid_coords[:, 2])

        ne, nn, nd = len(east_ax), len(north_ax), len(depth_ax)
        logger.info(
            "regular grid: %d × %d × %d = %d nodes",
            ne, nn, nd, ne * nn * nd,
        )

        # Build index lookup: (east, north, depth) → flat index
        e_idx = np.searchsorted(east_ax, grid_coords[:, 0])
        n_idx = np.searchsorted(north_ax, grid_coords[:, 1])
        d_idx = np.searchsorted(depth_ax, grid_coords[:, 2])

        grids_3d: dict[str, dict[str, np.ndarray]] = {}
        for phase, nsl_delays in flat_delays.items():
            grids_3d[phase] = {}
            for nsl_str, delays_1d in nsl_delays.items():
                arr = np.zeros((ne, nn, nd), dtype=np.float32)
                arr[e_idx, n_idx, d_idx] = delays_1d
                grids_3d[phase][nsl_str] = arr

        return (east_ax, north_ax, depth_ax), grids_3d

    def _build_interpolators(
        self,
        grid_axes: tuple[np.ndarray, ...],
        delay_grids: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, dict[str, object]]:
        """Build RegularGridInterpolator for each (phase, station).

        This is instant — no Delaunay triangulation needed.
        """
        from scipy.interpolate import RegularGridInterpolator

        method = self.delay_interpolation_method

        # Validate and select interpolation method
        if method == "nearest":
            scipy_method = "nearest"
        elif method == "linear":
            scipy_method = "linear"
        elif method == "cubic":
            # cubic requires scipy >= 1.10
            try:
                RegularGridInterpolator(
                    (np.array([0, 1]), np.array([0, 1]), np.array([0, 1])),
                    np.zeros((2, 2, 2)),
                    method="cubic",
                )
                scipy_method = "cubic"
            except ValueError:
                logger.warning("cubic interpolation not supported, using linear")
                scipy_method = "linear"
        else:
            logger.warning("unknown method '%s', falling back to linear", method)
            scipy_method = "linear"

        interpolators: dict[str, dict[str, object]] = {}
        for phase, nsl_delays in delay_grids.items():
            interpolators[phase] = {}
            for nsl_str, delays_3d in nsl_delays.items():
                interp = RegularGridInterpolator(
                    grid_axes,
                    delays_3d,
                    method=scipy_method,
                    bounds_error=False,
                    fill_value=0.0,
                )
                interpolators[phase][nsl_str] = interp

        n_total = sum(len(v) for v in interpolators.values())
        logger.info(
            "built %d interpolators (method=%s) — instant", n_total, scipy_method,
        )
        return interpolators

    # ── Cache Management ──────────────────────────────────────────────

    def _compute_cache_key(self) -> str:
        """Compute a hash key for cache invalidation."""
        key_parts = [
            str(sorted(str(p) for p in self.import_rundirs)),
            self.weighting,
            str(self.min_confidence),
            str(self.min_distance_border),
            str(self.min_num_picks),
            str(self.spatial_weighting_exponent),
            str(self.resolution_octree_level),
            self.delay_interpolation_method,
        ]

        for rd in sorted(self.import_rundirs, key=str):
            rd_path = Path(rd).resolve()
            for fname in ("detections.json", "detections_receivers.json"):
                f = rd_path / fname
                if f.exists():
                    stat = f.stat()
                    key_parts.append(f"{f}:{stat.st_mtime}:{stat.st_size}")

        return hashlib.sha256("|".join(key_parts).encode()).hexdigest()[:16]

    def _save_cache(
        self,
        cache_dir: Path,
        phases: list[str],
        station_nsls: list[str],
    ) -> None:
        """Save SSST grid and axes to cache files."""
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_key = self._compute_cache_key()
        (cache_dir / "cache_key.txt").write_text(cache_key)

        # Save grid axes
        np.savez(
            cache_dir / "grid_axes.npz",
            east=self._grid_axes[0],
            north=self._grid_axes[1],
            depth=self._grid_axes[2],
        )

        # Save 3D delay grids
        for phase, nsl_delays in self._delay_grids.items():
            phase_dir = cache_dir / phase.replace(":", "_")
            phase_dir.mkdir(exist_ok=True)
            for nsl_str, delays_3d in nsl_delays.items():
                np.save(
                    phase_dir / f"{nsl_str.replace('.', '_')}.npy",
                    delays_3d,
                )

        # Save metadata
        meta = {
            "cache_key": cache_key,
            "grid_shape": [len(a) for a in self._grid_axes],
            "n_grid_nodes": int(np.prod([len(a) for a in self._grid_axes])),
            "octree_level": self.resolution_octree_level,
            "interpolation_method": self.delay_interpolation_method,
            "phases": phases,
            "stations": station_nsls,
            "device": str(DEVICE),
        }
        (cache_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        logger.info("saved SSST cache (key=%s) to %s", cache_key, cache_dir)

    def _load_cache(
        self,
        cache_dir: Path,
        phases: list[str],
        station_nsls: list[str],
    ) -> bool:
        """Try to load SSST from cached files. Returns True on success."""
        cache_file = cache_dir / "cache_key.txt"
        if not cache_file.exists():
            return False

        cached_key = cache_file.read_text().strip()
        current_key = self._compute_cache_key()
        if cached_key != current_key:
            logger.info("SSST cache outdated (key mismatch), will recompute")
            return False

        axes_file = cache_dir / "grid_axes.npz"
        if not axes_file.exists():
            return False

        # Load grid axes
        axes_data = np.load(axes_file)
        self._grid_axes = (axes_data["east"], axes_data["north"], axes_data["depth"])

        # Load 3D delay grids
        self._delay_grids = {}
        for phase in phases:
            phase_dir = cache_dir / phase.replace(":", "_")
            if not phase_dir.is_dir():
                continue
            self._delay_grids[phase] = {}
            for nsl_str in station_nsls:
                npy_file = phase_dir / f"{nsl_str.replace('.', '_')}.npy"
                if npy_file.exists():
                    self._delay_grids[phase][nsl_str] = np.load(npy_file)

        if not self._delay_grids:
            return False

        n_nodes = int(np.prod([len(a) for a in self._grid_axes]))
        logger.info("loaded SSST cache: %d nodes, %d phases", n_nodes, len(phases))
        return True

    # ── Delay Lookup ──────────────────────────────────────────────────

    def get_delay(
        self,
        station_nsl: NSL,
        phase: PhaseDescription,
        node: Node | None = None,
    ) -> float:
        if node is None or not self._interpolators:
            return 0.0

        nsl_str = station_nsl.pretty
        phase_interps = self._interpolators.get(phase)
        if not phase_interps:
            return 0.0
        interp = phase_interps.get(nsl_str)
        if not interp:
            return 0.0

        val = float(interp([[node.east, node.north, node.depth]])[0])
        return val if np.isfinite(val) else 0.0

    async def get_delays(
        self,
        station_nsls: Sequence[NSL],
        phase: PhaseDescription,
        nodes: Sequence[Node],
    ) -> np.ndarray:
        n_nodes = len(nodes)
        n_stations = len(station_nsls)

        if not self._interpolators or phase not in self._interpolators:
            return np.zeros((n_nodes, n_stations), dtype=np.float32)

        node_coords = np.array(
            [(n.east, n.north, n.depth) for n in nodes], dtype=np.float32,
        )

        delays = np.zeros((n_nodes, n_stations), dtype=np.float32)
        phase_interps = self._interpolators[phase]

        for idx, nsl in enumerate(station_nsls):
            interp = phase_interps.get(nsl.pretty)
            if interp is None:
                continue
            col = interp(node_coords)
            col = np.asarray(col, dtype=np.float32)
            col[~np.isfinite(col)] = 0.0
            delays[:, idx] = col

        return delays

    # ── Prepare ───────────────────────────────────────────────────────

    async def prepare(
        self,
        stations: StationInventory,
        octree: Octree,
        phases: Iterable[PhaseDescription],
        rundir: Path,
    ) -> None:
        logger.info("preparing SourceSpecificStationCorrections (SSST)")
        phases_list = list(phases)
        station_nsls = [sta.nsl.pretty for sta in stations]
        cache_dir = rundir / "ssst_corrections"

        # ── Try cache ──
        if cache_dir.exists() and self._load_cache(
            cache_dir, phases_list, station_nsls,
        ):
            self._interpolators = self._build_interpolators(
                self._grid_axes, self._delay_grids,
            )
            logger.info("SSST ready from cache — GPU computation skipped")
            return

        # ── Compute from scratch ──
        logger.info("computing SSST from scratch (no valid cache)")

        # Collect delay records
        all_records: list[DelayRecord] = []
        for rd in self.import_rundirs:
            rd_path = Path(rd).resolve()
            if not rd_path.is_dir():
                logger.warning("rundir not found: %s", rd_path)
                continue
            records = await asyncio.to_thread(
                extract_delays_from_rundir,
                rd_path,
                self.min_distance_border,
                self.min_num_picks,
            )
            all_records.extend(records)

        if not all_records:
            logger.warning("no delay records, SSST corrections will be zero")
            return

        # Build SSST grid at specified octree level
        ssst_octree = octree.model_copy(deep=True)
        ssst_octree.model_post_init(None)
        ssst_octree.set_level(self.resolution_octree_level)

        grid_coords = np.array(
            [(n.east, n.north, n.depth) for n in ssst_octree],
            dtype=np.float32,
        )
        logger.info(
            "SSST grid: %d nodes at octree level %d",
            grid_coords.shape[0],
            self.resolution_octree_level,
        )

        # GPU computation
        flat_delays = await asyncio.to_thread(
            self._compute_ssst_grid_gpu,
            all_records,
            grid_coords,
            phases_list,
            station_nsls,
        )

        # Convert flat arrays → 3D regular grid
        self._grid_axes, self._delay_grids = self._flat_to_3d(
            grid_coords, flat_delays,
        )

        # Build interpolators (instant with RegularGridInterpolator)
        self._interpolators = self._build_interpolators(
            self._grid_axes, self._delay_grids,
        )

        # Save cache
        self._save_cache(cache_dir, phases_list, station_nsls)

        # Plot
        await asyncio.to_thread(
            _plot_ssst_3d_volume,
            grid_coords,
            flat_delays,
            cache_dir,
        )

