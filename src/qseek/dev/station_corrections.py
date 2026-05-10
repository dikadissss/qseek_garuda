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
    """Plot SSST delay grid as 3D volume visualisation.

    Creates 3D scatter plots showing spatial distribution of delays
    with blue-red diverging colormap, similar to a volume rendering.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        logger.warning("matplotlib not available, skipping SSST plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    east = grid_coords[:, 0]
    north = grid_coords[:, 1]
    depth = grid_coords[:, 2]

    for phase, nsl_delays in grid_delays.items():
        for nsl_str, delays in nsl_delays.items():
            if np.all(delays == 0):
                continue

            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection="3d")

            # Use diverging colormap centered at 0
            vmax = max(abs(np.nanmin(delays)), abs(np.nanmax(delays)))
            if vmax < 1e-6:
                continue
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

            sc = ax.scatter(
                east, north, -depth,  # negative depth so surface is up
                c=delays,
                cmap="RdBu_r",
                norm=norm,
                s=12,
                alpha=0.6,
                edgecolors="none",
            )

            ax.set_xlabel("Easting (m)", fontsize=10)
            ax.set_ylabel("Northing (m)", fontsize=10)
            ax.set_zlabel("Depth (m)", fontsize=10)

            phase_short = phase.split(":")[-1] if ":" in phase else phase
            ax.set_title(
                f"SSST Corrections — Phase: {phase_short}, Station: {nsl_str}",
                fontsize=12, pad=20,
            )

            cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
            cbar.set_label("Delay (s)", fontsize=10)

            ax.tick_params(labelsize=8)
            ax.view_init(elev=25, azim=-60)

            safe_nsl = nsl_str.replace(".", "_")
            safe_phase = phase.replace(":", "_")
            filename = output_dir / f"ssst_3d_{safe_nsl}_{safe_phase}.png"
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("saved SSST 3D plot: %s", filename)

        # Also create 2D slice plots (NE, ED, ND) for overview
        _plot_ssst_slices(east, north, depth, nsl_delays, phase, output_dir)


def _plot_ssst_slices(
    east: np.ndarray,
    north: np.ndarray,
    depth: np.ndarray,
    nsl_delays: dict[str, np.ndarray],
    phase: str,
    output_dir: Path,
) -> None:
    """Plot 2D slice views (top, front, side) of SSST grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    # Average delays across all stations for overview
    all_delays = [d for d in nsl_delays.values() if not np.all(d == 0)]
    if not all_delays:
        return
    avg_delays = np.mean(all_delays, axis=0)

    vmax = max(abs(np.nanmin(avg_delays)), abs(np.nanmax(avg_delays)))
    if vmax < 1e-6:
        return
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    phase_short = phase.split(":")[-1] if ":" in phase else phase
    safe_phase = phase.replace(":", "_")

    slices = [
        ("NE", east, north, "Easting (m)", "Northing (m)", "top_view"),
        ("ED", east, depth, "Easting (m)", "Depth (m)", "front_view"),
        ("ND", north, depth, "Northing (m)", "Depth (m)", "side_view"),
    ]

    for name, x, y, xlabel, ylabel, suffix in slices:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Group by unique (x, y) and take max along the third axis
        sc = ax.scatter(
            x, -y if "Depth" in ylabel else y,
            c=avg_delays,
            cmap="RdBu_r",
            norm=norm,
            s=20,
            alpha=0.7,
            edgecolors="none",
        )

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(
            f"-{ylabel}" if "Depth" in ylabel else ylabel, fontsize=11,
        )
        ax.set_title(
            f"SSST Average Delay — {name} view, Phase: {phase_short}",
            fontsize=12,
        )

        cbar = fig.colorbar(sc, ax=ax)
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
    # _grid_delays[phase][nsl_str] -> np.ndarray of shape (n_grid_nodes,)
    _grid_delays: dict[str, dict[str, np.ndarray]] = PrivateAttr(default_factory=dict)
    # Grid node coordinates (n_grid_nodes, 3) for interpolation
    _grid_coords: np.ndarray | None = PrivateAttr(default=None)
    _grid_nodes: list[Node] = PrivateAttr(default_factory=list)
    _interpolators: dict[str, dict[str, object]] = PrivateAttr(default_factory=dict)

    @property
    def n_stations(self) -> int:
        if not self._grid_delays:
            return 0
        first_phase = next(iter(self._grid_delays))
        return len(self._grid_delays[first_phase])

    def _compute_ssst_grid_gpu(
        self,
        records: list[DelayRecord],
        grid_coords: np.ndarray,
        phases: list[str],
        station_nsls: list[str],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Compute SSST grid using GPU CUDA acceleration."""
        n_grid = grid_coords.shape[0]
        device = DEVICE
        logger.info("computing SSST grid on %s (%d grid nodes)", device, n_grid)

        # Group records by (phase, nsl)
        phase_nsl_records: dict[str, dict[str, list[DelayRecord]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for rec in records:
            phase_nsl_records[rec.phase][rec.nsl.pretty].append(rec)

        # Convert grid coords to tensor
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

                # Build event tensors on GPU
                event_coords = torch.tensor(
                    [[r.event_east, r.event_north, r.event_depth] for r in recs],
                    dtype=torch.float32,
                    device=device,
                )
                delays_t = torch.tensor(
                    [r.delay for r in recs], dtype=torch.float32, device=device
                )
                base_weights = torch.tensor(
                    [compute_weight(self.weighting, r.confidence, r.semblance)
                     for r in recs],
                    dtype=torch.float32,
                    device=device,
                )

                # Compute distances: (n_grid, n_events)
                distances = torch.cdist(grid_tensor, event_coords)

                # Spatial weights: 1 / distance^exponent (avoid div by zero)
                eps = torch.tensor(1.0, device=device)
                spatial_weights = 1.0 / torch.pow(
                    torch.maximum(distances, eps), self.spatial_weighting_exponent
                )

                # Combined weights: (n_grid, n_events)
                combined_weights = spatial_weights * base_weights.unsqueeze(0)

                # For each grid node, compute weighted average delay
                # Adaptive radius: expand until min_confidence is met
                confidences = torch.tensor(
                    [r.confidence for r in recs], dtype=torch.float32, device=device
                )

                node_delays = torch.zeros(n_grid, dtype=torch.float32, device=device)

                # Batch computation: weighted average
                weight_sums = combined_weights.sum(dim=1)
                weighted_delays = (combined_weights * delays_t.unsqueeze(0)).sum(dim=1)

                # Check min_confidence per node
                cumulative_conf = (spatial_weights * confidences.unsqueeze(0)).sum(dim=1)

                # Where confidence is sufficient, use weighted average
                valid = (weight_sums > 0) & (cumulative_conf >= self.min_confidence)
                node_delays[valid] = weighted_delays[valid] / weight_sums[valid]

                # For nodes with insufficient confidence, use increasing radius
                insufficient = ~valid & (weight_sums > 0)
                if insufficient.any():
                    # Fallback: use all events with uniform spatial weight
                    total_w = base_weights.sum()
                    if total_w > 0:
                        global_delay = (delays_t * base_weights).sum() / total_w
                        node_delays[insufficient] = global_delay

                result[phase][nsl_str] = node_delays.cpu().numpy()

        return result

    def _build_interpolators(
        self,
        grid_coords: np.ndarray,
        grid_delays: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, dict[str, object]]:
        """Build scipy interpolators for delay lookup."""
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        interpolators: dict[str, dict[str, object]] = {}
        method = self.delay_interpolation_method

        for phase, nsl_delays in grid_delays.items():
            interpolators[phase] = {}
            for nsl_str, delays in nsl_delays.items():
                if method == "nearest":
                    interp = NearestNDInterpolator(grid_coords, delays)
                elif method == "linear":
                    try:
                        interp = LinearNDInterpolator(grid_coords, delays)
                    except Exception:
                        logger.warning(
                            "linear interpolation failed for %s/%s, "
                            "falling back to nearest",
                            phase, nsl_str,
                        )
                        interp = NearestNDInterpolator(grid_coords, delays)
                elif method == "cubic":
                    try:
                        from scipy.interpolate import CloughTocher2DInterpolator
                        # Cubic only works in 2D, fall back to linear for 3D
                        interp = LinearNDInterpolator(grid_coords, delays)
                    except Exception:
                        interp = NearestNDInterpolator(grid_coords, delays)
                else:
                    interp = NearestNDInterpolator(grid_coords, delays)
                interpolators[phase][nsl_str] = interp

        return interpolators

    def get_delay(
        self,
        station_nsl: NSL,
        phase: PhaseDescription,
        node: Node | None = None,
    ) -> float:
        if node is None or not self._interpolators:
            return 0.0

        nsl_str = station_nsl.pretty
        if phase not in self._interpolators:
            return 0.0
        if nsl_str not in self._interpolators[phase]:
            return 0.0

        interp = self._interpolators[phase][nsl_str]
        coords = np.array([[node.east, node.north, node.depth]])
        result = interp(coords)
        val = float(result[0])
        return val if np.isfinite(val) else 0.0

    async def get_delays(
        self,
        station_nsls: Sequence[NSL],
        phase: PhaseDescription,
        nodes: Sequence[Node],
    ) -> np.ndarray:
        if not self._interpolators or phase not in self._interpolators:
            return np.zeros(
                (len(nodes), len(station_nsls)), dtype=np.float32
            )

        # Build node coordinates
        node_coords = np.array(
            [(n.east, n.north, n.depth) for n in nodes], dtype=np.float32
        )

        n_nodes = len(nodes)
        n_stations = len(station_nsls)
        delays = np.zeros((n_nodes, n_stations), dtype=np.float32)

        phase_interps = self._interpolators.get(phase, {})
        for sta_idx, nsl in enumerate(station_nsls):
            nsl_str = nsl.pretty
            interp = phase_interps.get(nsl_str)
            if interp is None:
                continue
            result = await asyncio.to_thread(interp, node_coords)
            col = np.asarray(result, dtype=np.float32)
            col[~np.isfinite(col)] = 0.0
            delays[:, sta_idx] = col

        return delays

    def _compute_cache_key(self) -> str:
        """Compute a hash key based on config parameters and import rundirs.

        The cache is invalidated when any config parameter or the source
        data (detected by rundir modification times and file sizes) changes.
        """
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

        # Include file modification times from import rundirs for invalidation
        for rd in sorted(self.import_rundirs, key=str):
            rd_path = Path(rd).resolve()
            detections_file = rd_path / "detections.json"
            receivers_file = rd_path / "detections_receivers.json"
            for f in (detections_file, receivers_file):
                if f.exists():
                    stat = f.stat()
                    key_parts.append(f"{f}:{stat.st_mtime}:{stat.st_size}")

        key_str = "|".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _load_from_cache(
        self,
        corrections_dir: Path,
        phases: list[str],
        station_nsls: list[str],
    ) -> bool:
        """Try to load SSST grid delays from cached .npy files.

        Returns True if cache was loaded successfully, False otherwise.
        """
        cache_file = corrections_dir / "cache_key.txt"
        if not cache_file.exists():
            return False

        # Check cache key matches
        cached_key = cache_file.read_text().strip()
        current_key = self._compute_cache_key()
        if cached_key != current_key:
            logger.info(
                "SSST cache key mismatch (cached=%s, current=%s), recomputing",
                cached_key, current_key,
            )
            return False

        # Load grid coordinates
        coords_file = corrections_dir / "grid_coords.npy"
        if not coords_file.exists():
            return False
        self._grid_coords = np.load(coords_file)

        # Load grid delays from .npy files
        self._grid_delays = {}
        for phase in phases:
            phase_dir = corrections_dir / phase.replace(":", "_")
            if not phase_dir.is_dir():
                continue
            self._grid_delays[phase] = {}
            for nsl_str in station_nsls:
                npy_file = phase_dir / f"{nsl_str.replace('.', '_')}.npy"
                if npy_file.exists():
                    self._grid_delays[phase][nsl_str] = np.load(npy_file)
                else:
                    self._grid_delays[phase][nsl_str] = np.zeros(
                        self._grid_coords.shape[0], dtype=np.float32
                    )

        if not self._grid_delays:
            return False

        logger.info(
            "loaded SSST corrections from cache (%d phases, %d grid nodes)",
            len(self._grid_delays),
            self._grid_coords.shape[0],
        )
        return True

    def _save_to_cache(
        self,
        corrections_dir: Path,
        phases: list[str],
        station_nsls: list[str],
    ) -> None:
        """Save SSST grid delays to cache files."""
        corrections_dir.mkdir(exist_ok=True)

        # Save cache key
        cache_key = self._compute_cache_key()
        (corrections_dir / "cache_key.txt").write_text(cache_key)

        # Save grid coordinates
        np.save(corrections_dir / "grid_coords.npy", self._grid_coords)

        # Save metadata
        meta = {
            "cache_key": cache_key,
            "n_grid_nodes": len(self._grid_nodes),
            "octree_level": self.resolution_octree_level,
            "phases": phases,
            "stations": station_nsls,
            "device": str(DEVICE),
        }
        (corrections_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        # Save grid delays as numpy arrays
        for phase, nsl_delays in self._grid_delays.items():
            phase_dir = corrections_dir / phase.replace(":", "_")
            phase_dir.mkdir(exist_ok=True)
            for nsl_str, delays in nsl_delays.items():
                np.save(
                    phase_dir / f"{nsl_str.replace('.', '_')}.npy",
                    delays,
                )

        logger.info("saved SSST cache (key=%s) to %s", cache_key, corrections_dir)

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

        corrections_dir = rundir / "ssst_corrections"

        # === Try loading from cache first ===
        if corrections_dir.exists():
            cache_loaded = self._load_from_cache(
                corrections_dir, phases_list, station_nsls,
            )
            if cache_loaded:
                # Build octree grid nodes for coordinate reference
                ssst_octree = octree.model_copy(deep=True)
                ssst_octree.model_post_init(None)
                ssst_octree.set_level(self.resolution_octree_level)
                self._grid_nodes = list(ssst_octree)

                # Build interpolators from cached data
                self._interpolators = await asyncio.to_thread(
                    self._build_interpolators,
                    self._grid_coords,
                    self._grid_delays,
                )
                logger.info(
                    "SSST corrections loaded from cache — "
                    "skipped GPU computation"
                )
                return

        # === No valid cache, compute from scratch ===
        logger.info("no valid SSST cache found, computing from scratch")

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
        self._grid_nodes = list(ssst_octree)

        # Get grid coordinates
        self._grid_coords = np.array(
            [(n.east, n.north, n.depth) for n in self._grid_nodes],
            dtype=np.float32,
        )
        logger.info(
            "SSST grid: %d nodes at octree level %d",
            len(self._grid_nodes),
            self.resolution_octree_level,
        )

        # Compute SSST grid using GPU
        self._grid_delays = await asyncio.to_thread(
            self._compute_ssst_grid_gpu,
            all_records,
            self._grid_coords,
            phases_list,
            station_nsls,
        )

        # Build interpolators
        self._interpolators = await asyncio.to_thread(
            self._build_interpolators,
            self._grid_coords,
            self._grid_delays,
        )

        # Save to cache
        self._save_to_cache(corrections_dir, phases_list, station_nsls)

        # Plot SSST 3D volume
        await asyncio.to_thread(
            _plot_ssst_3d_volume,
            self._grid_coords,
            self._grid_delays,
            corrections_dir,
        )
