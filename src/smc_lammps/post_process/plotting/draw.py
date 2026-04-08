# Copyright (c) 2025-2026 Lucas Dooms

from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.typing import ColorType

from smc_lammps.post_process.util import (
    get_post_processing_parameters,
    get_scaled_cum_runtimes,
    scale_indices,
)
from smc_lammps.reader.lammps_data import ID_TYPE


def get_runtime_lines(
    ax: Axes,
    path: Path,
    indices_array: Sequence[list[ID_TYPE]],
    tscale: float = 1.0,
    iscale: float = 1.0,
    cut_at_axes: bool = True,
) -> list[LineCollection]:
    scaled_array = list(map(partial(scale_indices, iscale=iscale), indices_array))

    ppp = get_post_processing_parameters(path)

    color_mapping = {"APO": "blue", "ATP": "red", "ADP": "green"}

    ymin, ymax = (
        min(min(val) for val in scaled_array),
        max(max(val) for val in scaled_array),
    )

    if cut_at_axes:
        ymin, ymax = ax.get_ylim()
    else:
        # extend to edges of graph
        diff = ymax - ymin
        ymin -= 0.2 * diff
        ymax += 0.2 * diff

    lines = []
    for smc_phase, start_times in get_scaled_cum_runtimes(ppp["runtimes"], tscale=tscale).items():
        if smc_phase == "all":
            continue

        segments = [np.array([(t, ymin), (t, ymax)]) for t in start_times]

        lc = LineCollection(
            segments,
            color=color_mapping[smc_phase],
            label=smc_phase,
        )

        # draw in background
        lc.set_zorder(-10.0)
        lc.set_linestyle("--")

        # set clipping so that svg files work properly
        lc.set_clip_on(True)
        lc.set_clip_box(ax.bbox)

        lines.append(lc)

    return lines


def fill_between_runtime_lines(ax: Axes, path: Path, tscale: float = 1.0):
    ppp = get_post_processing_parameters(path)
    q = get_scaled_cum_runtimes(ppp["runtimes"], tscale=tscale)["all"]

    segments = zip(q, q[1:])

    xlims = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    try:
        for i, (t, u) in enumerate(segments):
            ax.fill_between(
                [t, u],
                [ymin, ymin],
                [ymax, ymax],
                color=["lightblue", "lightpink", "yellow"][i % 3],
                alpha=0.2,
                zorder=-20.0,
                clip_on=True,
            )
    finally:
        ax.set_xlim(xlims)
        ax.set_ylim(ymin, ymax)


def get_colored_axis_segments(
    colors: Sequence[ColorType],
    starts: Sequence[float],
    ends: Sequence[float],
    line_width=3.0,
    zorder=0.5,
    y_offset=5.0,
) -> LineCollection:
    if not (len(colors) == len(starts) == len(ends)):
        raise ValueError(
            f"These arrays must have the same size: colors ({len(colors)}), starts ({len(starts)}), ends ({len(ends)})"
        )

    segments: list[list[tuple[float, float]]] = []
    for start, end in zip(starts, ends):
        segments.append([(start, y_offset), (end, y_offset)])

    lc = LineCollection(
        segments,
        colors=colors,
        linewidths=line_width,
        capstyle="butt",
        zorder=zorder,
        clip_on=True,
    )

    # Ensure the x-axis spine is visible and ticks draw over the segments
    # ax.spines["bottom"].set_visible(True)

    return lc
