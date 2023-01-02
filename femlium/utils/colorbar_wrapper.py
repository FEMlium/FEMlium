# Copyright (C) 2021-2023 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT
"""Wrap a branca.colormap.LinearColormap."""

import branca
import numpy as np
import numpy.typing


def ColorbarWrapper(
    colors: np.typing.NDArray[str], values: np.typing.NDArray[np.float64], caption: str
) -> branca.colormap.LinearColormap:
    """
    Wrap a branca.colormap.LinearColormap.

    Parameters
    ----------
    colors
        Vector containing the colors.
    values
        Vector containing the values associated to each color.
    caption
        Colorbar caption.

    Returns
    -------
    :
        Colorbar to be attached to the plot.
    """
    return branca.colormap.LinearColormap(
        colors=colors, index=values, vmin=values[0], vmax=values[-1], caption=caption)
