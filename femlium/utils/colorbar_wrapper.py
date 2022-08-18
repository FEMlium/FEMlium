# Copyright (C) 2021-2022 by the FEMlium authors
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
    colors: 1d numpy array
        Vector containing the colors.
    values: 1d numpy array
        Vector containing the values associated to each color.
    caption: str
        Colorbar caption.

    Returns
    -------
    branca.colormap.LinearColormap
        Colorbar to be attached to the plot.
    """
    return branca.colormap.LinearColormap(
        colors=colors, index=values, vmin=values[0], vmax=values[-1], caption=caption)
