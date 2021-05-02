# Copyright (C) 2021 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT

import branca


def ColorbarWrapper(colors, values, caption):
    """
    This class contains a wrapper to a branca.colormap.LinearColormap.

    Parameters
    ----------
    colors: 1d numpy array
        Vector containing the colors.
    colors: 1d numpy array
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
