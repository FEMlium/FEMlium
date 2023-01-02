# Copyright (C) 2021-2023 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT
"""Wrap a pyproj.Transformer object."""

import typing

import numpy as np
import numpy.typing
import pyproj


class TransformerWrapper(object):
    """
    Wrap a pyproj.Transformer object, or None.

    Parameters
    ----------
    transformer
        Defines an optional transformation between coordinate reference systems (CRS) if
        the input data use a different CRS than the output plot.
        If not provided, the identity map is used.

    Attributes
    ----------
    transformer
        The first input parameter.
    """

    def __init__(self, transformer: typing.Optional[pyproj.Transformer]) -> None:
        self.transformer = transformer

    def __call__(self, *args: np.float64) -> np.typing.NDArray[np.float64]:
        """
        Apply the transformation. If no transformer has been provided this method implements the identity map.

        Parameters
        ----------
        *args
            Input coordinates to be transformed.

        Returns
        -------
        :
            Output coordinates after transformation.
        """
        if self.transformer is None:
            return args
        else:
            return self.transformer.transform(*args)
