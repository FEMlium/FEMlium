# Copyright (C) 2021 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT

import numpy as np
from femlium.utils import TransformerWrapper


class BasePlotter(object):
    """
    This class contains the interface of a geographic plotter.

    Parameters
    ----------
    transformer : pyproj.Transformer, optional
        Defines an optional transformation between coordinate reference systems (CRS) if
        the input data use a different CRS than the output plot.
        If not provided, the identity map is used.

    Attributes
    ----------
    transformer : femlium.TransformerWrapper
        Wrapper to the transformer object provided as first input parameter.
    """

    def __init__(self, transformer):
        self.transformer = TransformerWrapper(transformer)

    @staticmethod
    def _process_optional_argument_on_markers(argument, default, unique_markers):
        expected_type = type(default)
        assert isinstance(argument, (expected_type, dict)) or argument is None
        if isinstance(argument, dict):
            assert all(isinstance(value, expected_type) for (_, value) in argument.items())

        if isinstance(default, str):
            dtype = np.dtype(object)  # otherwise np.dtype(str) only allows a single character
        else:
            dtype = np.dtype(expected_type)

        assert np.min(unique_markers) >= 0
        output = np.full(np.max(unique_markers) + 1, default, dtype=dtype)
        for m in unique_markers:
            if argument is None:
                pass
            elif isinstance(argument, expected_type):
                output[m] = argument
            elif isinstance(argument, dict):
                if m in argument:
                    output[m] = argument[m]
            else:
                raise ValueError("Invalid argument provided")
        return output
