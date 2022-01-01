# Copyright (C) 2021-2022 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT

class TransformerWrapper(object):
    """
    This class contains a wrapper to a pyproj.Transformer object, or None.

    Parameters
    ----------
    transformer : pyproj.Transformer, optional
        Defines an optional transformation between coordinate reference systems (CRS) if
        the input data use a different CRS than the output plot.
        If not provided, the identity map is used.

    Attributes
    ----------
    transformer : pyproj.Transformer, or None
        The first input parameter.
    """

    def __init__(self, transformer):
        self.transformer = transformer

    def __call__(self, *args):
        """
        Apply the transformation. If no transformer has been provided this method implements the identity map.

        Parameters
        ----------
        *args : tuple
            Input coordinates to be transformed.

        Returns
        -------
        tuple
            Output coordinates after transformation.
        """

        if self.transformer is None:
            return args
        else:
            return self.transformer.transform(*args)
