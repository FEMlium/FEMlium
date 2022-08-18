# Copyright (C) 2021-2022 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT
"""Utils module for tutorial 01."""

try:
    import dolfin  # noqa: F401
except ImportError:
    pass
else:
    from .gmsh_to_fenics import gmsh_to_fenics  # noqa: F401, I2041, I2045
