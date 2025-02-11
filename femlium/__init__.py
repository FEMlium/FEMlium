# Copyright (C) 2021-2025 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT
"""FEMlium main module."""

from femlium.base_mesh_plotter import BaseMeshPlotter
from femlium.base_plotter import BasePlotter
from femlium.base_solution_plotter import BaseSolutionPlotter
from femlium.domain_plotter import DomainPlotter
from femlium.numpy_plotter import NumpyPlotter

try:
    import meshio
except ImportError:  # pragma: no cover
    pass
else:
    from femlium.meshio_plotter import MeshioPlotter

try:
    import dolfinx
except ImportError:
    pass
else:
    from femlium.dolfinx_plotter import DolfinxPlotter

try:
    import firedrake
except ImportError:
    pass
else:
    from femlium.firedrake_plotter import FiredrakePlotter
