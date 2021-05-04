# Copyright (C) 2021 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT

from femlium.base_plotter import BasePlotter
from femlium.base_mesh_plotter import BaseMeshPlotter
from femlium.base_solution_plotter import BaseSolutionPlotter
from femlium.domain_plotter import DomainPlotter
from femlium.numpy_plotter import NumpyPlotter

__all__ = [
    "BasePlotter",
    "BaseMeshPlotter",
    "BaseSolutionPlotter",
    "DomainPlotter",
    "NumpyPlotter"
]

try:
    import meshio  # noqa: F401
except ImportError:
    pass
else:
    from femlium.meshio_plotter import MeshioPlotter
    __all__ += ["MeshioPlotter"]

try:
    import dolfin  # noqa: F401
except ImportError:
    pass
else:
    from femlium.dolfin_plotter import DolfinPlotter
    __all__ += ["DolfinPlotter"]

try:
    import dolfinx  # noqa: F401
except ImportError:
    pass
else:
    from femlium.dolfinx_plotter import DolfinxPlotter
    __all__ += ["DolfinxPlotter"]

try:
    import firedrake  # noqa: F401
except ImportError:
    pass
else:
    from femlium.firedrake_plotter import FiredrakePlotter
    __all__ += ["FiredrakePlotter"]
