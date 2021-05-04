# Copyright (C) 2021 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT

__all__ = []

try:
    import dolfin  # noqa: F401
except ImportError:
    pass
else:
    from .gmsh_to_fenics import gmsh_to_fenics
    __all__ += ["gmsh_to_fenics"]

try:
    import dolfinx  # noqa: F401
except ImportError:
    pass
else:
    from .gmsh_to_fenicsx import gmsh_to_fenicsx
    __all__ += ["gmsh_to_fenicsx"]
