# Copyright (C) 2021 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT

import os
import numpy as np
import meshio
import dolfin


def gmsh_to_fenics(msh_path):
    assert msh_path.endswith(".msh")
    base_path = msh_path[:-4]

    # Read back in the mesh with meshio
    meshio_mesh = meshio.read(msh_path)

    # Save volume mesh in xdmf format
    mesh_xdmf_path = base_path + "_mesh.xdmf"
    if os.path.exists(mesh_xdmf_path):
        os.remove(mesh_xdmf_path)
    if os.path.exists(mesh_xdmf_path.replace(".xdmf", ".h5")):
        os.remove(mesh_xdmf_path.replace(".xdmf", ".h5"))
    points = meshio_mesh.points[:, :2]
    cells = meshio_mesh.cells_dict["triangle"]
    if ("gmsh:physical" in meshio_mesh.cell_data_dict
            and "triangle" in meshio_mesh.cell_data_dict["gmsh:physical"]):
        subdomains_data = meshio_mesh.cell_data_dict["gmsh:physical"]["triangle"]
    else:
        subdomains_data = np.zeros_like(cells)
    meshio.write(
        mesh_xdmf_path,
        meshio.Mesh(
            points=points,
            cells={"triangle": cells},
            cell_data={"subdomains": [subdomains_data]}
        )
    )

    # Save boundary mesh in xdmf format
    boundaries_xdmf_path = base_path + "_boundaries.xdmf"
    if os.path.exists(boundaries_xdmf_path):
        os.remove(boundaries_xdmf_path)
    if os.path.exists(boundaries_xdmf_path.replace(".xdmf", ".h5")):
        os.remove(boundaries_xdmf_path.replace(".xdmf", ".h5"))
    facets = meshio_mesh.cells_dict["line"]
    if ("gmsh:physical" in meshio_mesh.cell_data_dict
            and "line" in meshio_mesh.cell_data_dict["gmsh:physical"]):
        boundaries_data = meshio_mesh.cell_data_dict["gmsh:physical"]["line"]
    else:
        boundaries_data = np.zeros_like(facets)
    meshio.write(
        boundaries_xdmf_path,
        meshio.Mesh(
            points=points,
            cells={"line": facets},
            cell_data={"boundaries": [boundaries_data]}
        )
    )

    # Read back in the mesh with dolfin
    mesh = dolfin.Mesh()
    with dolfin.XDMFFile(mesh_xdmf_path) as infile:
        infile.read(mesh)

    # Read back in subdomains with dolfin
    subdomains_mvc = dolfin.MeshValueCollection("size_t", mesh, mesh.topology().dim())
    with dolfin.XDMFFile(mesh_xdmf_path) as infile:
        infile.read(subdomains_mvc, "subdomains")
    subdomains = dolfin.cpp.mesh.MeshFunctionSizet(mesh, subdomains_mvc)

    # Clean up mesh file
    os.remove(mesh_xdmf_path)
    os.remove(mesh_xdmf_path.replace(".xdmf", ".h5"))

    # Read back in boundaries with dolfin, and explicitly set to 0 any facet
    # which had not been marked by gmsh
    boundaries_mvc = dolfin.MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
    with dolfin.XDMFFile(boundaries_xdmf_path) as infile:
        infile.read(boundaries_mvc, "boundaries")
    boundaries_mvc_dict = boundaries_mvc.values()
    for c in dolfin.cells(mesh):
        for f, _ in enumerate(dolfin.facets(c)):
            if (c.index(), f) not in boundaries_mvc_dict:
                boundaries_mvc.set_value(c.index(), f, 0)
    boundaries = dolfin.cpp.mesh.MeshFunctionSizet(mesh, boundaries_mvc)

    # Clean up boundary mesh file
    os.remove(boundaries_xdmf_path)
    os.remove(boundaries_xdmf_path.replace(".xdmf", ".h5"))
    return mesh, subdomains, boundaries
