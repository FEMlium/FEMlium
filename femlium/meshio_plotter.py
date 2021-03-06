# Copyright (C) 2021-2022 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT

import numpy as np
from femlium.base_mesh_plotter import BaseMeshPlotter


class MeshioPlotter(BaseMeshPlotter):
    """
    This class contains the interface of a geographic plotter for mesh-related plots for meshes read in with meshio.
    """

    def add_mesh_to(self, geo_map, mesh, unmarked_face_marker=None,
                    cell_colors=None, face_colors=None, face_weights=None):
        """
        Add a triangular mesh imported from meshio to a folium map.

        Parameters
        ----------
        geo_map : folium.Map
            Map to which the mesh plot should be added.
        mesh: meshio.Mesh
            A meshio mesh.
        unmarked_face_marker: int, optional
            Marker to be assigned to any unmarked face (e.g., internal faces in a typical scenario
            in which only boundary faces are marked).
            If not provided, it is set to 0.
        cell_colors: str or dict of str, optional
            If a dictionary is provided, it should contain key: value pairs defining the mapping
            marker: color for cells.
            If a string is provided instead of a dictionary, the same color will be used for all
            cell markers.
            If not provided, the cells will not be colored.
        face_colors: str or dict of str, optional
            If a dictionary is provided, it should contain key: value pairs defining the mapping
            marker: color for faces.
            If a string is provided instead of a dictionary, the same color will be used for all
            face markers.
            If not provided, a default black color will be used for faces.
        face_weights: int or dict of int, optional
            Line weight of each face. Input should be provided following a similar convention for
            the face_colors argument.
            If not provided, a unit weight will be used.
        """

        if unmarked_face_marker is None:
            unmarked_face_marker = 0

        vertices = mesh.points[:, :2]
        cells = mesh.cells_dict["triangle"]

        if "gmsh:physical" in mesh.cell_data_dict and "triangle" in mesh.cell_data_dict["gmsh:physical"]:
            cell_markers = mesh.cell_data_dict["gmsh:physical"]["triangle"]
        else:
            cell_markers = None

        if "gmsh:physical" in mesh.cell_data_dict and "line" in mesh.cell_data_dict["gmsh:physical"]:
            faces = mesh.cells_dict["line"]
            face_data = mesh.cell_data_dict["gmsh:physical"]["line"]
            face_data_dict = dict()
            for f in range(faces.shape[0]):
                assert len(faces[f, :]) == 2
                if faces[f, 0] < faces[f, 1]:
                    key = (faces[f, 0], faces[f, 1])
                else:
                    key = (faces[f, 1], faces[f, 0])
                face_data_dict[key] = face_data[f]
            face_markers = np.zeros(cells.shape, dtype=np.dtype(int))
            for c in range(cells.shape[0]):
                for (f, pair) in enumerate(((0, 1), (1, 2), (0, 2))):
                    if cells[c, pair[0]] < cells[c, pair[1]]:
                        key = (cells[c, pair[0]], cells[c, pair[1]])
                    else:
                        key = (cells[c, pair[1]], cells[c, pair[0]])
                    face_markers[c, f] = face_data_dict.get(key, unmarked_face_marker)
        else:
            face_markers = None

        return BaseMeshPlotter.add_mesh_to(
            self, geo_map, vertices, cells, cell_markers, face_markers, cell_colors, face_colors, face_weights)
