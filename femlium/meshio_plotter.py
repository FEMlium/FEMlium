# Copyright (C) 2021-2022 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT
"""Interface of a geographic plotter for mesh-related plots for meshes read in with meshio."""

import typing

import folium
import meshio
import numpy as np

from femlium.base_mesh_plotter import BaseMeshPlotter


class MeshioPlotter(BaseMeshPlotter):
    """Interface of a geographic plotter for mesh-related plots for meshes read in with meshio."""

    def add_mesh_to(
        self, geo_map: folium.Map, mesh: meshio.Mesh, unmarked_face_marker: typing.Optional[int] = None,
        cell_colors: typing.Optional[typing.Union[str, typing.Dict[int, str]]] = None,
        face_colors: typing.Optional[typing.Union[str, typing.Dict[int, str]]] = None,
        face_weights: typing.Optional[typing.Union[int, typing.Dict[int, int]]] = None
    ) -> None:
        """
        Add a triangular mesh imported from meshio to a folium map.

        Parameters
        ----------
        geo_map
            Map to which the mesh plot should be added.
        mesh
            A meshio mesh.
        unmarked_face_marker
            Marker to be assigned to any unmarked face (e.g., internal faces in a typical scenario
            in which only boundary faces are marked).
            If not provided, it is set to 0.
        cell_colors
            If a dictionary is provided, it should contain key: value pairs defining the mapping
            marker: color for cells.
            If a string is provided instead of a dictionary, the same color will be used for all
            cell markers.
            If not provided, the cells will not be colored.
        face_colors
            If a dictionary is provided, it should contain key: value pairs defining the mapping
            marker: color for faces.
            If a string is provided instead of a dictionary, the same color will be used for all
            face markers.
            If not provided, a default black color will be used for faces.
        face_weights
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
        else:  # pragma: no cover
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
            face_markers = np.zeros(cells.shape, dtype=np.int64)
            for c in range(cells.shape[0]):
                for (f, pair) in enumerate(((0, 1), (1, 2), (0, 2))):
                    if cells[c, pair[0]] < cells[c, pair[1]]:
                        key = (cells[c, pair[0]], cells[c, pair[1]])
                    else:
                        key = (cells[c, pair[1]], cells[c, pair[0]])
                    face_markers[c, f] = face_data_dict.get(key, unmarked_face_marker)
        else:  # pragma: no cover
            face_markers = None

        return BaseMeshPlotter.add_mesh_to(
            self, geo_map, vertices, cells, cell_markers, face_markers, cell_colors, face_colors, face_weights)
