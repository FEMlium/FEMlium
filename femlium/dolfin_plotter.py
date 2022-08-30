# Copyright (C) 2021-2022 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT
"""Interface of a geographic plotter for dolfin meshes and solutions."""

import typing

import dolfin
import folium
import numpy as np

from femlium.base_mesh_plotter import BaseMeshPlotter
from femlium.base_solution_plotter import BaseSolutionPlotter


class DolfinPlotter(BaseMeshPlotter, BaseSolutionPlotter):
    """Interface of a geographic plotter for dolfin meshes and solutions."""

    def add_mesh_to(
        self, geo_map: folium.Map, mesh: dolfin.Mesh,
        cell_mesh_function: typing.Optional[dolfin.MeshFunction] = None,
        face_mesh_function: typing.Optional[dolfin.MeshFunction] = None,
        cell_colors: typing.Optional[typing.Union[str, typing.Dict[int, str]]] = None,
        face_colors: typing.Optional[typing.Union[str, typing.Dict[int, str]]] = None,
        face_weights: typing.Optional[typing.Union[int, typing.Dict[int, int]]] = None
    ) -> None:
        """
        Add a triangular mesh stored in a dolfin.Mesh to a folium map.

        Parameters
        ----------
        geo_map
            Map to which the mesh plot should be added.
        mesh
            A dolfin mesh.
        cell_mesh_function
            A dolfin mesh function of topological dimension 2 that stores cell markers.
            If not provided, the marker will be set to 0 everywhere.
        face_mesh_function
            A dolfin mesh function of topological dimension 1 that stores face markers.
            If not provided, the marker will be set to 0 everywhere.
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
        vertices = mesh.coordinates()
        cells = mesh.cells()

        if cell_mesh_function is not None:
            cell_markers = cell_mesh_function.array()
        else:
            cell_markers = None

        if face_mesh_function is not None:
            mesh.init(2, 1)
            cell_to_faces_connectivity = mesh.topology()(2, 1)

            # The local face to vertex connectivity in FIAT is
            # 0: (1, 2), 1: (0, 2), 2: (0, 1)
            # while in FEMlium we assume
            # 0: (0, 1), 1: (1, 2), 2: (0, 2)
            fiat_to_femlium = {0: 1, 1: 2, 2: 0}

            face_markers = np.zeros(cells.shape, dtype=np.int64)
            for c in range(cells.shape[0]):
                for (f, global_face_number) in enumerate(cell_to_faces_connectivity(c)):
                    face_markers[c, fiat_to_femlium[f]] = face_mesh_function[global_face_number]
        else:
            face_markers = None

        return BaseMeshPlotter.add_mesh_to(
            self, geo_map, vertices, cells, cell_markers, face_markers, cell_colors, face_colors, face_weights)

    def add_scalar_field_to(
        self, geo_map: folium.Map, scalar_field: dolfin.Function, mode: typing.Optional[str] = None,
        levels: typing.Optional[typing.Union[int, typing.List[float]]] = None,
        cmap: typing.Optional[str] = None, name: typing.Optional[str] = None
    ) -> None:
        """
        Add a scalar field to a folium map.

        Parameters
        ----------
        geo_map
            Map to which the mesh plot should be added.
        scalar_field
            A dolfin Function representing the scalar field.
        mode
            Plot to be generated, either contourf or contour.
            If not provided, contourf is used.
        levels
            Values of the contour lines.
            If integer, it will determine the number of equispaced values to be used between
            the minimum and maximum entry of the scalar field.
            If list, it will determine the values to be used.
            If not provided, 10 levels are used by default.
        cmap
            matplotlib color map to be used.
            If not provided, the jet colormap is used.
        name
            Name of the field, to be used in the creation of the color bar.
            If not provided, the name "scalar field" will be used.
        """
        mesh = scalar_field.function_space().mesh()
        vertices = mesh.coordinates()
        cells = mesh.cells()
        scalar_field_values = scalar_field.compute_vertex_values(mesh)

        return BaseSolutionPlotter.add_scalar_field_to(
            self, geo_map, vertices, cells, scalar_field_values, mode, levels, cmap, name)

    def add_vector_field_to(
        self, geo_map: folium.Map, vector_field: dolfin.Function, mode: typing.Optional[str] = None,
        levels: typing.Optional[typing.Union[int, typing.List[float]]] = None, scale: typing.Optional[float] = None,
        cmap: typing.Optional[str] = None, name: typing.Optional[str] = None
    ) -> None:
        """
        Add a vector field to a folium map.

        Parameters
        ----------
        geo_map
            Map to which the mesh plot should be added.
        vector_field
            A dolfin Function representing the vector field.
        mode
            Plot to be generated, either contourf, contour or quiver.
            If not provided, contourf is used.
        levels
            In contourf or contour mode: values of the contour lines.
            In quiver mode: number of ticks to be added to the color bar.
            If integer, it will determine the number of equispaced values to be used between
            the minimum and maximum entry of the scalar field.
            If list, it will determine the values to be used.
            If not provided, 10 levels are used by default.
        scale
            This is only applicable for quiver mode: scaling to be applied before drawing arrows.
            If not provided, no scaling will be applied.
        cmap
            matplotlib color map to be used.
            If not provided, the jet colormap is used.
        name
            Name of the field, to be used in the creation of the color bar.
            If not provided, the name "vector field" will be used.
        """
        mesh = vector_field.function_space().mesh()
        vertices = mesh.coordinates()
        cells = mesh.cells()
        vector_field_values = vector_field.compute_vertex_values(mesh).reshape(2, -1).T

        return BaseSolutionPlotter.add_vector_field_to(
            self, geo_map, vertices, cells, vector_field_values, mode, levels, scale, cmap, name)
