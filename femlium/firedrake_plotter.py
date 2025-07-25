# Copyright (C) 2021-2025 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT
"""Interface of a geographic plotter for firedrake meshes and solutions."""

import typing

import firedrake
import folium
import numpy as np

from femlium.base_mesh_plotter import BaseMeshPlotter
from femlium.base_solution_plotter import BaseSolutionPlotter


class FiredrakePlotter(BaseMeshPlotter, BaseSolutionPlotter):
    """Interface of a geographic plotter for firedrake meshes and solutions."""

    def add_mesh_to(
        self, geo_map: folium.Map, mesh: firedrake.Mesh,
        unmarked_cell_marker: typing.Optional[int] = None, unmarked_face_marker: typing.Optional[int] = None,
        cell_colors: typing.Optional[typing.Union[str, dict[int, str]]] = None,
        face_colors: typing.Optional[typing.Union[str, dict[int, str]]] = None,
        face_weights: typing.Optional[typing.Union[int, dict[int, int]]] = None
    ) -> None:
        """
        Add a triangular mesh stored in a firedrake.Mesh to a folium map.

        Parameters
        ----------
        geo_map
            Map to which the mesh plot should be added.
        mesh
            A firedrake mesh.
        unmarked_cell_marker
            Marker to be assigned to any unmarked cell (e.g. in problems which do not require the definition
            of subdomains).
            If not provided, it is set to 0.
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
        if unmarked_cell_marker is None:
            unmarked_cell_marker = 0

        if unmarked_face_marker is None:
            unmarked_face_marker = 0

        vertices = mesh.coordinates.dat.data_ro
        cells = mesh.coordinates.cell_node_map().values

        unique_cell_markers = tuple(mesh.topology_dm.getLabelIdIS(
            firedrake.cython.dmcommon.CELL_SETS_LABEL).indices.tolist())
        unique_face_markers = tuple(mesh.topology_dm.getLabelIdIS(
            firedrake.cython.dmcommon.FACE_SETS_LABEL).indices.tolist())

        cell_markers = np.full((cells.shape[0], ), unmarked_cell_marker, dtype=np.int64)
        for cm in unique_cell_markers:
            cell_markers[mesh.cell_subset(cm).indices] = cm

        exterior_facet_size = mesh.exterior_facets.measure_set("exterior_facet", "everywhere").size
        exterior_facet_markers = np.full((exterior_facet_size, ), unmarked_face_marker, dtype=np.int64)
        interior_facet_size = mesh.interior_facets.measure_set("interior_facet", "everywhere").size
        interior_facet_markers = np.full((interior_facet_size, ), unmarked_face_marker, dtype=np.int64)
        for fm in unique_face_markers:
            exterior_facet_markers[mesh.exterior_facets.measure_set("exterior_facet", fm).indices] = fm
            interior_facet_markers[mesh.interior_facets.measure_set("interior_facet", fm).indices] = fm

        face_markers = np.full(cells.shape, unmarked_face_marker, dtype=np.int64)
        # The local face to vertex connectivity in FInAT is
        # 0: (1, 2), 1: (0, 2), 2: (0, 1)
        # while in FEMlium we assume
        # 0: (0, 1), 1: (1, 2), 2: (0, 2)
        finat_to_femlium = {0: 1, 1: 2, 2: 0}

        exterior_facet_to_cell_connectivity = mesh.exterior_facets.facet_cell_map.values.reshape(-1)
        exterior_facet_reference_face_number = mesh.exterior_facets.local_facet_dat.data_ro.reshape(-1)
        assert exterior_facet_to_cell_connectivity.shape == exterior_facet_reference_face_number.shape
        assert len(exterior_facet_to_cell_connectivity.shape) == 1
        for global_face_number in range(exterior_facet_to_cell_connectivity.shape[0]):
            cell = exterior_facet_to_cell_connectivity[global_face_number]
            femlium_face_number = finat_to_femlium[exterior_facet_reference_face_number[global_face_number]]
            face_markers[cell, femlium_face_number] = exterior_facet_markers[global_face_number]

        interior_facet_to_cell_connectivity = mesh.interior_facets.facet_cell_map.values
        interior_facet_reference_face_number = mesh.interior_facets.local_facet_dat.data_ro
        assert interior_facet_to_cell_connectivity.shape == interior_facet_reference_face_number.shape
        assert len(interior_facet_to_cell_connectivity.shape) == 2
        for global_face_number in range(interior_facet_to_cell_connectivity.shape[0]):
            cells_ = interior_facet_to_cell_connectivity[global_face_number, :].tolist()
            femlium_face_numbers = [
                finat_to_femlium[finat_face_number]
                for finat_face_number in interior_facet_reference_face_number[global_face_number]]
            for (cell, femlium_face_number) in zip(cells_, femlium_face_numbers):
                face_markers[cell, femlium_face_number] = interior_facet_markers[global_face_number]

        return BaseMeshPlotter.add_mesh_to(
            self, geo_map, vertices, cells, cell_markers, face_markers, cell_colors, face_colors, face_weights)

    def add_scalar_field_to(
        self, geo_map: folium.Map, scalar_field: firedrake.Function, mode: typing.Optional[str] = None,
        levels: typing.Optional[typing.Union[int, list[float]]] = None,
        cmap: typing.Optional[str] = None, name: typing.Optional[str] = None
    ) -> None:
        """
        Add a scalar field to a folium map.

        Parameters
        ----------
        geo_map
            Map to which the mesh plot should be added.
        scalar_field
            A firedrake Function representing the scalar field.
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
        scalar_function_space = firedrake.FunctionSpace(mesh, "CG", 1)
        vertices = mesh.coordinates.dat.data_ro
        cells = mesh.coordinates.cell_node_map().values
        scalar_field_values = firedrake.Function(scalar_function_space).interpolate(scalar_field).vector().array()

        return BaseSolutionPlotter.add_scalar_field_to(
            self, geo_map, vertices, cells, scalar_field_values, mode, levels, cmap, name)

    def add_vector_field_to(
        self, geo_map: folium.Map, vector_field: firedrake.Function, mode: typing.Optional[str] = None,
        levels: typing.Optional[typing.Union[int, list[float]]] = None, scale: typing.Optional[float] = None,
        cmap: typing.Optional[str] = None, name: typing.Optional[str] = None
    ) -> None:
        """
        Add a vector field to a folium map.

        Parameters
        ----------
        geo_map
            Map to which the mesh plot should be added.
        vector_field
            A firedrake Function representing the vector field.
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
        vector_function_space = firedrake.VectorFunctionSpace(mesh, "CG", 1)
        vertices = mesh.coordinates.dat.data_ro
        cells = mesh.coordinates.cell_node_map().values
        vector_field_values = firedrake.Function(
            vector_function_space).interpolate(vector_field).vector().array().reshape(-1, 2)

        return BaseSolutionPlotter.add_vector_field_to(
            self, geo_map, vertices, cells, vector_field_values, mode, levels, scale, cmap, name)
