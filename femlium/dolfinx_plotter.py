# Copyright (C) 2021-2022 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT

import numpy as np
from femlium.base_mesh_plotter import BaseMeshPlotter
from femlium.base_solution_plotter import BaseSolutionPlotter


class DolfinxPlotter(BaseMeshPlotter, BaseSolutionPlotter):
    """
    This class contains the interface of a geographic plotter for dolfinx meshes and solutions.
    """

    def add_mesh_to(self, geo_map, mesh,
                    cell_mesh_tags=None, face_mesh_tags=None,
                    unmarked_cell_marker=None, unmarked_face_marker=None,
                    cell_colors=None, face_colors=None, face_weights=None):
        """
        Add a triangular mesh stored in a dolfinx.Mesh to a folium map.

        Parameters
        ----------
        geo_map : folium.Map
            Map to which the mesh plot should be added.
        mesh: dolfinx.Mesh
            A dolfinx mesh.
        cell_mesh_tags: dolfinx.MeshTags, optional
            A dolfinx mesh tags of topological dimension 2 that stores cell markers.
            If not provided, the marker will be set to 0 everywhere.
        face_mesh_tags: dolfinx.MeshTags, optional
            A dolfinx mesh tags of topological dimension 1 that stores face markers.
            If not provided, the marker will be set to 0 everywhere.
        unmarked_cell_marker: int, optional
            Marker to be assigned to any unmarked cell (e.g. in problems which do not require the definition
            of subdomains).
            If not provided, it is set to 0.
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

        if unmarked_cell_marker is None:
            unmarked_cell_marker = 0

        if unmarked_face_marker is None:
            unmarked_face_marker = 0

        vertices = mesh.geometry.x[:, :mesh.topology.dim]
        cells = mesh.geometry.dofmap.array.reshape((-1, mesh.topology.dim + 1))

        if cell_mesh_tags is not None:
            cell_markers = np.full((cells.shape[0], ), unmarked_cell_marker, dtype=np.dtype(int))
            cell_markers[cell_mesh_tags.indices] = cell_mesh_tags.values
        else:
            cell_markers = None

        if face_mesh_tags is not None:
            mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
            mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
            face_to_cells_connectivity = mesh.topology.connectivity(mesh.topology.dim - 1, mesh.topology.dim)
            cell_to_faces_connectivity = mesh.topology.connectivity(mesh.topology.dim, mesh.topology.dim - 1)

            # The local face to vertex connectivity in basix is
            # 0: (1, 2), 1: (0, 2), 2: (0, 1)
            # while in FEMlium we assume
            # 0: (0, 1), 1: (1, 2), 2: (0, 2)
            basix_to_femlium = {0: 1, 1: 2, 2: 0}

            face_markers = np.full(cells.shape, unmarked_face_marker, dtype=np.dtype(int))
            for (global_face_number, face_mesh_tag) in zip(face_mesh_tags.indices, face_mesh_tags.values):
                cells_ = face_to_cells_connectivity.links(global_face_number)
                basix_face_numbers = np.array([
                    np.argwhere(cell_to_faces_connectivity.links(c) == global_face_number)[0] for c in cells_])
                assert basix_face_numbers.shape == (cells_.shape[0], 1)
                femlium_face_numbers = [
                    basix_to_femlium[basix_face_number] for basix_face_number in basix_face_numbers.reshape(-1)]
                for (cell, femlium_face_number) in zip(cells_, femlium_face_numbers):
                    face_markers[cell, femlium_face_number] = face_mesh_tag
        else:
            face_markers = None

        return BaseMeshPlotter.add_mesh_to(
            self, geo_map, vertices, cells, cell_markers, face_markers, cell_colors, face_colors, face_weights)

    def add_scalar_field_to(self, geo_map, scalar_field, mode=None, levels=None, cmap=None, name=None):
        """
        Add a scalar field to a folium map.

        Parameters
        ----------
        geo_map : folium.Map
            Map to which the mesh plot should be added.
        scalar_field: dolfinx.Function
            A dolfinx Function representing the scalar field.
        mode: str, optional
            Plot to be generated, either contourf or contour.
            If not provided, contourf is used.
        levels: int or list of numbers, optional
            Values of the contour lines.
            If integer, it will determine the number of equispaced values to be used between
            the minimum and maximum entry of the scalar field.
            If list, it will determine the values to be used.
            If not provided, 10 levels are used by default.
        cmap: str, optional
            matplotlib color map to be used.
            If not provided, the jet colormap is used.
        name: str, optional
            Name of the field, to be used in the creation of the color bar.
            If not provided, the name "scalar field" will be used.
        """

        mesh = scalar_field.function_space.mesh
        vertices = mesh.geometry.x[:, :mesh.topology.dim]
        cells = mesh.geometry.dofmap.array.reshape((-1, mesh.topology.dim + 1))
        scalar_field_values = scalar_field.compute_point_values()
        assert scalar_field_values.shape[1] == 1
        scalar_field_values = scalar_field_values.reshape(-1)

        return BaseSolutionPlotter.add_scalar_field_to(
            self, geo_map, vertices, cells, scalar_field_values, mode, levels, cmap, name)

    def add_vector_field_to(self, geo_map, vector_field, mode=None, levels=None, scale=None, cmap=None, name=None):
        """
        Add a vector field to a folium map.

        Parameters
        ----------
        geo_map : folium.Map
            Map to which the mesh plot should be added.
        vector_field: dolfinx.Function
            A dolfinx Function representing the vector field.
        mode: str, optional
            Plot to be generated, either contourf, contour or quiver.
            If not provided, contourf is used.
        levels: int or list of numbers, optional
            In contourf or contour mode: values of the contour lines.
            In quiver mode: number of ticks to be added to the color bar.
            If integer, it will determine the number of equispaced values to be used between
            the minimum and maximum entry of the scalar field.
            If list, it will determine the values to be used.
            If not provided, 10 levels are used by default.
        scale: number, optional
            This is only applicable for quiver mode: scaling to be applied before drawing arrows.
            If not provided, no scaling will be applied.
        cmap: str, optional
            matplotlib color map to be used.
            If not provided, the jet colormap is used.
        name: str, optional
            Name of the field, to be used in the creation of the color bar.
            If not provided, the name "vector field" will be used.
        """

        mesh = vector_field.function_space.mesh
        vertices = mesh.geometry.x[:, :mesh.topology.dim]
        cells = mesh.geometry.dofmap.array.reshape((-1, mesh.topology.dim + 1))
        vector_field_values = vector_field.compute_point_values()

        return BaseSolutionPlotter.add_vector_field_to(
            self, geo_map, vertices, cells, vector_field_values, mode, levels, scale, cmap, name)
