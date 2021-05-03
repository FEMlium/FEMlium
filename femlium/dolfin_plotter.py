# Copyright (C) 2021 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT

import numpy as np
from femlium.base_mesh_plotter import BaseMeshPlotter
from femlium.base_solution_plotter import BaseSolutionPlotter


class DolfinPlotter(BaseMeshPlotter, BaseSolutionPlotter):
    """
    This class contains the interface of a geographic plotter for dolfin meshes and solutions.
    """

    def add_mesh_to(self, geo_map, mesh,
                    cell_mesh_function=None, face_mesh_function=None,
                    cell_colors=None, face_colors=None, face_weights=None):
        """
        Add a triangular mesh stored in a dolfin.Mesh to a folium map.

        Parameters
        ----------
        geo_map : folium.Map
            Map to which the mesh plot should be added.
        mesh: dolfin.Mesh
            A dolfin mesh.
        cell_mesh_function: dolfin.MeshFunction, optional
            A dolfin mesh function of topological dimension 2 that stores cell markers.
            If not provided, the marker will be set to 0 everywhere.
        face_mesh_function: dolfin.MeshFunction, optional
            A dolfin mesh function of topological dimension 1 that stores face markers.
            If not provided, the marker will be set to 0 everywhere.
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

        vertices = mesh.coordinates()
        cells = mesh.cells()

        if cell_mesh_function is not None:
            cell_markers = cell_mesh_function.array()
        else:
            cell_markers = None

        if face_mesh_function is not None:
            mesh.init(2, 1)
            cell_to_faces_connectivity = mesh.topology()(2, 1)

            fiat_to_femlium = {0: 1, 1: 2, 2: 0}

            face_markers = np.zeros(cells.shape, dtype=np.dtype(int))
            for c in range(cells.shape[0]):
                for (f, global_face_number) in enumerate(cell_to_faces_connectivity(c)):
                    face_markers[c, fiat_to_femlium[f]] = face_mesh_function[global_face_number]
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
        scalar_field: dolfin.Function
            A dolfin Function representing the scalar field.
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

        mesh = scalar_field.function_space().mesh()
        vertices = mesh.coordinates()
        cells = mesh.cells()
        scalar_field_values = scalar_field.compute_vertex_values(mesh)

        return BaseSolutionPlotter.add_scalar_field_to(
            self, geo_map, vertices, cells, scalar_field_values, mode, levels, cmap, name)

    def add_vector_field_to(self, geo_map, vector_field, mode=None, levels=None, scale=None, cmap=None, name=None):
        """
        Add a vector field to a folium map.

        Parameters
        ----------
        geo_map : folium.Map
            Map to which the mesh plot should be added.
        vector_field: dolfin.Function
            A dolfin Function representing the vector field.
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

        mesh = vector_field.function_space().mesh()
        vertices = mesh.coordinates()
        cells = mesh.cells()
        vector_field_values = vector_field.compute_vertex_values(mesh).reshape(2, -1).T

        return BaseSolutionPlotter.add_vector_field_to(
            self, geo_map, vertices, cells, vector_field_values, mode, levels, scale, cmap, name)
