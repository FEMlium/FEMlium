# Copyright (C) 2021 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT

import numpy as np
import matplotlib
import matplotlib._tri
import matplotlib.pyplot as plt
import geojson
import folium
from femlium.base_plotter import BasePlotter
from femlium.utils import ColorbarWrapper, GeoJsonWithArrows


class BaseSolutionPlotter(BasePlotter):
    """
    This class contains the base interface of a geographic plotter for solution-related plots.
    """

    def add_scalar_field_to(self, geo_map, vertices, cells, scalar_field,
                            mode=None, levels=None, cmap=None, name=None):
        """
        Add a scalar field to a folium map.

        Parameters
        ----------
        geo_map : folium.Map
            Map to which the mesh plot should be added.
        vertices: 2d numpy array
            Matrix containing the coordinates of the vertices.
            The matrix should have as many rows as vertices in the mesh, and two columns.
        cells: 2d numpy array
            Matrix containing the connectivity of the cells.
            The matrix should have as many rows as cells in the mesh, and three columns.
        scalar_field: 1d numpy array
            Vector containing the value of the field at each vertex.
            The vector should have as many entries as vertices in the mesh.
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

        if mode is None:
            mode = "contourf"
        assert mode in ("contourf", "contour")

        if levels is None:
            levels = 10
        assert isinstance(levels, (int, list, np.ndarray))
        if isinstance(levels, int):
            levels = np.linspace(scalar_field.min(), scalar_field.max(), levels)
        elif isinstance(levels, list):
            levels = np.array(levels)
        elif isinstance(levels, np.ndarray):
            pass
        else:
            raise ValueError("Invalid levels")

        if cmap is None:
            cmap = "jet"
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap, lut=levels.shape[0])
        cnorm = plt.Normalize(vmin=levels[0], vmax=levels[-1])
        colors = [matplotlib.colors.to_hex(cmap(cnorm(lev))) for lev in levels]

        if name is None:
            name = "Scalar field"

        def style_function(x):
            if x["geometry"]["type"] == "MultiPolygon":
                assert mode == "contourf"
                return {
                    # Boundary properties
                    "stroke": False,
                    # Interior properties
                    "fillColor": x["properties"]["fillColor"],
                    "fillOpacity": x["properties"]["fillOpacity"]
                }
            elif x["geometry"]["type"] == "MultiLineString":
                assert mode == "contour"
                return {
                    "color": x["properties"]["color"],
                    "weight": x["properties"]["weight"]
                }
            else:
                raise ValueError("Invalid type")

        json = self._convert_scalar_field_to_geojson(vertices, cells, scalar_field, mode, levels, colors)
        folium.GeoJson(json, style_function=style_function).add_to(geo_map)

        colorbar = ColorbarWrapper(colors=colors, values=levels, caption=name)
        colorbar.add_to(geo_map)

    def _convert_scalar_field_to_geojson(self, vertices, cells, scalar_field, mode, levels, colors):
        """
        Convert a scalar field to a geojson FeatureCollection.

        Parameters
        ----------
        geo_map : folium.Map
            Map to which the mesh plot should be added.
        vertices: 2d numpy array
            Matrix containing the coordinates of the vertices.
            The matrix should have as many rows as vertices in the mesh, and two columns.
        cells: 2d numpy array
            Matrix containing the connectivity of the cells.
            The matrix should have as many rows as cells in the mesh, and three columns.
        scalar_field: 1d numpy array
            Vector containing the value of the field at each vertex.
            The vector should have as many entries as vertices in the mesh.
        mode: str
            Plot to be generated, either contourf or contour.
        levels: 1d numpy array
            Values of the contour lines.
        colors: list of str
            Color associated to each level.

        Returns
        -------
        geojson.FeatureCollection
            A geojson FeatureCollection representing the scalar field.
        """

        tri = matplotlib.tri.Triangulation(vertices[:, 0], vertices[:, 1], cells)
        countour_generator = matplotlib._tri.TriContourGenerator(tri.get_cpp_triangulation(), scalar_field)

        if mode == "contour":
            multiline_coordinates = dict()
            multiline_properties = dict()
            for lev in range(levels.shape[0]):
                curves = countour_generator.create_contour(levels[lev])
                if isinstance(curves, tuple):  # non backward compatible change in matplotlib commit 178012
                    assert len(curves) == 2
                    curves = curves[0]
                assert isinstance(curves, list)
                for curve in curves:
                    assert len(curve.shape) == 2
                    if curve.shape[0] > 1:
                        coordinates = [self.transformer(*curve[p, :]) for p in range(curve.shape[0])]
                        # Store current curve coordinates
                        if lev not in multiline_coordinates:
                            multiline_coordinates[lev] = list()
                        multiline_coordinates[lev].append(coordinates)
                        # Store current curve properties
                        properties = {
                            "color": colors[lev],
                            "weight": 2
                        }
                        if lev not in multiline_properties:
                            multiline_properties[lev] = properties
                        else:
                            assert multiline_properties[lev] == properties

            multiline_features = list()
            for lev in multiline_coordinates.keys():
                multiline = geojson.MultiLineString(coordinates=multiline_coordinates[lev])
                feature = geojson.Feature(
                    geometry=multiline,
                    properties=multiline_properties[lev]
                )
                multiline_features.append(feature)

            return geojson.FeatureCollection(multiline_features)
        elif mode == "contourf":
            multipolygon_coordinates = dict()
            multipolygon_properties = dict()
            for lev in range(levels.shape[0] - 1):
                filled_contour = countour_generator.create_filled_contour(levels[lev], levels[lev + 1])
                assert len(filled_contour) == 2
                if isinstance(filled_contour[0], list):  # non backward compatible change in matplotlib commit 178012
                    assert isinstance(filled_contour[1], list)
                    assert len(filled_contour[0]) == 1
                    assert len(filled_contour[1]) == 1
                    filled_contour = (np.array(filled_contour[0][0]), np.array(filled_contour[1][0]))
                assert len(filled_contour[0].shape) == 2
                assert len(filled_contour[1].shape) == 1
                filled_contour_split = np.split(filled_contour[0], np.where(filled_contour[1] == 1)[0][1:])
                for i in range(len(filled_contour_split)):
                    filled_contour_split[i] = np.vstack((filled_contour_split[i], filled_contour_split[i][0, :]))
                for curve in filled_contour_split:
                    if curve.shape[0] > 2:
                        coordinates = [self.transformer(*curve[p, :]) for p in range(curve.shape[0])]
                        # Store current polygon coordinates
                        if lev not in multipolygon_coordinates:
                            multipolygon_coordinates[lev] = list()
                        multipolygon_coordinates[lev].append([coordinates])
                        # Store current polygon properties
                        properties = {
                            "fillColor": colors[lev],
                            "fillOpacity": 1
                        }
                        if lev not in multipolygon_properties:
                            multipolygon_properties[lev] = properties
                        else:
                            assert multipolygon_properties[lev] == properties

            multipolygon_features = list()
            for lev in multipolygon_coordinates.keys():
                multipolygon = geojson.MultiPolygon(coordinates=multipolygon_coordinates[lev])
                feature = geojson.Feature(
                    geometry=multipolygon,
                    properties=multipolygon_properties[lev]
                )
                multipolygon_features.append(feature)

            return geojson.FeatureCollection(multipolygon_features)
        else:
            raise ValueError("Invalid mode")

    def add_vector_field_to(self, geo_map, vertices, cells, vector_field,
                            mode=None, levels=None, scale=None, cmap=None, name=None):
        """
        Add a vector field to a folium map.

        Parameters
        ----------
        geo_map : folium.Map
            Map to which the mesh plot should be added.
        vertices: 2d numpy array
            Matrix containing the coordinates of the vertices.
            The matrix should have as many rows as vertices in the mesh, and two columns.
        cells: 2d numpy array
            Matrix containing the connectivity of the cells.
            The matrix should have as many rows as cells in the mesh, and three columns.
        vector_field: 2d numpy array
            Matrix containing the value of the field at each vertex.
            The matrix should have as many rows as vertices in the mesh, and two columns.
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

        if mode is None:
            mode = "contourf"
        assert mode in ("contourf", "contour", "quiver")

        vector_field_magnitude = np.linalg.norm(vector_field, axis=1)

        if levels is None:
            levels = 10
        assert isinstance(levels, (int, list, np.ndarray))
        if isinstance(levels, int):
            levels = np.linspace(vector_field_magnitude.min(), vector_field_magnitude.max(), levels)
        elif isinstance(levels, list):
            levels = np.array(levels)
        elif isinstance(levels, np.ndarray):
            pass
        else:
            raise ValueError("Invalid levels")

        if cmap is None:
            cmap = "jet"
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap, lut=levels.shape[0])
        cnorm = plt.Normalize(vmin=levels[0], vmax=levels[-1])

        if name is None:
            name = "Vector field"

        if mode in ("contourf", "contour"):
            return BaseSolutionPlotter.add_scalar_field_to(
                self, geo_map, vertices, cells, vector_field_magnitude, mode, levels, cmap, name)
        elif mode == "quiver":
            json = self._convert_vector_field_to_geojson(
                vertices, vector_field_magnitude, vector_field, scale,
                lambda lev: matplotlib.colors.to_hex(cmap(cnorm(lev))))

            def style_function(x):
                return {
                    "color": x["properties"]["color"],
                    "weight": x["properties"]["weight"]
                }

            GeoJsonWithArrows(json, style_function=style_function, frequency="endonly").add_to(geo_map)

            colors = [matplotlib.colors.to_hex(cmap(cnorm(lev))) for lev in levels]
            colorbar = ColorbarWrapper(colors=colors, values=levels, caption=name)
            colorbar.add_to(geo_map)
        else:
            raise ValueError("Invalid mode")

    def _convert_vector_field_to_geojson(self, vertices, vector_field_magnitude, vector_field, scale, cmap):
        """
        Convert a scalar field to a geojson FeatureCollection.

        Parameters
        ----------
        geo_map : folium.Map
            Map to which the mesh plot should be added.
        vertices: 2d numpy array
            Matrix containing the coordinates of the vertices.
            The matrix should have as many rows as vertices in the mesh, and two columns.
        vector_field_magnitude: 1d numpy array
            Vector containing the magnitude of the field at each vertex.
            The vector should have as many entries as vertices in the mesh.
        vector_field: 2d numpy array
            Matrix containing the value of the field at each vertex.
            The matrix should have as many rows as vertices in the mesh, and two columns.
        scale: number
            Scaling to be applied before drawing arrows.
        cmap: function
            Color map to be used.

        Returns
        -------
        geojson.FeatureCollection
            A geojson FeatureCollection representing the scalar field.
        """

        multiline_coordinates = dict()
        multiline_properties = dict()
        for v in range(vertices.shape[0]):
            coordinates = [
                self.transformer(*vertices[v, :]),
                self.transformer(*(vertices[v, :] + scale * vector_field[v, :]))]
            color = cmap(vector_field_magnitude[v])
            # Store current coordinates
            if color not in multiline_coordinates:
                multiline_coordinates[color] = list()
            multiline_coordinates[color].append(coordinates)
            # Store current properties
            properties = {
                "color": color,
                "weight": 2
            }
            if color not in multiline_properties:
                multiline_properties[color] = properties
            else:
                assert multiline_properties[color] == properties

        multiline_features = list()
        for color in multiline_coordinates.keys():
            multiline = geojson.MultiLineString(coordinates=multiline_coordinates[color])
            feature = geojson.Feature(
                geometry=multiline,
                properties=multiline_properties[color]
            )
            multiline_features.append(feature)

        return geojson.FeatureCollection(multiline_features)
