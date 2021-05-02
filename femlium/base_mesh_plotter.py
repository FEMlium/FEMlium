# Copyright (C) 2021 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT

import numpy as np
import geojson
import folium
import branca
from femlium.base_plotter import BasePlotter


class BaseMeshPlotter(BasePlotter):
    """
    This class contains the base interface of a geographic plotter for mesh-related plots.
    """

    def add_mesh_to(self, geo_map, vertices, cells,
                    cell_markers=None, face_markers=None,
                    cell_colors=None, face_colors=None, face_weights=None):
        """
        Add a triangular mesh to a folium map.

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
        cell_markers: 1d numpy array, optional
            Vector containing a marker (i.e., an integer number) for each cell.
            The vector should have as many entries as cells in the mesh.
            If not provided, the marker will be set to 0 everywhere.
        face_markers: 2d numpy array, optional
            Vector containing a marker (i.e., an integer number) for each face.
            The matrix should have the same shape of the cells argument.
            Given a row index r, the entry face_markers[r, 0] is the marker of the
            face connecting the first and second vertex of the r-th cell.
            Similarly, face_markers[r, 1] is the marker associated to the face connecting
            the second and third vertex of the r-th cell. Finally, face_markers[r, 2] is
            the marker associated to the face connecting the first and third vertex of the
            r-th cell.
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

        if cell_markers is None:
            cell_markers = np.zeros((cells.shape[0], ), dtype=np.dtype(int))
        else:
            assert cell_markers.shape[0] == cells.shape[0]

        if face_markers is None:
            face_markers = np.zeros(cells.shape, dtype=np.dtype(int))
        else:
            assert face_markers.shape == cells.shape

        unique_cell_markers = np.unique(cell_markers).astype(int)
        unique_face_markers = np.unique(face_markers).astype(int)
        cell_colors = self._process_optional_argument_on_markers(cell_colors, "none", unique_cell_markers)
        face_colors = self._process_optional_argument_on_markers(face_colors, "black", unique_face_markers)
        face_weights = self._process_optional_argument_on_markers(face_weights, 1, unique_face_markers)

        def style_function(x):
            if x["geometry"]["type"] == "MultiPolygon":
                return {
                    # Boundary properties
                    "stroke": x["properties"]["stroke"],
                    "color": x["properties"]["color"],
                    "weight": x["properties"]["weight"],
                    # Interior properties
                    "fill": x["properties"]["fill"],
                    "fillColor": x["properties"]["fillColor"],
                    "fillOpacity": x["properties"]["fillOpacity"]
                }
            elif x["geometry"]["type"] == "MultiLineString":
                return {
                    "stroke": x["properties"]["stroke"],
                    "color": x["properties"]["color"],
                    "weight": x["properties"]["weight"]
                }
            else:
                raise ValueError("Invalid type")

        json = self._convert_mesh_to_geojson(
            vertices, cells, cell_markers, face_markers, cell_colors, face_colors, face_weights)
        folium.GeoJson(
            json,
            style_function=style_function
        ).add_to(geo_map)

        cell_colors_where_none = np.argwhere(cell_colors == "none")
        cell_colors_not_none = np.delete(cell_colors, cell_colors_where_none)
        cell_colors_values = np.arange(0, np.max(unique_cell_markers) + 1)
        cell_colors_values_not_none = np.delete(cell_colors_values, cell_colors_where_none)
        assert cell_colors_not_none.shape == cell_colors_values_not_none.shape
        if np.unique(cell_colors_not_none).shape[0] > 1:
            colormap = branca.colormap.LinearColormap(
                            colors=cell_colors_not_none,
                            index=cell_colors_values_not_none,
                            vmin=cell_colors_values_not_none[0], vmax=cell_colors_values_not_none[-1],
                            caption="Cell markers")
            geo_map.add_child(colormap)

        face_colors_values = np.arange(0, np.max(unique_face_markers) + 1)
        assert face_colors.shape == face_colors_values.shape
        if np.unique(face_colors).shape[0] > 1:
            colormap = branca.colormap.LinearColormap(
                            colors=face_colors,
                            index=face_colors_values, vmin=face_colors_values[0], vmax=face_colors_values[-1],
                            caption="Face markers")
            geo_map.add_child(colormap)

    def _convert_mesh_to_geojson(self, vertices, cells, cell_markers, face_markers,
                                 cell_colors, face_colors, face_weights):
        """
        Convert a mesh to a geojson FeatureCollection.

        Parameters
        ----------
        vertices: 2d numpy array
            Matrix containing the coordinates of the vertices.
            The matrix should have as many rows as vertices in the mesh, and two columns.
        cells: 2d numpy array
            Matrix containing the connectivity of the cells.
            The matrix should have as many rows as cells in the mesh, and three columns.
        cell_markers: 1d numpy array
            Vector containing a marker (i.e., an integer number) for each cell.
            The vector should have as many entries as cells in the mesh.
        face_markers: 2d numpy array
            Vector containing a marker (i.e., an integer number) for each face.
            The matrix should have the same shape of the cells argument.
        cell_colors: 1d numpy array
            Vector associating a cell marker to its color (i.e., a string).
            The vector should have as many entries as the number of cell markers.
        face_colors: 1d numpy array
            Vector associating a face marker to its color (i.e., a string).
            The vector should have as many entries as the number of face markers.
        face_weights: 1d numpy array
            Vector associating a face marker to its weight (i.e., a int).
            The vector should have as many entries as the number of face markers.

        Returns
        -------
        geojson.FeatureCollection
            A geojson FeatureCollection representing the mesh.
        """

        multipolygon_coordinates = dict()
        multipolygon_properties = dict()
        multiline_coordinates = dict()
        multiline_properties = dict()
        for c in range(cells.shape[0]):
            coordinates = [self.transformer(*vertices[cells[c, v], :]) for v in range(3)]
            coordinates.append(coordinates[0])
            cell_face_markers = np.unique([face_markers[c, f] for f in range(3)]).astype(int)
            if cell_face_markers.shape[0] == 1:
                cell_key = (cell_markers[c], True)
                cell_properties = {
                    # Boundary properties
                    "stroke": True,
                    "color": face_colors[cell_face_markers[0]],
                    "weight": str(face_weights[cell_face_markers[0]]),
                }
            else:
                cell_key = (cell_markers[c], False)
                cell_properties = {
                    # Boundary properties
                    "stroke": False,
                    "color": None,
                    "weight": None
                }
            if cell_colors[cell_key[0]] != "none":
                cell_properties.update({
                    # Interior properties
                    "fill": True,
                    "fillColor": cell_colors[cell_key[0]],
                    "fillOpacity": str(1)
                })
            else:
                cell_properties.update({
                    # Interior properties
                    "fill": False,
                    "fillColor": None,
                    "fillOpacity": None
                })
            # Store current cell
            if cell_key not in multipolygon_coordinates:
                multipolygon_coordinates[cell_key] = list()
            multipolygon_coordinates[cell_key].append([coordinates])
            # Store current cell properties
            if cell_key not in multipolygon_properties:
                multipolygon_properties[cell_key] = cell_properties
            else:
                assert multipolygon_properties[cell_key] == cell_properties
            # Store faces only if there are multiple face markers in this cell,
            # otherwise the boundary representation of the cell is sufficient.
            if not cell_key[1]:
                for (f, pair) in enumerate(((0, 1), (1, 2), (0, 2))):
                    face_key = face_markers[c, f]
                    # Store current face
                    if face_key not in multiline_coordinates:
                        multiline_coordinates[face_key] = list()
                    multiline_coordinates[face_key].append([coordinates[pair[0]], coordinates[pair[1]]])
                    # Store current face properties
                    face_properties = {
                        "stroke": True,
                        "color": face_colors[face_markers[c, f]],
                        "weight": str(face_weights[face_markers[c, f]]),
                    }
                    if face_key not in multiline_properties:
                        multiline_properties[face_key] = face_properties
                    else:
                        assert multiline_properties[face_key] == face_properties

        multipolygon_features = list()
        for cell_key in multipolygon_coordinates.keys():
            multipolygon = geojson.MultiPolygon(coordinates=multipolygon_coordinates[cell_key])
            feature = geojson.Feature(
                geometry=multipolygon,
                properties=multipolygon_properties[cell_key]
            )
            multipolygon_features.append(feature)

        multiline_features = list()
        for face_key in multiline_coordinates.keys():
            multiline = geojson.MultiLineString(coordinates=multiline_coordinates[face_key])
            feature = geojson.Feature(
                geometry=multiline,
                properties=multiline_properties[face_key]
            )
            multiline_features.append(feature)

        return geojson.FeatureCollection(multipolygon_features + multiline_features)
