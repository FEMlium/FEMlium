# Copyright (C) 2021-2022 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT

import numpy as np
import geojson
import folium
from femlium.base_plotter import BasePlotter
from femlium.utils import ColorbarWrapper


class DomainPlotter(BasePlotter):
    """
    This class contains a geographic plotter for the computational domain.
    """

    def add_domain_to(self, geo_map, vertices, segment_markers=None, colors=None, weights=None):
        """
        Add a domain to a folium map.

        Parameters
        ----------
        geo_map : folium.Map
            Map to which the domain plot should be added.
        vertices: 2d numpy array
            Matrix containing the coordinates of the vertices.
            The matrix should have as many rows as vertices in the domain, and two columns.
            The domain will be constructed connecting vertices at two consecutive rows.
            Note that, since the domain is enclosed by a closed curve, the first and last row
            should be equal (i.e., the start/end point of the closed curve should be repeated twice).
        segment_markers: 1d numpy array, optional
            Vector containing a marker (i.e., an integer number) for each face.
            The vector should have a number of entries equal to the number of vertices in the domain
            minus 1. The i-th segment is defined as the segment connecting the i-th vertex to the
            (i+1)-th vertex.
            If not provided, the marker will be set to 0 everywhere.
        colors: str or dict of str, optional
            If a dictionary is provided, it should contain key: value pairs defining the mapping
            marker: color.
            If a string is provided instead of a dictionary, the same color will be used for all markers.
            If not provided, a default black color will be used.
        weights: int or dict of int, optional
            Line weight of each segment. Input should be provided following a similar convention for
            the colors argument.
            If not provided, a unit weight will be used.
        """

        if segment_markers is None:
            segment_markers = np.zeros((vertices.shape[0] - 1, ), dtype=np.dtype(int))
        else:
            assert segment_markers.shape[0] == vertices.shape[0] - 1

        unique_markers = np.unique(segment_markers).astype(int)
        colors = self._process_optional_argument_on_markers(colors, "black", unique_markers)
        weights = self._process_optional_argument_on_markers(weights, 1, unique_markers)

        def style_function(x):
            return {
                "color": x["properties"]["color"],
                "weight": x["properties"]["weight"]
            }

        json = self._convert_domain_to_geojson(vertices, segment_markers, colors, weights)
        folium.GeoJson(json, style_function=style_function).add_to(geo_map)

        colors_values = np.arange(0, np.max(unique_markers) + 1)
        assert colors.shape == colors_values.shape
        colors_in_figure = np.delete(colors, np.setdiff1d(colors_values, unique_markers))
        colors_values_in_figure = np.delete(colors_values, np.setdiff1d(colors_values, unique_markers))
        if np.unique(colors_in_figure).shape[0] > 1:
            colorbar = ColorbarWrapper(
                colors=colors_in_figure, values=colors_values_in_figure, caption="Segment markers")
            colorbar.add_to(geo_map)

    def _convert_domain_to_geojson(self, vertices, segment_markers, colors, weights):
        """
        Convert a domain to a geojson FeatureCollection.

        Parameters
        ----------
        vertices: 2d numpy array
            Matrix containing the coordinates of the vertices.
            The matrix should have as many rows as vertices in the mesh, and two columns.
        segment_markers: 1d numpy array
            Vector containing a marker (i.e., an integer number) for each segment.
            The vector should have a number of entries equal to the number of vertices in the domain
            minus 1.
        colors: 1d numpy array
            Vector associating a marker to its color (i.e., a string).
            The vector should have as many entries as the number of markers.
        weights: 1d numpy array
            Vector associating a marker to its weight (i.e., a int).
            The vector should have as many entries as the number of markers.

        Returns
        -------
        geojson.FeatureCollection
            A geojson FeatureCollection representing the domain.
        """

        multiline_coordinates = dict()
        multiline_properties = dict()
        previous_marker = segment_markers[0]
        previous_coordinates = np.array(self.transformer(*vertices[0, :]))
        finalize_placeholder = "finalize"
        for v in range(vertices.shape[0]):
            if v < vertices.shape[0] - 1:
                next_vertex = self.transformer(*vertices[v + 1, :])
                current_marker = segment_markers[v]
            else:
                current_marker = finalize_placeholder
            if current_marker != previous_marker:
                # Store previous line
                if previous_marker not in multiline_coordinates:
                    multiline_coordinates[previous_marker] = list()
                multiline_coordinates[previous_marker].append(previous_coordinates.tolist())
                # Store properties of the previous line
                previous_properties = {
                    "color": colors[previous_marker],
                    "weight": int(weights[previous_marker])
                }
                if previous_marker not in multiline_properties:
                    multiline_properties[previous_marker] = previous_properties
                else:
                    assert multiline_properties[previous_marker] == previous_properties
                # Reset in preparation of new line
                previous_marker = current_marker
                previous_coordinates = np.array(previous_coordinates[-1, :])
            if current_marker != finalize_placeholder:
                previous_coordinates = np.vstack((previous_coordinates, next_vertex))

        multiline_features = list()
        for marker in multiline_coordinates.keys():
            multiline = geojson.MultiLineString(coordinates=multiline_coordinates[marker])
            feature = geojson.Feature(
                geometry=multiline,
                properties=multiline_properties[marker]
            )
            multiline_features.append(feature)
        return geojson.FeatureCollection(multiline_features)
