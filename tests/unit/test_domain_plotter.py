# Copyright (C) 2021 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT

import pytest
import numpy as np
import pyproj
import json
import folium
import femlium


@pytest.fixture
def transformer():
    """
    Transformer between EPSG 3395 used in the definition of the vertices and
    EPSG 4326 used on the map.
    """
    return pyproj.Transformer.from_crs("epsg:3395", "epsg:4326", always_xy=True)


@pytest.fixture
def loop():
    """
    Closed loop over the vertices of a unit square domain.
    """

    return np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0., 0.]])


@pytest.fixture
def single_marker():
    """
    Vertex marker with a single value.
    """

    return np.zeros(4).astype(int)


@pytest.fixture
def multiple_markers():
    """
    Vertex marker with a multiple values.
    """

    return np.array([1, 2, 1, 2], dtype=np.dtype(int))


def test_domain_plotter_without_transformer_add_domain_to_map_vertices_only(loop):
    """
    Test femlium.DomainPlotter.add_domain_to in a case without transformer, and providing only
    the first input argument (vertices).
    """

    geo_map = folium.Map(location=[0, 0], zoom_start=8)
    domain_plotter = femlium.DomainPlotter()
    domain_plotter.add_domain_to(geo_map, loop)

    expected_geojson = {
        "features": [{
            "geometry": {
                "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
                "type": "MultiLineString"
            },
            "properties": {
                "color": "black",
                "weight": "1"
            },
            "type": "Feature"
        }],
        "type": "FeatureCollection"
    }
    expected_geojson = folium.utilities.normalize(json.dumps(expected_geojson))
    assert expected_geojson in folium.utilities.normalize(geo_map._parent.render())


def test_domain_plotter_with_transformer_add_domain_to_map_vertices_only(transformer, loop):
    """
    Test femlium.DomainPlotter.add_domain_to in a case with transformer, and providing only
    the first input argument (vertices).
    """

    geo_map = folium.Map(location=[0, 0], zoom_start=8)
    domain_plotter = femlium.DomainPlotter(transformer)
    domain_plotter.add_domain_to(geo_map, loop * 1e5)

    expected_geojson = {
        "features": [{
            "geometry": {
                "coordinates": [[[0.0, 0.0], [0.898315, 0.0], [0.898315, 0.904331], [0.0, 0.904331], [0.0, 0.0]]],
                "type": "MultiLineString"
            },
            "properties": {
                "color": "black",
                "weight": "1"
            },
            "type": "Feature"
        }],
        "type": "FeatureCollection"
    }
    expected_geojson = folium.utilities.normalize(json.dumps(expected_geojson))
    assert expected_geojson in folium.utilities.normalize(geo_map._parent.render())


def test_domain_plotter_add_domain_to_map_vertices_single_color_weight(loop):
    """
    Test femlium.DomainPlotter.add_domain_to when providing vertices, a uniform color and uniform weight.
    """

    geo_map = folium.Map(location=[0, 0], zoom_start=8)
    domain_plotter = femlium.DomainPlotter()
    domain_plotter.add_domain_to(geo_map, loop, colors="blue", weights=3)

    expected_geojson = {
        "features": [{
            "geometry": {
                "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
                "type": "MultiLineString"
            },
            "properties": {
                "color": "blue",
                "weight": "3"
            },
            "type": "Feature"
        }],
        "type": "FeatureCollection"
    }
    expected_geojson = folium.utilities.normalize(json.dumps(expected_geojson))
    assert expected_geojson in folium.utilities.normalize(geo_map._parent.render())


@pytest.mark.parametrize("offset", (0, 1))
def test_domain_plotter_add_domain_to_map_vertices_single_marker(loop, single_marker, offset):
    """
    Test femlium.DomainPlotter.add_domain_to when providing vertices and a single marker for all vertices.
    The marker value is equal to the default marker if the parameter offset is 0, while it is different
    from the default marker if the parameter offset is 1.
    """

    geo_map = folium.Map(location=[0, 0], zoom_start=8)
    domain_plotter = femlium.DomainPlotter()
    domain_plotter.add_domain_to(geo_map, loop, single_marker + offset)

    expected_geojson = {
        "features": [{
            "geometry": {
                "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
                "type": "MultiLineString"
            },
            "properties": {
                "color": "black",
                "weight": "1"
            },
            "type": "Feature"
        }],
        "type": "FeatureCollection"
    }
    expected_geojson = folium.utilities.normalize(json.dumps(expected_geojson))
    assert expected_geojson in folium.utilities.normalize(geo_map._parent.render())


@pytest.mark.parametrize("offset", (0, 1))
def test_domain_plotter_add_domain_to_map_vertices_single_marker_color_weight(loop, single_marker, offset):
    """
    Test femlium.DomainPlotter.add_domain_to when providing vertices, a single marker for all vertices,
    a uniform color and uniform weight.
    The marker value is equal to the default marker if the parameter offset is 0, while it is different
    from the default marker if the parameter offset is 1.
    """

    geo_map = folium.Map(location=[0, 0], zoom_start=8)
    domain_plotter = femlium.DomainPlotter()
    domain_plotter.add_domain_to(geo_map, loop, single_marker + offset, colors="blue", weights=3)

    expected_geojson = {
        "features": [{
            "geometry": {
                "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
                "type": "MultiLineString"
            },
            "properties": {
                "color": "blue",
                "weight": "3"
            },
            "type": "Feature"
        }],
        "type": "FeatureCollection"
    }
    expected_geojson = folium.utilities.normalize(json.dumps(expected_geojson))
    assert expected_geojson in folium.utilities.normalize(geo_map._parent.render())


def test_domain_plotter_add_domain_to_map_vertices_multiple_markers(loop, multiple_markers):
    """
    Test femlium.DomainPlotter.add_domain_to when providing vertices with different markers.
    """

    geo_map = folium.Map(location=[0, 0], zoom_start=8)
    domain_plotter = femlium.DomainPlotter()
    domain_plotter.add_domain_to(geo_map, loop, multiple_markers)

    expected_geojson = {
        "features": [{
            "geometry": {
                "coordinates": [[[0.0, 0.0], [1.0, 0.0]], [[1.0, 1.0], [0.0, 1.0]]],
                "type": "MultiLineString"
            },
            "id": "0",
            "properties": {
                "color": "black",
                "weight": "1"
            },
            "type": "Feature"
        }, {
            "geometry": {
                "coordinates": [[[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [0.0, 0.0]]],
                "type": "MultiLineString"
            },
            "id": "1",
            "properties": {
                "color": "black",
                "weight": "1"
            },
            "type": "Feature"
        }],
        "type": "FeatureCollection"
    }
    expected_geojson = folium.utilities.normalize(json.dumps(expected_geojson))
    assert expected_geojson in folium.utilities.normalize(geo_map._parent.render())


def test_domain_plotter_add_domain_to_map_vertices_multiple_markers_colors_weights(loop, multiple_markers):
    """
    Test femlium.DomainPlotter.add_domain_to when providing vertices with different markers,
    multiple colors and multiple weights.
    """

    geo_map = folium.Map(location=[0, 0], zoom_start=8)
    domain_plotter = femlium.DomainPlotter()
    colors = {
        1: "blue",
        2: "red"
    }
    weights = {
        1: 2,
        2: 3
    }
    domain_plotter.add_domain_to(geo_map, loop, multiple_markers, colors=colors, weights=weights)

    expected_geojson = {
        "features": [{
            "geometry": {
                "coordinates": [[[0.0, 0.0], [1.0, 0.0]], [[1.0, 1.0], [0.0, 1.0]]],
                "type": "MultiLineString"
            },
            "properties": {
                "color": "blue",
                "weight": "2"
            },
            "type": "Feature"
        }, {
            "geometry": {
                "coordinates": [[[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [0.0, 0.0]]],
                "type": "MultiLineString"
            },
            "properties": {
                "color": "red",
                "weight": "3"
            },
            "type": "Feature"
        }],
        "type": "FeatureCollection"
    }
    expected_geojson = folium.utilities.normalize(json.dumps(expected_geojson))
    geo_map_html = folium.utilities.normalize(geo_map._parent.render())
    assert expected_geojson in geo_map_html

    # Confirm presence of color bar
    assert "d3.scale.linear().domain([1.0,2.0])" in geo_map_html


def test_domain_plotter_add_domain_to_map_vertices_multiple_markers_missing_colors_weights(
        loop, multiple_markers):
    """
    Test femlium.DomainPlotter.add_domain_to when providing vertices with different markers,
    but the provided colors and weights do not map all the available marker values.
    In this case, any unmapped marker value will revert to default color and weight.
    """

    geo_map = folium.Map(location=[0, 0], zoom_start=8)
    domain_plotter = femlium.DomainPlotter()
    colors = {
        1: "blue"
    }
    weights = {
        1: 2
    }
    domain_plotter.add_domain_to(geo_map, loop, multiple_markers, colors=colors, weights=weights)

    expected_geojson = {
        "features": [{
            "geometry": {
                "coordinates": [[[0.0, 0.0], [1.0, 0.0]], [[1.0, 1.0], [0.0, 1.0]]],
                "type": "MultiLineString"
            },
            "properties": {
                "color": "blue",
                "weight": "2"
            },
            "type": "Feature"
        }, {
            "geometry": {
                "coordinates": [[[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [0.0, 0.0]]],
                "type": "MultiLineString"
            },
            "properties": {
                "color": "black",
                "weight": "1"
            },
            "type": "Feature"
        }],
        "type": "FeatureCollection"
    }
    expected_geojson = folium.utilities.normalize(json.dumps(expected_geojson))
    geo_map_html = folium.utilities.normalize(geo_map._parent.render())
    assert expected_geojson in geo_map_html

    # Confirm presence of color bar
    assert "d3.scale.linear().domain([1.0,2.0])" in geo_map_html
