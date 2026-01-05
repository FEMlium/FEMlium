# Copyright (C) 2021-2026 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT
"""Tests for femlium.base_mesh_plotter module."""

import json

import folium
import numpy as np
import numpy.typing as npt
import pytest

import femlium


@pytest.fixture
def vertices() -> npt.NDArray[np.float64]:
    """Vertices of a unit square domain divided in two triangular cells."""
    return np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])


@pytest.fixture
def cells() -> npt.NDArray[np.int64]:
    """Cells of a unit square domain divided in two triangular cells."""
    return np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)


@pytest.fixture
def single_face_marker() -> npt.NDArray[np.int64]:
    """Face marker with a single value."""
    return np.zeros((2, 3), dtype=np.int64)


@pytest.fixture
def multiple_face_markers() -> npt.NDArray[np.int64]:
    """
    Face marker with a multiple values.

    Such values are chosen so that the first cell will have a uniform face marker, while the second cell
    will have a different value on different faces.
    """
    return np.array([[3, 3, 3], [3, 1, 2]], dtype=np.int64)


@pytest.fixture
def single_cell_marker() -> npt.NDArray[np.int64]:
    """Cell marker with a single value."""
    return np.zeros(2, dtype=np.int64)


@pytest.fixture
def multiple_cell_markers() -> npt.NDArray[np.int64]:
    """Cell marker with a multiple values."""
    return np.array([1, 2], dtype=np.int64)


def test_base_mesh_plotter_add_mesh_to_map_vertices_cells_only(
    vertices: npt.NDArray[np.float64], cells: npt.NDArray[np.int64]
) -> None:
    """Test femlium.BaseMeshPlotter.add_mesh_to providing only the required input arguments (vertices and cells)."""
    geo_map = folium.Map(location=[0, 0], zoom_start=8)
    mesh_plotter = femlium.BaseMeshPlotter()
    mesh_plotter.add_mesh_to(geo_map, vertices, cells)

    expected_geojson = {
        "features": [{
            "geometry": {
                "coordinates": [
                    [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
                    [[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]],
                "type": "MultiPolygon"
            },
            "properties": {
                "color": "black",
                "fill": False,
                "fillColor": None,
                "fillOpacity": None,
                "stroke": True,
                "weight": 1
            },
            "type": "Feature"
        }],
        "type": "FeatureCollection"
    }
    expected_geojson = folium.utilities.normalize(json.dumps(expected_geojson))
    assert expected_geojson in folium.utilities.normalize(geo_map._parent.render())


def test_base_mesh_plotter_add_mesh_to_map_vertices_cells_single_face_color_weight(
    vertices: npt.NDArray[np.float64], cells: npt.NDArray[np.int64]
) -> None:
    """Test femlium.BaseMeshPlotter.add_mesh_to providing vertices, cells, a uniform face color & weight."""
    geo_map = folium.Map(location=[0, 0], zoom_start=8)
    mesh_plotter = femlium.BaseMeshPlotter()
    mesh_plotter.add_mesh_to(geo_map, vertices, cells, face_colors="red", face_weights=2)

    expected_geojson = {
        "features": [{
            "geometry": {
                "coordinates": [
                    [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
                    [[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]],
                "type": "MultiPolygon"
            },
            "properties": {
                "color": "red",
                "fill": False,
                "fillColor": None,
                "fillOpacity": None,
                "stroke": True,
                "weight": 2
            },
            "type": "Feature"
        }],
        "type": "FeatureCollection"
    }
    expected_geojson = folium.utilities.normalize(json.dumps(expected_geojson))
    assert expected_geojson in folium.utilities.normalize(geo_map._parent.render())


@pytest.mark.parametrize("offset", (0, 1))
def test_base_mesh_plotter_add_mesh_to_map_vertices_cells_single_face_marker(
    vertices: npt.NDArray[np.float64], cells: npt.NDArray[np.int64],
    single_face_marker: npt.NDArray[np.int64], offset: int
) -> None:
    """
    Test femlium.BaseMeshPlotter.add_mesh_to providing vertices, cells, and a single marker for all faces.

    The face marker value is equal to the default marker if the parameter offset is 0, while it is different
    from the default marker if the parameter offset is 1.
    """
    geo_map = folium.Map(location=[0, 0], zoom_start=8)
    mesh_plotter = femlium.BaseMeshPlotter()
    mesh_plotter.add_mesh_to(geo_map, vertices, cells, face_markers=single_face_marker + offset)

    expected_geojson = {
        "features": [{
            "geometry": {
                "coordinates": [
                    [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
                    [[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]],
                "type": "MultiPolygon"
            },
            "properties": {
                "color": "black",
                "fill": False,
                "fillColor": None,
                "fillOpacity": None,
                "stroke": True,
                "weight": 1
            },
            "type": "Feature"
        }],
        "type": "FeatureCollection"
    }
    expected_geojson = folium.utilities.normalize(json.dumps(expected_geojson))
    assert expected_geojson in folium.utilities.normalize(geo_map._parent.render())


@pytest.mark.parametrize("offset", (0, 1))
def test_base_mesh_plotter_add_mesh_to_map_vertices_cells_single_face_marker_color_weight(
    vertices: npt.NDArray[np.float64], cells: npt.NDArray[np.int64],
    single_face_marker: npt.NDArray[np.int64], offset: int
) -> None:
    """
    Test femlium.BaseMeshPlotter.add_mesh_to providing vertices, cells, a single face marker, uniform color & weight.

    The face marker value is equal to the default marker if the parameter offset is 0, while it is different
    from the default marker if the parameter offset is 1.
    """
    geo_map = folium.Map(location=[0, 0], zoom_start=8)
    mesh_plotter = femlium.BaseMeshPlotter()
    mesh_plotter.add_mesh_to(
        geo_map, vertices, cells, face_markers=single_face_marker + offset, face_colors="blue", face_weights=3)

    expected_geojson = {
        "features": [{
            "geometry": {
                "coordinates": [
                    [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
                    [[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]],
                "type": "MultiPolygon"
            },
            "properties": {
                "color": "blue",
                "fill": False,
                "fillColor": None,
                "fillOpacity": None,
                "stroke": True,
                "weight": 3
            },
            "type": "Feature"
        }],
        "type": "FeatureCollection"
    }
    expected_geojson = folium.utilities.normalize(json.dumps(expected_geojson))
    assert expected_geojson in folium.utilities.normalize(geo_map._parent.render())


def test_base_mesh_plotter_add_mesh_to_map_vertices_cells_multiple_face_markers(
    vertices: npt.NDArray[np.float64], cells: npt.NDArray[np.int64],
    multiple_face_markers: npt.NDArray[np.int64]
) -> None:
    """
    Test femlium.BaseMeshPlotter.add_mesh_to providing vertices, cells, where faces have different markers.

    If all faces of a cell have the same marker, then they will be colored through the cell polygon properties
    (stroke property equal to true).
    If there are at least two faces with different markers, then cell polygon properties will not color faces
    (stroke property equal to false) because different faces might need different colors, and its faces
    are represented as standalone segments.
    Note that this is not really needed here, because all face markers would be represented anyway by
    the same face color.
    """
    geo_map = folium.Map(location=[0, 0], zoom_start=8)
    mesh_plotter = femlium.BaseMeshPlotter()
    mesh_plotter.add_mesh_to(geo_map, vertices, cells, face_markers=multiple_face_markers)

    expected_geojson = {
        "features": [{
            "geometry": {
                "coordinates": [[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]],
                "type": "MultiPolygon"
            },
            "id": "0",
            "properties": {
                "color": "black",
                "fill": False,
                "fillColor": None,
                "fillOpacity": None,
                "stroke": True,
                "weight": 1
            },
            "type": "Feature"
        }, {
            "geometry": {
                "coordinates": [[[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]],
                "type": "MultiPolygon"
            },
            "id": "1",
            "properties": {
                "color": None,
                "fill": False,
                "fillColor": None,
                "fillOpacity": None,
                "stroke": False,
                "weight": None
            },
            "type": "Feature"
        }, {
            "geometry": {
                "coordinates": [[[0.0, 0.0], [1.0, 1.0]]],
                "type": "MultiLineString"
            },
            "id": "2",
            "properties": {
                "color": "black",
                "stroke": True,
                "weight": 1
            },
            "type": "Feature"
        }, {
            "geometry": {
                "coordinates": [[[1.0, 1.0], [0.0, 1.0]]],
                "type": "MultiLineString"
            },
            "id": "3",
            "properties": {
                "color": "black",
                "stroke": True,
                "weight": 1
            },
            "type": "Feature"
        }, {
            "geometry": {
                "coordinates": [[[0.0, 0.0], [0.0, 1.0]]],
                "type": "MultiLineString"
            },
            "id": "4",
            "properties": {
                "color": "black",
                "stroke": True,
                "weight": 1
            },
            "type": "Feature"
        }],
        "type": "FeatureCollection"
    }
    expected_geojson = folium.utilities.normalize(json.dumps(expected_geojson))
    assert expected_geojson in folium.utilities.normalize(geo_map._parent.render())


def test_base_mesh_plotter_add_mesh_to_map_vertices_cells_multiple_face_markers_colors_weights(
    vertices: npt.NDArray[np.float64], cells: npt.NDArray[np.int64],
    multiple_face_markers: npt.NDArray[np.int64]
) -> None:
    """
    Test femlium.BaseMeshPlotter.add_mesh_to providing vertices, cells, with multiple face markers & colors & weights.

    If all faces of a cell have the same marker, then they will be colored through the cell polygon properties
    (stroke property equal to true).
    If there are at least two faces with different markers, then cell polygon properties will not color faces
    (stroke property equal to false) because different faces might need different colors, and its faces
    are represented as standalone segments.
    In contrast to the previous test, this is actually needed here because we have different colors
    associated to different markers.
    """
    geo_map = folium.Map(location=[0, 0], zoom_start=8)
    mesh_plotter = femlium.BaseMeshPlotter()
    face_colors = {
        1: "blue",
        2: "red",
        3: "green"
    }
    face_weights = {
        1: 2,
        2: 3,
        3: 4
    }
    mesh_plotter.add_mesh_to(
        geo_map, vertices, cells, face_markers=multiple_face_markers,
        face_colors=face_colors, face_weights=face_weights)

    expected_geojson = {
        "features": [{
            "geometry": {
                "coordinates": [[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]],
                "type": "MultiPolygon"
            },
            "id": "0",
            "properties": {
                "color": "green",
                "fill": False,
                "fillColor": None,
                "fillOpacity": None,
                "stroke": True,
                "weight": 4
            },
            "type": "Feature"
        }, {
            "geometry": {
                "coordinates": [[[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]],
                "type": "MultiPolygon"
            },
            "id": "1",
            "properties": {
                "color": None,
                "fill": False,
                "fillColor": None,
                "fillOpacity": None,
                "stroke": False,
                "weight": None
            },
            "type": "Feature"
        }, {
            "geometry": {
                "coordinates": [[[0.0, 0.0], [1.0, 1.0]]],
                "type": "MultiLineString"
            },
            "id": "2",
            "properties": {
                "color": "green",
                "stroke": True,
                "weight": 4
            },
            "type": "Feature"
        }, {
            "geometry": {
                "coordinates": [[[1.0, 1.0], [0.0, 1.0]]],
                "type": "MultiLineString"
            },
            "id": "3",
            "properties": {
                "color": "blue",
                "stroke": True,
                "weight": 2
            },
            "type": "Feature"
        }, {
            "geometry": {
                "coordinates": [[[0.0, 0.0], [0.0, 1.0]]],
                "type": "MultiLineString"
            },
            "id": "4",
            "properties": {
                "color": "red",
                "stroke": True,
                "weight": 3
            },
            "type": "Feature"
        }],
        "type": "FeatureCollection"
    }
    expected_geojson = folium.utilities.normalize(json.dumps(expected_geojson))
    geo_map_html = folium.utilities.normalize(geo_map._parent.render())
    assert expected_geojson in geo_map_html

    # Confirm presence of color bar
    assert "d3.scale.linear().domain([1.0,3.0])" in geo_map_html


def test_base_mesh_plotter_add_mesh_to_map_vertices_cells_multiple_face_markers_missing_colors_weights(
    vertices: npt.NDArray[np.float64], cells: npt.NDArray[np.int64],
    multiple_face_markers: npt.NDArray[np.int64]
) -> None:
    """
    Test femlium.BaseMeshPlotter.add_mesh_to providing vertices, cells, with multiple face markers with missing values.

    The provided colors and weights do not map all the available marker values, and thus some values are missing.
    In this case, any unmapped marker value will revert to default color and weight.
    """
    geo_map = folium.Map(location=[0, 0], zoom_start=8)
    mesh_plotter = femlium.BaseMeshPlotter()
    face_colors = {
        1: "blue",
        2: "red"
    }
    face_weights = {
        1: 2,
        3: 4
    }
    mesh_plotter.add_mesh_to(
        geo_map, vertices, cells, face_markers=multiple_face_markers,
        face_colors=face_colors, face_weights=face_weights)

    expected_geojson = {
        "features": [{
            "geometry": {
                "coordinates": [[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]],
                "type": "MultiPolygon"
            },
            "id": "0",
            "properties": {
                "color": "black",
                "fill": False,
                "fillColor": None,
                "fillOpacity": None,
                "stroke": True,
                "weight": 4
            },
            "type": "Feature"
        }, {
            "geometry": {
                "coordinates": [[[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]],
                "type": "MultiPolygon"
            },
            "id": "1",
            "properties": {
                "color": None,
                "fill": False,
                "fillColor": None,
                "fillOpacity": None,
                "stroke": False,
                "weight": None
            },
            "type": "Feature"
        }, {
            "geometry": {
                "coordinates": [[[0.0, 0.0], [1.0, 1.0]]],
                "type": "MultiLineString"
            },
            "id": "2",
            "properties": {
                "color": "black",
                "stroke": True,
                "weight": 4
            },
            "type": "Feature"
        }, {
            "geometry": {
                "coordinates": [[[1.0, 1.0], [0.0, 1.0]]],
                "type": "MultiLineString"
            },
            "id": "3",
            "properties": {
                "color": "blue",
                "stroke": True,
                "weight": 2
            },
            "type": "Feature"
        }, {
            "geometry": {
                "coordinates": [[[0.0, 0.0], [0.0, 1.0]]],
                "type": "MultiLineString"
            },
            "id": "4",
            "properties": {
                "color": "red",
                "stroke": True,
                "weight": 1
            },
            "type": "Feature"
        }],
        "type": "FeatureCollection"
    }
    expected_geojson = folium.utilities.normalize(json.dumps(expected_geojson))
    geo_map_html = folium.utilities.normalize(geo_map._parent.render())
    assert expected_geojson in geo_map_html

    # Confirm presence of color bar
    assert "d3.scale.linear().domain([1.0,3.0])" in geo_map_html


def test_base_mesh_plotter_add_mesh_to_map_vertices_cells_single_cell_color(
    vertices: npt.NDArray[np.float64], cells: npt.NDArray[np.int64]
) -> None:
    """Test femlium.BaseMeshPlotter.add_mesh_to providing vertices, cells, and a uniform cell color."""
    geo_map = folium.Map(location=[0, 0], zoom_start=8)
    mesh_plotter = femlium.BaseMeshPlotter()
    mesh_plotter.add_mesh_to(geo_map, vertices, cells, cell_colors="orange")

    expected_geojson = {
        "features": [{
            "geometry": {
                "coordinates": [
                    [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
                    [[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]],
                "type": "MultiPolygon"
            },
            "properties": {
                "color": "black",
                "fill": True,
                "fillColor": "orange",
                "fillOpacity": 1,
                "stroke": True,
                "weight": 1
            },
            "type": "Feature"
        }],
        "type": "FeatureCollection"
    }
    expected_geojson = folium.utilities.normalize(json.dumps(expected_geojson))
    assert expected_geojson in folium.utilities.normalize(geo_map._parent.render())


@pytest.mark.parametrize("offset", (0, 1))
def test_base_mesh_plotter_add_mesh_to_map_vertices_cells_single_cell_marker(
    vertices: npt.NDArray[np.float64], cells: npt.NDArray[np.int64],
    single_cell_marker: npt.NDArray[np.int64], offset: int
) -> None:
    """
    Test femlium.BaseMeshPlotter.add_mesh_to providing vertices, cells, and a single marker for all cells.

    The cell marker value is equal to the default marker if the parameter offset is 0, while it is different
    from the default marker if the parameter offset is 1.
    The default cell color will be used in both cases, which corresponds to not coloring the cell at all.
    """
    geo_map = folium.Map(location=[0, 0], zoom_start=8)
    mesh_plotter = femlium.BaseMeshPlotter()
    mesh_plotter.add_mesh_to(geo_map, vertices, cells, cell_markers=single_cell_marker + offset)

    expected_geojson = {
        "features": [{
            "geometry": {
                "coordinates": [
                    [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
                    [[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]],
                "type": "MultiPolygon"
            },
            "properties": {
                "color": "black",
                "fill": False,
                "fillColor": None,
                "fillOpacity": None,
                "stroke": True,
                "weight": 1
            },
            "type": "Feature"
        }],
        "type": "FeatureCollection"
    }
    expected_geojson = folium.utilities.normalize(json.dumps(expected_geojson))
    assert expected_geojson in folium.utilities.normalize(geo_map._parent.render())


@pytest.mark.parametrize("offset", (0, 1))
def test_base_mesh_plotter_add_mesh_to_map_vertices_cells_single_cell_marker_color(
    vertices: npt.NDArray[np.float64], cells: npt.NDArray[np.int64],
    single_cell_marker: npt.NDArray[np.int64], offset: int
) -> None:
    """
    Test femlium.BaseMeshPlotter.add_mesh_to providing vertices, cells, and a single marker for all cells.

    The cell marker value is equal to the default marker if the parameter offset is 0, while it is different
    from the default marker if the parameter offset is 1.
    """
    geo_map = folium.Map(location=[0, 0], zoom_start=8)
    mesh_plotter = femlium.BaseMeshPlotter()
    mesh_plotter.add_mesh_to(
        geo_map, vertices, cells, cell_markers=single_cell_marker + offset, cell_colors="orange")

    expected_geojson = {
        "features": [{
            "geometry": {
                "coordinates": [
                    [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
                    [[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]],
                "type": "MultiPolygon"
            },
            "properties": {
                "color": "black",
                "fill": True,
                "fillColor": "orange",
                "fillOpacity": 1,
                "stroke": True,
                "weight": 1
            },
            "type": "Feature"
        }],
        "type": "FeatureCollection"
    }
    expected_geojson = folium.utilities.normalize(json.dumps(expected_geojson))
    assert expected_geojson in folium.utilities.normalize(geo_map._parent.render())


def test_base_mesh_plotter_add_mesh_to_map_vertices_cells_multiple_cell_markers(
    vertices: npt.NDArray[np.float64], cells: npt.NDArray[np.int64],
    multiple_cell_markers: npt.NDArray[np.int64]
) -> None:
    """
    Test femlium.BaseMeshPlotter.add_mesh_to providing vertices, cells, where cells have different markers.

    The default cell color will be used in all cases, which corresponds to not coloring the cell at all.
    """
    geo_map = folium.Map(location=[0, 0], zoom_start=8)
    mesh_plotter = femlium.BaseMeshPlotter()
    mesh_plotter.add_mesh_to(geo_map, vertices, cells, cell_markers=multiple_cell_markers)

    expected_geojson = {
        "features": [{
            "geometry": {
                "coordinates": [[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]],
                "type": "MultiPolygon"
            },
            "id": "0",
            "properties": {
                "color": "black",
                "fill": False,
                "fillColor": None,
                "fillOpacity": None,
                "stroke": True,
                "weight": 1
            },
            "type": "Feature"
        }, {
            "geometry": {
                "coordinates": [[[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]],
                "type": "MultiPolygon"
            },
            "id": "1",
            "properties": {
                "color": "black",
                "fill": False,
                "fillColor": None,
                "fillOpacity": None,
                "stroke": True,
                "weight": 1
            },
            "type": "Feature"
        }],
        "type": "FeatureCollection"
    }
    expected_geojson = folium.utilities.normalize(json.dumps(expected_geojson))
    assert expected_geojson in folium.utilities.normalize(geo_map._parent.render())


def test_base_mesh_plotter_add_mesh_to_map_vertices_cells_multiple_cell_markers_colors(
    vertices: npt.NDArray[np.float64], cells: npt.NDArray[np.int64],
    multiple_cell_markers: npt.NDArray[np.int64]
) -> None:
    """Test femlium.BaseMeshPlotter.add_mesh_to providing vertices, cells, with multiple cell markers & colors."""
    geo_map = folium.Map(location=[0, 0], zoom_start=8)
    mesh_plotter = femlium.BaseMeshPlotter()
    cell_colors = {
        1: "blue",
        2: "red"
    }
    mesh_plotter.add_mesh_to(
        geo_map, vertices, cells, cell_markers=multiple_cell_markers, cell_colors=cell_colors)

    expected_geojson = {
        "features": [{
            "geometry": {
                "coordinates": [[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]],
                "type": "MultiPolygon"
            },
            "properties": {
                "color": "black",
                "fill": True,
                "fillColor": "blue",
                "fillOpacity": 1,
                "stroke": True,
                "weight": 1
            },
            "type": "Feature"
        }, {
            "geometry": {
                "coordinates": [[[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]],
                "type": "MultiPolygon"
            },
            "properties": {
                "color": "black",
                "fill": True,
                "fillColor": "red",
                "fillOpacity": 1,
                "stroke": True,
                "weight": 1
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


def test_base_mesh_plotter_add_mesh_to_map_vertices_cells_multiple_cell_markers_missing_colors(
    vertices: npt.NDArray[np.float64], cells: npt.NDArray[np.int64],
    multiple_cell_markers: npt.NDArray[np.int64]
) -> None:
    """
    Test femlium.BaseMeshPlotter.add_mesh_to providing vertices, cells, with multiple cell markers with missing values.

    The provided colors do not map all the available marker values.
    In this case, any unmapped marker value will revert to default color, i.e. to uncolored cells.
    """
    geo_map = folium.Map(location=[0, 0], zoom_start=8)
    mesh_plotter = femlium.BaseMeshPlotter()
    cell_colors = {
        1: "blue"
    }
    mesh_plotter.add_mesh_to(
        geo_map, vertices, cells, cell_markers=multiple_cell_markers, cell_colors=cell_colors)

    expected_geojson = {
        "features": [{
            "geometry": {
                "coordinates": [[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]],
                "type": "MultiPolygon"
            },
            "properties": {
                "color": "black",
                "fill": True,
                "fillColor": "blue",
                "fillOpacity": 1,
                "stroke": True,
                "weight": 1
            },
            "type": "Feature"
        }, {
            "geometry": {
                "coordinates": [[[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]],
                "type": "MultiPolygon"
            },
            "properties": {
                "color": "black",
                "fill": False,
                "fillColor": None,
                "fillOpacity": None,
                "stroke": True,
                "weight": 1
            },
            "type": "Feature"
        }],
        "type": "FeatureCollection"
    }
    expected_geojson = folium.utilities.normalize(json.dumps(expected_geojson))
    geo_map_html = folium.utilities.normalize(geo_map._parent.render())
    assert expected_geojson in geo_map_html

    # Confirm absence of color bar
    assert "d3.scale.linear()" not in geo_map_html
