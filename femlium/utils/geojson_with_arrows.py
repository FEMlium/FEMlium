# Copyright (C) 2021-2023 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT
"""A GeoJson object with arrows for plotting into a Map."""

import json
import typing

import folium.elements
import folium.features
import folium.utilities


class GeoJsonWithArrows(folium.elements.JSCSSMixin, folium.features.GeoJson):
    """
    A GeoJson object with arrows for plotting into a Map.

    The GeoJson object is patched to add arrows from the leaflet-arrowheads project.

    Parameters
    ----------
    data
        The GeoJSON data you want to plot.
    yawn
        Defines the width of the opening of the arrowhead, given in degrees.
    size
        Determines the size of the arrowhead.
    frequency
        How many arrowheads are rendered on a polyline.
    proportionalToTotal
        If True, render the arrowhead(s) with a size proportional to the entire length of the
        multi-segmented polyline. If False, size is proportional to the average length of all the segments.
    **kwargs
        Remaining keyword arguments for the initialization of the standard GeoJson instance.

    See https://github.com/slutske22/leaflet-arrowheads for more information.
    """

    default_js: typing.ClassVar[typing.List[typing.Tuple[str, str]]] = [
        ("geometryutil",
         "https://cdn.jsdelivr.net/npm/leaflet-arrowheads@1.2.2/src/leaflet-geometryutil.js"),
        ("arrowheads",
         "https://cdn.jsdelivr.net/npm/leaflet-arrowheads@1.2.2/src/leaflet-arrowheads.js")
    ]

    def __init__(
        self, data: typing.Any,  # noqa: ANN401
        yawn: typing.Optional[int] = 60, size: typing.Optional[str] = "15%",
        frequency: typing.Optional[str] = "allvertices",
        proportionalToTotal: typing.Optional[bool] = False,   # noqa: N803
        **kwargs: typing.Any  # noqa: ANN401
    ) -> None:
        super(GeoJsonWithArrows, self).__init__(data, **kwargs)
        self._name = "GeoJsonWithArrows"
        self.arrows_options = folium.utilities.parse_options(
            yawn=yawn,
            size=size,
            frequency=frequency,
            proportionalToTotal=proportionalToTotal
        )

    def render(self, **kwargs: typing.Any) -> None:  # noqa: ANN401
        """Render the GeoJson object, adding arrowheads options among L.geoJson properties."""
        original_script = self._template.module.__dict__.get("script", None)
        assert original_script is not None
        patched_script = PatchedScript(original_script, self.arrows_options)
        self._template.module.__dict__["script"] = patched_script
        super(GeoJsonWithArrows, self).render(**kwargs)


class PatchedScript(object):
    """Patch a jinja2.runtime.Macro object associated to the script part of a jinja2 template to add arrowheads."""

    def __init__(self, original_script: typing.Callable[..., str], options: typing.Dict[str, typing.Any]) -> None:
        self.original_script = original_script
        self.options = options

    def __call__(self, *args: typing.Any, **kwargs: typing.Any) -> str:  # noqa: ANN401
        """Apply patch."""
        output = self.original_script(*args, **kwargs)
        assert "L.geoJson(null, {" in output
        if "arrowheads:" not in output:
            output = output.replace(
                "L.geoJson(null, {",
                """L.geoJson(null, {
                    arrowheads: """ + json.dumps(self.options) + ""","""
            )
        return output
