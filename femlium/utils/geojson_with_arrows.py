# Copyright (C) 2021-2022 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT

import json
from folium.elements import JSCSSMixin
from folium.features import GeoJson
from folium.utilities import parse_options


class GeoJsonWithArrows(JSCSSMixin, GeoJson):
    """
    Creates a GeoJson object for plotting into a Map. The GeoJson object is patched
    to add arrows from the leaflet-arrowheads project.

    Parameters
    ----------
    data: file, dict or str.
        The GeoJSON data you want to plot.
    yawn: int, default 60
        Defines the width of the opening of the arrowhead, given in degrees.
    size: number or str, default "15%"
        Determines the size of the arrowhead.
    frequency: str, default "allvertices"
        How many arrowheads are rendered on a polyline.
    proportionalToTotal: bool, default False
        If True, render the arrowhead(s) with a size proportional to the entire length of the
        multi-segmented polyline. If False, size is proportional to the average length of all the segments.
    **kwargs
        Remaining keyword arguments for the initialization of the standard GeoJson instance.

    See https://github.com/slutske22/leaflet-arrowheads for more information.
    """

    default_js = [
        ("geometryutil",
         "https://cdn.jsdelivr.net/npm/leaflet-arrowheads@1.2.2/src/leaflet-geometryutil.js"),
        ("arrowheads",
         "https://cdn.jsdelivr.net/npm/leaflet-arrowheads@1.2.2/src/leaflet-arrowheads.js")
    ]

    def __init__(self, data, yawn=60, size="15%", frequency="allvertices", proportionalToTotal=False, **kwargs):
        super(GeoJsonWithArrows, self).__init__(data, **kwargs)
        self._name = 'GeoJsonWithArrows'
        self.arrows_options = parse_options(
            yawn=yawn,
            size=size,
            frequency=frequency,
            proportionalToTotal=proportionalToTotal
        )

    def render(self, **kwargs):
        """
        Renders the GeoJson object, adding arrowheads options among L.geoJson properties.
        """
        original_script = self._template.module.__dict__.get("script", None)
        assert original_script is not None
        patched_script = PatchedScript(original_script, self.arrows_options)
        self._template.module.__dict__["script"] = patched_script
        super(GeoJsonWithArrows, self).render(**kwargs)


class PatchedScript(object):
    """
    Patch a jinja2.runtime.Macro object associated to the script part of a jinja2 template
    to add arrowheads to the geojson script.
    """

    def __init__(self, original_script, options):
        self.original_script = original_script
        self.options = options

    def __call__(self, *args, **kwargs):
        output = self.original_script(*args, **kwargs)
        assert "L.geoJson(null, {" in output
        if "arrowheads:" not in output:
            output = output.replace(
                "L.geoJson(null, {",
                """L.geoJson(null, {
                    arrowheads: """ + json.dumps(self.options) + ""","""
            )
        return output
