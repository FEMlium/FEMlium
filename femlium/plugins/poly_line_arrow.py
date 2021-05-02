# Copyright (C) 2021 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT

from folium.elements import JSCSSMixin
from folium.features import MacroElement
from folium.utilities import parse_options

from jinja2 import Template


class PolyLineArrow(JSCSSMixin, MacroElement):
    """
    Add arrow heads along a PolyLine.

    Parameters
    ----------
    polyline: folium.features.PolyLine object
        The folium.features.PolyLine object to attach the arrows to.
    yawn: int, default 60
        Defines the width of the opening of the arrowhead, given in degrees.
    size: number or str, default "15%"
        Determines the size of the arrowhead.
    frequency: str, default "allvertices"
        How many arrowheads are rendered on a polyline.
    proportionalToTotal: bool, default False
        If True, render the arrowhead(s) with a size proportional to the entire length of the
        multi-segmented polyline. If False, size is proportional to the average length of all the segments.

    See https://github.com/slutske22/leaflet-arrowheads for more information.
    """

    _template = Template(u"""
        {% macro script(this, kwargs) %}
            {{ this.polyline.get_name() }}.arrowheads(
                {{ this.options|tojson }}
            );
        {% endmacro %}
        """)

    default_js = [
        ("geometryutil",
         "https://cdn.jsdelivr.net/npm/leaflet-arrowheads@1.2.2/src/leaflet-geometryutil.js"),
        ("arrowheads",
         "https://cdn.jsdelivr.net/npm/leaflet-arrowheads@1.2.2/src/leaflet-arrowheads.js")
    ]

    def __init__(self, polyline, yawn=60, size="15%", frequency="allvertices", proportionalToTotal=False, **kwargs):
        super(PolyLineArrow, self).__init__()
        self._name = 'PolyLineArrow'
        self.polyline = polyline
        self.options = parse_options(
            yawn=yawn,
            size=size,
            frequency=frequency,
            proportionalToTotal=proportionalToTotal,
            **kwargs
        )
