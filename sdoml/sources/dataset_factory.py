# The contents of this file were inspired by the SunPy library:
# Project     : https://github.com/sunpy/sunpy/
# File        : sunpy/map/map_factory.py
# Commit hash : 18658e45868a97f80a17ba2dcf411c45fe4edec1

# ---
# SunPy is released under a BSD-style open source license:
#
# Copyright (c) 2013-2022 The SunPy developers
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

"""
Module for defining the framework around the Data factory.
"""

from sdoml.sources.sdoml_gcs import GenericData
from sunpy.util.datatype_factory_base import BasicRegistrationFactory
from sunpy.map.map_factory import (
    NoMatchError,
    MultipleMatchError,
)

__all__ = ["DataSourceFactory", "DataSource"]


class DataSourceFactory(BasicRegistrationFactory):
    """
    Data factory class. Used to create a variety of Data objects.
    Valid data structures are specified by registering them with the
    factory.
    """

    def __call__(self, instrument, meta, years, cache_size, **kwargs):
        """
        This function iterates over each registered type,
        checking to see if WidgetType matches the arguments.
        If the match exists, use that type. If there are 0 or >=1 matches,
        NoMatchError or MultipleMatchError will be raised.
        """
        new_data = []

        try:
            new_datum = self._check_registered_widgets(
                instrument, meta, years, cache_size, **kwargs
            )
            new_data.append(new_datum)
        except (NoMatchError, MultipleMatchError) as e:
            print(f"One of the data sources failed to validate with: {e}")

        if len(new_data) == 1:
            return new_data[0]

    def _check_registered_widgets(
        self, instrument, meta, years, cache_size, **kwargs
    ):
        """
        Implementation of a basic check to see if arguments match a widget.
        """

        # iterate through the registry
        candidate_widget_types = [
            key
            for key in self.registry
            if self.registry[key](instrument, meta, **kwargs)
        ]

        # ensure that only one match exists; throws errors otherwise
        num_matches = len(candidate_widget_types)

        if num_matches == 0:
            if self.default_widget_type is None:
                raise NoMatchError(
                    "No candidate types identified from the specified arguements, and no default is set"
                )
            else:
                # the default_widget_type is set when calling the DataSourceFactory
                candidate_widget_types = [self.default_widget_type]
        elif num_matches > 1:
            raise MultipleMatchError(
                f"Too many candidate types identified ({candidate_widget_types})"
            )

        # Only one is found
        WidgetType = candidate_widget_types[0]

        return WidgetType(instrument, meta, years, cache_size, **kwargs)


DataSource = DataSourceFactory(
    default_widget_type=None,  # !TODO consider using ``GenericData``
    registry=GenericData._registry,
    additional_validation_functions=["datasource"],
)
