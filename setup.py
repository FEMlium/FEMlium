# Copyright (C) 2021 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT

from setuptools import find_packages, setup

setup(name="FEMlium",
      description="Interactive geographic plots of finite element data with folium",
      long_description="Interactive geographic plots of finite element data with folium",
      author="Francesco Ballarin (and contributors)",
      author_email="francesco.ballarin@unicatt.it",
      version="0.0.dev1",
      license="MIT License",
      url="https://github.com/femlium/felium",
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "License :: OSI Approved :: MIT License",
          "Topic :: Scientific/Engineering :: GIS",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Visualization",
          "Topic :: Software Development :: Libraries :: Python Modules",
      ],
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          "branca",
          "folium",
          "geojson",
          "matplotlib",
          "numpy",
          "pytest-runner"
      ],
      tests_require=[
          "pyproj",
          "pytest",
          "pytest-flake8"
      ]
      )
