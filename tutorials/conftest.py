# Copyright (C) 2021-2022 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT
"""pytest configuration file for tutorials tests."""

import fnmatch
import os
import shutil

import _pytest
import nbvalx.pytest_hooks_notebooks
import pytest

pytest_addoption = nbvalx.pytest_hooks_notebooks.addoption
pytest_sessionstart = nbvalx.pytest_hooks_notebooks.sessionstart
pytest_collect_file = nbvalx.pytest_hooks_notebooks.collect_file
pytest_runtest_makereport = nbvalx.pytest_hooks_notebooks.runtest_makereport
pytest_runtest_teardown = nbvalx.pytest_hooks_notebooks.runtest_teardown


def pytest_runtest_setup(item: pytest.File) -> None:
    """Copy data files to destination folder, and check backend availability."""
    # Do the setup as in nbvalx
    nbvalx.pytest_hooks_notebooks.runtest_setup(item)
    # Get notebook name
    notebook_name = item.parent.name
    # Copy data files
    if item.name == "Cell 0":
        work_dir = item.parent.config.option.work_dir
        notebook_original_dir = os.path.dirname(notebook_name).replace(work_dir, "")
        for dir_entry in _pytest.pathlib.visit(notebook_original_dir, lambda _: True):
            if dir_entry.is_file():
                source_path = str(dir_entry.path)
                if fnmatch.fnmatch(source_path, "**/*.csv") or fnmatch.fnmatch(source_path, "**/*.msh"):
                    destination_path = os.path.join(
                        notebook_original_dir, work_dir, os.path.relpath(source_path, notebook_original_dir))
                    if not os.path.exists(destination_path):
                        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                        shutil.copyfile(source_path, destination_path)
    # Check backend availability depending on the item name
    if notebook_name.endswith("dolfin.ipynb"):
        pytest.importorskip("dolfin")
    elif notebook_name.endswith("dolfinx.ipynb"):
        pytest.importorskip("dolfinx")
    elif notebook_name.endswith("firedrake.ipynb"):
        pytest.importorskip("firedrake")
    elif notebook_name.endswith("meshio.ipynb"):
        pytest.importorskip("meshio")
    elif notebook_name.endswith("generate_mesh.ipynb"):
        pass
    else:
        raise ValueError("Invalid notebook name " + notebook_name)
