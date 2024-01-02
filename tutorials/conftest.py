# Copyright (C) 2021-2024 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT
"""pytest configuration file for tutorials tests."""

import nbvalx.pytest_hooks_notebooks
import pytest

pytest_addoption = nbvalx.pytest_hooks_notebooks.addoption
pytest_sessionstart = nbvalx.pytest_hooks_notebooks.sessionstart
pytest_collect_file = nbvalx.pytest_hooks_notebooks.collect_file
pytest_runtest_makereport = nbvalx.pytest_hooks_notebooks.runtest_makereport
pytest_runtest_teardown = nbvalx.pytest_hooks_notebooks.runtest_teardown


def pytest_sessionstart(session: pytest.Session) -> None:
    """Automatically mark mesh files as data to be copied to the work directory."""
    # Add mesh files as data to be copied
    copy_data_to_work_dir = session.config.option.copy_data_to_work_dir
    assert len(copy_data_to_work_dir) == 0
    copy_data_to_work_dir.extend(["**/*.csv", "**/*.msh"])
    # Start session as in nbvalx
    nbvalx.pytest_hooks_notebooks.sessionstart(session)


def pytest_runtest_setup(item: pytest.File) -> None:
    """Copy data files to destination folder, and check backend availability."""
    # Do the setup as in nbvalx
    nbvalx.pytest_hooks_notebooks.runtest_setup(item)
    # Get notebook name
    notebook_name = item.parent.name
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
