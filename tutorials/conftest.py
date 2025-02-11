# Copyright (C) 2021-2025 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT
"""pytest configuration file for tutorials tests."""

import nbvalx.pytest_hooks_notebooks
import pytest

pytest_addoption = nbvalx.pytest_hooks_notebooks.addoption
pytest_collect_file = nbvalx.pytest_hooks_notebooks.collect_file


def pytest_sessionstart(session: pytest.Session) -> None:
    """Automatically mark mesh files as data to be linked in the work directory."""
    # Add mesh files as data to be linked
    link_data_in_work_dir = session.config.option.link_data_in_work_dir
    assert len(link_data_in_work_dir) == 0
    link_data_in_work_dir.extend(["**/*.csv", "**/*.msh"])
    # Start session as in nbvalx
    nbvalx.pytest_hooks_notebooks.sessionstart(session)


def pytest_runtest_setup(item: pytest.File) -> None:
    """Check backend availability."""
    # Get notebook name
    notebook_name = item.parent.name
    # Check backend availability depending on the item name
    if notebook_name.endswith("dolfinx.ipynb"):
        pytest.importorskip("dolfinx")
    elif notebook_name.endswith("firedrake.ipynb"):
        pytest.importorskip("firedrake")
    elif notebook_name.endswith("meshio.ipynb"):
        pytest.importorskip("meshio")
    elif notebook_name.endswith("generate_mesh.ipynb"):
        pass
    else:
        raise ValueError("Invalid notebook name " + notebook_name)
