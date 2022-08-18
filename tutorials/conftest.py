# Copyright (C) 2021-2022 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT

import os
import re
import sys
import pytest
import nbformat
from nbconvert.exporters import PythonExporter
from nbconvert.preprocessors import ExecutePreprocessor
import nbconvert.filters


def pytest_ignore_collect(collection_path, path, config):
    if collection_path.suffix == ".py" and collection_path.with_suffix(".ipynb").exists():
        # ignore .py files obtained from previous runs
        return True
    else:
        return False


def pytest_collect_file(file_path, path, parent):
    """
    Collect tutorial files.
    """

    if file_path.suffix == ".ipynb":
        if not file_path.name.startswith("x"):
            return TutorialFile.from_parent(parent=parent, path=file_path)
        else:
            return DoNothingFile.from_parent(parent=parent, path=file_path)
    elif file_path.suffix == ".py":
        assert not file_path.with_suffix(".ipynb").exists(), (
            "Please run pytest on jupyter notebooks, not plain python files.")
        return DoNothingFile.from_parent(parent=parent, path=file_path)


def pytest_pycollect_makemodule(module_path, path, parent):
    """
    Disable running .py files produced by previous runs, as they may get out of sync with the corresponding .ipynb file.
    """

    if module_path.suffix == ".py":
        assert not module_path.with_suffix(".ipynb").exists(), (
            "Please run pytest on jupyter notebooks, not plain python files.")
        return DoNothingFile.from_parent(parent=parent, path=module_path)


class TutorialFile(pytest.File):
    """
    Custom file handler for tutorial files.
    """

    def collect(self):
        yield TutorialItem.from_parent(
            parent=self, name="run_tutorial -> " + os.path.relpath(str(self.path), str(self.parent.path)))


class TutorialItem(pytest.Item):
    """
    Handle the execution of the tutorial.
    """

    def __init__(self, name, parent):
        super(TutorialItem, self).__init__(name, parent)

    def runtest(self):
        self._import_backend_or_skip()
        os.chdir(self.parent.path.parent)
        sys.path.append(str(self.parent.path.parent))
        with open(self.parent.path) as f:
            nb = nbformat.read(f, as_version=4)
        execute_preprocessor = ExecutePreprocessor()
        try:
            execute_preprocessor.preprocess(nb)
        finally:
            with open(self.parent.path, "w") as f:
                nbformat.write(nb, f)

    def _import_backend_or_skip(self):
        if self.name.endswith("dolfin.ipynb"):
            pytest.importorskip("dolfin")
        elif self.name.endswith("dolfinx.ipynb"):
            pytest.importorskip("dolfinx")
        elif self.name.endswith("firedrake.ipynb"):
            pytest.importorskip("firedrake")
        elif self.name.endswith("meshio.ipynb"):
            pytest.importorskip("meshio")
        elif self.name.endswith("generate_mesh.ipynb"):
            pass
        else:
            raise ValueError("Invalid name " + self.name)

    def reportinfo(self):
        return self.path, 0, self.name


class DoNothingFile(pytest.File):
    """
    Custom file handler to avoid running twice python files explicitly provided on the command line.
    """

    def collect(self):
        return []
