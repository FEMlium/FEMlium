# Copyright (C) 2021 by the FEMlium authors
#
# This file is part of FEMlium.
#
# SPDX-License-Identifier: MIT

import os
import re
import sys
import importlib
import pytest
import pytest_flake8
import matplotlib.pyplot as plt
from nbconvert.exporters import PythonExporter
import nbconvert.filters
plt.switch_backend("Agg")


def pytest_ignore_collect(path, config):
    if path.ext == ".py" and path.new(ext=".ipynb").exists():  # ignore .py files obtained from previous runs
        return True
    else:
        return False


def pytest_collect_file(path, parent):
    """
    Collect tutorial files.
    """

    if path.ext == ".ipynb":
        # Convert .ipynb notebooks to plain .py files
        def comment_lines(text, prefix="# "):
            regex = re.compile(r".{1,80}(?:\s+|$)")
            input_lines = text.split("\n")
            output_lines = [split_line.rstrip() for line in input_lines for split_line in regex.findall(line)]
            output = prefix + ("\n" + prefix).join(output_lines)
            return output.replace(prefix + "\n", prefix.rstrip(" ") + "\n")

        def ipython2python(code):
            return nbconvert.filters.ipython2python(code).rstrip("\n") + "\n"

        filters = {
            "comment_lines": comment_lines,
            "ipython2python": ipython2python
        }
        exporter = PythonExporter(filters=filters)
        exporter.exclude_input_prompt = True
        code, _ = exporter.from_filename(path)
        code = code.rstrip("\n") + "\n"
        with open(path.new(ext=".py"), "w", encoding="utf-8") as f:
            f.write(code)
        # Collect the corresponding .py file
        config = parent.config
        if config.getoption("--flake8"):
            return pytest_flake8.pytest_collect_file(path.new(ext=".py"), parent)
        else:
            if not path.basename.startswith("x"):
                return TutorialFile.from_parent(parent=parent, fspath=path.new(ext=".py"))
            else:
                return DoNothingFile.from_parent(parent=parent, fspath=path.new(ext=".py"))
    elif path.ext == ".py":
        assert not path.new(ext=".ipynb").exists(), "Please run pytest on jupyter notebooks, not plain python files."
        return DoNothingFile.from_parent(parent=parent, fspath=path)


def pytest_pycollect_makemodule(path, parent):
    """
    Disable running .py files produced by previous runs, as they may get out of sync with the corresponding .ipynb file.
    """

    if path.ext == ".py":
        assert not path.new(ext=".ipynb").exists(), "Please run pytest on jupyter notebooks, not plain python files."
        return DoNothingFile.from_parent(parent=parent, fspath=path)


class TutorialFile(pytest.File):
    """
    Custom file handler for tutorial files.
    """

    def collect(self):
        yield TutorialItem.from_parent(
            parent=self, name="run_tutorial -> " + os.path.relpath(str(self.fspath), str(self.parent.fspath)))


class TutorialItem(pytest.Item):
    """
    Handle the execution of the tutorial.
    """

    def __init__(self, name, parent):
        super(TutorialItem, self).__init__(name, parent)

    def runtest(self):
        self._import_backend_or_skip()
        os.chdir(self.parent.fspath.dirname)
        sys.path.append(self.parent.fspath.dirname)
        spec = importlib.util.spec_from_file_location(self.name, str(self.parent.fspath))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        plt.close("all")  # do not trigger matplotlib max_open_warning

    def _import_backend_or_skip(self):
        if self.name.endswith("dolfin.py"):
            pytest.importorskip("dolfin")
        if self.name.endswith("dolfinx.py"):
            pytest.importorskip("dolfinx")
        elif self.name.endswith("firedrake.py"):
            pytest.importorskip("firedrake")
        elif self.name.endswith("meshio.py"):
            pytest.importorskip("meshio")

    def reportinfo(self):
        return self.fspath, 0, self.name


class DoNothingFile(pytest.File):
    """
    Custom file handler to avoid running twice python files explicitly provided on the command line.
    """

    def collect(self):
        return []
