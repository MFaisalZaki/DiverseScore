"""The package's public surface: what a user gets from importing it."""

import re
from pathlib import Path

import pytest

import plandiversity
from plandiversity import metrics, models, solvers

METRIC_NAMES = ["Stability", "States", "Uniqueness"]
MODEL_NAMES = ["MaxMean", "MaxMin", "MaxSum"]
SOLVER_NAMES = ["ExactSolver", "GreedySolver"]


class TestImport:
    def test_the_package_imports_without_optional_build_tooling(self):
        """It used to import pkg_resources, which recent setuptools no longer
        ships, so `import plandiversity` failed outright on a clean 3.12+ env."""
        assert plandiversity.__version__

    def test_the_version_is_a_dotted_release(self):
        assert re.fullmatch(r"\d+\.\d+\.\d+", plandiversity.__version__)

    def test_the_declared_version_matches_pyproject(self):
        pyproject = Path(plandiversity.__file__).parent.parent / "pyproject.toml"
        declared = re.search(r'^version = "(.+)"$', pyproject.read_text(), re.MULTILINE)
        assert declared and declared.group(1) == plandiversity.__version__


class TestPublicNames:
    @pytest.mark.parametrize("name", METRIC_NAMES)
    def test_metrics_are_exported(self, name):
        assert name in metrics.__all__
        assert issubclass(getattr(metrics, name), metrics.Metric)

    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_models_are_exported(self, name):
        assert name in models.__all__
        assert issubclass(getattr(models, name), models.Model)

    @pytest.mark.parametrize("name", SOLVER_NAMES)
    def test_solvers_are_exported(self, name):
        assert name in solvers.__all__
        assert issubclass(getattr(solvers, name), solvers.Solver)

    @pytest.mark.parametrize("name", METRIC_NAMES + MODEL_NAMES + SOLVER_NAMES)
    def test_shortcuts_re_export_everything(self, name):
        from plandiversity import shortcuts

        assert name in shortcuts.__all__
        assert hasattr(shortcuts, name)

    def test_shortcuts_also_re_export_the_framework(self):
        """A script should need only this one star import."""
        namespace = {}
        exec("from plandiversity.shortcuts import *", namespace)  # noqa: S102
        for name in ("PDDLReader", "Problem", "Fluent", "InstantaneousAction"):
            assert name in namespace

    @pytest.mark.parametrize("name", METRIC_NAMES)
    def test_every_metric_names_itself(self, name):
        assert getattr(metrics, name).name == name

    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_every_model_names_itself(self, name):
        assert getattr(models, name).name == name


class TestModuleLayout:
    """The modules are snake_case; the classes they export are not."""

    @pytest.mark.parametrize(
        "module",
        [
            "plandiversity.metrics.base",
            "plandiversity.metrics.stability",
            "plandiversity.metrics.states",
            "plandiversity.metrics.uniqueness",
            "plandiversity.models.base",
            "plandiversity.models.max_mean",
            "plandiversity.models.max_min",
            "plandiversity.models.max_sum",
            "plandiversity.solvers.base",
            "plandiversity.solvers.exact",
            "plandiversity.solvers.greedy",
        ],
    )
    def test_module_is_importable(self, module):
        __import__(module)
