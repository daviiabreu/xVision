"""Suite que so aprova nao prova nada: aqui ela reprova quem viola a propria spec."""

from __future__ import annotations

import pytest

pytest_plugins = ["pytester"]


def test_conformance_fails_a_module_that_ignores_its_declared_preprocess(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile(
        test_violating="""
        import pytest
        from edge.testing.example_module import ViolatingModule
        from edge.modules.conformance import ModuleConformance

        class TestViolating(ModuleConformance):
            @pytest.fixture
            def module(self):
                return ViolatingModule()
        """
    )
    result = pytester.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(passed=_quantos_testes() - 1, failed=1)
    result.stdout.fnmatch_lines(["*test_conformance_respects_declared_preprocess*"])


def _quantos_testes() -> int:
    """Derivar da suite, e nao cravar o total, para teste novo de conformidade nao quebrar este."""
    from edge.modules.conformance import ModuleConformance

    return sum(1 for nome in vars(ModuleConformance) if nome.startswith("test_"))
