# Copyright (c) Meta Platforms, Inc. and affiliates.


import unittest

from tritonparse.reproducer.import_info import ImportInfo
from tritonparse.reproducer.placeholder_replacer import _render_dependent_imports


def _from_import(names: list[str], aliases: dict[str, str] | None = None) -> ImportInfo:
    return ImportInfo(
        import_type="from_import",
        module="pkg.helpers",
        names=names,
        source_file="/x/kernel.py",
        resolved_path="/x/helpers.py",
        is_external=False,
        lineno=1,
        aliases=aliases or {},
    )


class TestRenderDependentImports(unittest.TestCase):
    """Tests for pruning imports of already-embedded dependency functions."""

    def test_embedded_name_is_dropped(self) -> None:
        # `from pkg.helpers import gelu` where gelu is embedded as a def.
        stmts, bindings = _render_dependent_imports([_from_import(["gelu"])], {"gelu"})
        self.assertEqual(stmts, [])
        self.assertEqual(bindings, [])

    def test_embedded_aliased_name_emits_binding_not_import(self) -> None:
        # `from pkg.helpers import gelu as g` where gelu is embedded: keep the
        # local binding `g` via an assignment, do NOT keep the broken import.
        stmts, bindings = _render_dependent_imports(
            [_from_import(["gelu"], {"g": "gelu"})], {"gelu"}
        )
        self.assertEqual(stmts, [])
        self.assertEqual(bindings, ["g = gelu"])

    def test_non_embedded_name_is_kept(self) -> None:
        stmts, bindings = _render_dependent_imports(
            [_from_import(["helper"])], {"gelu"}
        )
        self.assertEqual(stmts, ["from pkg.helpers import helper"])
        self.assertEqual(bindings, [])

    def test_mixed_names_keep_only_non_embedded(self) -> None:
        stmts, bindings = _render_dependent_imports(
            [_from_import(["gelu", "helper"])], {"gelu"}
        )
        self.assertEqual(stmts, ["from pkg.helpers import helper"])
        self.assertEqual(bindings, [])


if __name__ == "__main__":
    unittest.main()
