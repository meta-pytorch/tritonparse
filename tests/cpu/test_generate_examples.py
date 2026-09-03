#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for the example-trace generator's machine-independence guarantees.

The generator itself needs a GPU, but the parts that decide what ends up in a
published artifact -- path rewriting, the leak check, and the raw-log cap -- are
pure and are what actually has to be right. An example that ships an internal
path is the failure this module is here to prevent.
"""

import gzip
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tritonparse.tools.generate_examples import (
    _build_rewrite_rules,
    _write_capped_raw_log,
    Artifacts,
    assert_clean,
    generate,
    install,
    RewriteRule,
    sanitize_text,
    TRITON_TRACE_TOKEN,
    Workload,
    WORKLOADS,
)


class SanitizeTextTest(unittest.TestCase):
    """Tests for path rewriting."""

    def test_rewrites_plain_path(self):
        rules = [RewriteRule("/checkout/tritonparse", "/tritonparse")]
        self.assertEqual(
            sanitize_text('{"filename": "/checkout/tritonparse/a.py"}', rules),
            '{"filename": "/tritonparse/a.py"}',
        )

    def test_rewrites_inside_ir_loc_records(self):
        """Paths also live inside IR text, not just in JSON string values."""
        rules = [RewriteRule("/checkout/tritonparse", "/tritonparse")]
        ir = '#loc2 = loc("/checkout/tritonparse/tests/k.py":20:28)'
        self.assertEqual(
            sanitize_text(ir, rules),
            '#loc2 = loc("/tritonparse/tests/k.py":20:28)',
        )

    def test_applies_every_matching_rule(self):
        rules = [RewriteRule("/checkout", "/repo"), RewriteRule("/opt/env", "/python")]
        text = "/checkout/a.py and /opt/env/lib/x.py"
        self.assertEqual(sanitize_text(text, rules), "/repo/a.py and /python/lib/x.py")

    def test_torchinductor_regex_rule_matches_any_user(self):
        rules = [
            RewriteRule(
                r"/tmp/torchinductor_[^/\"]+", "/tmp/torchinductor", is_regex=True
            )
        ]
        text = '"/tmp/torchinductor_someone/ab/kernel.py"'
        self.assertEqual(
            sanitize_text(text, rules), '"/tmp/torchinductor/ab/kernel.py"'
        )

    def test_literal_rule_does_not_interpret_regex_metacharacters(self):
        """is_regex is explicit, so a path containing [ or . stays literal."""
        rules = [RewriteRule("/checkout/a.b", "/x")]
        self.assertEqual(
            sanitize_text("/checkout/a.b /checkout/axb", rules), "/x /checkout/axb"
        )

    def test_regex_replacement_is_treated_literally(self):
        rule = RewriteRule(r"/old/[^/]+", r"C:\new\1", is_regex=True)
        self.assertEqual(rule.apply("/old/path/file.py"), r"C:\new\1/file.py")

    def test_output_is_still_valid_json(self):
        rules = [RewriteRule("/checkout/tritonparse", "/tritonparse")]
        record = {"stack": [{"filename": "/checkout/tritonparse/tests/k.py"}]}
        rewritten = sanitize_text(json.dumps(record), rules)
        self.assertEqual(
            json.loads(rewritten)["stack"][0]["filename"], "/tritonparse/tests/k.py"
        )


class RewriteRulesTest(unittest.TestCase):
    """Tests for the machine-derived rule set."""

    def test_rules_are_longest_prefix_first(self):
        """The repo lives under $HOME on a devserver.

        If the broad $HOME rule ran first it would rewrite the prefix out from
        under the specific repo rule, and every repo path would come out as
        /home/user/... instead of /tritonparse/...
        """
        lengths = [
            len(rule.pattern) for rule in _build_rewrite_rules() if not rule.is_regex
        ]
        self.assertEqual(lengths, sorted(lengths, reverse=True))

    def test_regex_rules_have_explicit_precedence(self):
        rules = _build_rewrite_rules()
        self.assertTrue(rules[0].is_regex)
        self.assertGreater(rules[0].priority, rules[-1].priority)

    def test_rules_cover_the_repo_root(self):
        replacements = [rule.replacement for rule in _build_rewrite_rules()]
        self.assertIn("/tritonparse", replacements)


class AssertCleanTest(unittest.TestCase):
    """Tests for the pre-publication leak check."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write(self, name, text):
        path = Path(self.tmp) / name
        if name.endswith(".gz"):
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                fh.write(text)
        else:
            path.write_text(text, encoding="utf-8")
        return path

    def test_accepts_sanitized_content(self):
        path = self._write("clean.ndjson", '{"filename": "/tritonparse/a.py"}')
        assert_clean(path)  # must not raise

    def test_rejects_fbsource_path(self):
        path = self._write(
            "leaky.ndjson", '{"filename": "/data/users/x/fbsource/fbcode/a.py"}'
        )
        with self.assertRaises(RuntimeError) as ctx:
            assert_clean(path)
        self.assertIn("fbsource", str(ctx.exception))

    def test_rejects_home_directory(self):
        path = self._write("leaky.ndjson", f'{{"f": "{os.path.expanduser("~")}/a.py"}}')
        with self.assertRaises(RuntimeError):
            assert_clean(path)

    @patch("tritonparse.tools.generate_examples.os.path.expanduser", return_value="/")
    def test_ignores_root_as_home_directory(self, _expanduser):
        path = self._write("clean.ndjson", '{"filename": "/tritonparse/a.py"}')
        assert_clean(path)

    def test_checks_inside_gzip(self):
        """The published example is gzipped; the check must see through it."""
        path = self._write("leaky.ndjson.gz", '{"f": "/x/fbsource/fbcode/a.py"}')
        with self.assertRaises(RuntimeError):
            assert_clean(path)

    def test_trace_token_is_not_flagged(self):
        """The token in the filename is a reviewed value, not a leak."""
        path = self._write("ok.ndjson", f'{{"note": "{TRITON_TRACE_TOKEN}"}}')
        assert_clean(path)

    def test_active_workload_trace_token_is_not_flagged(self):
        path = self._write("ok.ndjson", '{"file": "trace_custom-user_mapped"}')
        with patch.dict(os.environ, {"USER": "custom-user"}):
            assert_clean(path, allowed_trace_token="custom-user")

    @patch(
        "tritonparse.tools.generate_examples.socket.gethostname",
        return_value="devvm26749",
    )
    def test_hostname_check_matches_embedded_substrings(self, _gethostname):
        path = self._write("leaky.ndjson", '{"host": "prefixdevvm26749suffix"}')
        with self.assertRaises(RuntimeError):
            assert_clean(path)

    @patch("tritonparse.tools.generate_examples.socket.gethostname", return_value="dev")
    def test_hostname_check_matches_filename_token(self, _gethostname):
        path = self._write("leaky.ndjson", '{"file": "host_dev_rank_0"}')
        with self.assertRaises(RuntimeError):
            assert_clean(path)

    @patch("tritonparse.tools.generate_examples.getpass.getuser")
    def test_username_falls_back_when_user_is_empty(self, getuser):
        getuser.return_value = "fallback-user"
        path = self._write("leaky.ndjson", '{"owner": "user_fallback-user_rank"}')
        with patch.dict(os.environ, {"USER": ""}):
            with self.assertRaises(RuntimeError):
                assert_clean(path)

    def test_username_check_matches_embedded_substrings(self):
        path = self._write("leaky.ndjson", '{"owner": "prefixbuildusersuffix"}')
        with patch.dict(os.environ, {"USER": "builduser"}):
            with self.assertRaises(RuntimeError):
                assert_clean(path)

    @patch("tritonparse.tools.generate_examples.getpass.getuser", return_value="")
    def test_unresolvable_username_is_an_error(self, _getuser):
        path = self._write("clean.ndjson", "{}")
        with patch.dict(os.environ, {"USER": ""}):
            with self.assertRaisesRegex(RuntimeError, "current username"):
                assert_clean(path)


class CappedRawLogTest(unittest.TestCase):
    """Tests for the raw-log launch cap."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _make_log(self, n_launches):
        path = Path(self.tmp) / "raw.ndjson"
        lines = [json.dumps({"event_type": "compilation", "i": 0})]
        lines += [
            json.dumps({"event_type": "launch", "i": i}) for i in range(n_launches)
        ]
        lines.append(json.dumps({"event_type": "autotune", "i": 0}))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _read(self, path):
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def test_caps_launches_but_keeps_other_events(self):
        source = self._make_log(100)
        target = Path(self.tmp) / "out.ndjson"
        _write_capped_raw_log(source, target, max_launches=5)

        events = self._read(target)
        kinds = [e["event_type"] for e in events]
        self.assertEqual(kinds.count("launch"), 5)
        # Every non-launch record survives: the fixture's job is to cover each
        # shape the writer emits.
        self.assertEqual(kinds.count("compilation"), 1)
        self.assertEqual(kinds.count("autotune"), 1)

    def test_keeps_the_first_launches_in_order(self):
        source = self._make_log(10)
        target = Path(self.tmp) / "out.ndjson"
        _write_capped_raw_log(source, target, max_launches=3)

        launches = [e for e in self._read(target) if e["event_type"] == "launch"]
        self.assertEqual([e["i"] for e in launches], [0, 1, 2])

    def test_under_the_cap_is_a_passthrough(self):
        source = self._make_log(2)
        target = Path(self.tmp) / "out.ndjson"
        _write_capped_raw_log(source, target, max_launches=20)
        self.assertEqual(len(self._read(target)), 4)

    def test_unparseable_line_is_kept(self):
        """A malformed line is not a launch, so it must not be dropped."""
        source = Path(self.tmp) / "raw.ndjson"
        source.write_text('not json\n{"event_type": "launch"}\n', encoding="utf-8")
        target = Path(self.tmp) / "out.ndjson"
        _write_capped_raw_log(source, target, max_launches=0)

        lines = target.read_text().splitlines()
        self.assertEqual(lines, ["not json"])

    def test_non_object_json_lines_are_kept(self):
        source = Path(self.tmp) / "raw.ndjson"
        source.write_text('["launch"]\n"launch"\n1\n', encoding="utf-8")
        target = Path(self.tmp) / "out.ndjson"
        _write_capped_raw_log(source, target, max_launches=0)

        self.assertEqual(target.read_text(), source.read_text())

    def test_blank_lines_are_kept(self):
        source = Path(self.tmp) / "raw.ndjson"
        source.write_text('{"event_type": "compilation"}\n\n', encoding="utf-8")
        target = Path(self.tmp) / "out.ndjson"
        _write_capped_raw_log(source, target, max_launches=0)
        self.assertEqual(target.read_text(), source.read_text())


class WorkloadRegistryTest(unittest.TestCase):
    """Tests for the workload declarations."""

    def test_every_workload_is_well_formed(self):
        self.assertGreater(len(WORKLOADS), 0)
        for name, workload in WORKLOADS.items():
            with self.subTest(workload=name):
                self.assertEqual(workload.name, name)
                self.assertTrue(workload.description)
                self.assertTrue(callable(workload.run))
                # The token becomes the USER field of the trace filename, which
                # unified_parse only accepts after the fixed prefix.
                self.assertTrue(workload.trace_token)
                self.assertNotIn("/", workload.trace_token)
                # The viewer loads a single file, so no example may split its
                # parsed output across per-frame files.
                self.assertFalse(workload.split_inductor_compilations)
                if workload.keep_raw_log:
                    self.assertTrue(workload.raw_fixture_subdir)

    def test_triton_workload_keeps_the_published_filename(self):
        """Renaming the published example would break already-shared URLs.

        README, docs and website/src/App.tsx all hard-code
        dedicated_log_triton_trace_findhao__mapped.ndjson.gz.
        """
        self.assertEqual(WORKLOADS["triton"].trace_token, "findhao")


class GenerateTest(unittest.TestCase):
    def test_rejects_compressed_trace_configuration_before_running_workload(self):
        workload = WORKLOADS["triton"]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"TRITON_TRACE_COMPRESSION": "gzip"}),
        ):
            with self.assertRaisesRegex(RuntimeError, "unset TRITON_TRACE_COMPRESSION"):
                generate(workload, Path(tmp) / "out")


class InstallTest(unittest.TestCase):
    def test_raw_log_uses_the_workload_fixture_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapped = root / "mapped.ndjson.gz"
            raw_log = root / "raw.ndjson"
            mapped.write_bytes(b"mapped")
            raw_log.write_text("raw\n", encoding="utf-8")
            workload = Workload(
                name="custom",
                description="custom fixture layout",
                run=lambda: None,
                init_kwargs={},
                trace_token="custom",
                raw_fixture_subdir="custom_logs",
                keep_raw_log=True,
            )
            artifacts = Artifacts(mapped=mapped, parsed_extras=[], raw_log=raw_log)
            fixture_root = root / "fixtures"
            with patch(
                "tritonparse.tools.generate_examples.FIXTURE_ROOT", fixture_root
            ):
                install(workload, artifacts)
            self.assertEqual(
                (fixture_root / "custom_logs" / "raw.ndjson").read_text(), "raw\n"
            )
