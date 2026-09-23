"""Unit tests for eval harness pure functions (no weights, no GPU)."""
import unittest


class TestCaseKeys(unittest.TestCase):
    def test_key_disambiguates_versions(self):
        from benchmarks.bfcl_eval import case_key

        # simple_0 exists in v1 AND v2 AND v3 with different ground truths
        self.assertNotEqual(case_key({"version": "v1", "id": "simple_0"}),
                            case_key({"version": "v2", "id": "simple_0"}))

    def test_filter_categories(self):
        from benchmarks.bfcl_eval import filter_cases

        cases = [{"version": "v3", "id": "a", "category": "simple"},
                 {"version": "v3", "id": "b", "category": "multi_turn"}]
        self.assertEqual(len(filter_cases(cases)), 2)
        kept = filter_cases(cases, categories=["multi_turn"])
        self.assertEqual([c["id"] for c in kept], ["b"])

    def test_filter_done(self):
        from benchmarks.bfcl_eval import filter_cases

        cases = [{"version": "v2", "id": "x", "category": "simple"},
                 {"version": "v2", "id": "y", "category": "simple"}]
        kept = filter_cases(cases, done={"v2/x"})
        self.assertEqual([c["id"] for c in kept], ["y"])

    def test_load_done_ids_missing_file(self):
        from benchmarks.bfcl_eval import load_done_ids

        self.assertEqual(load_done_ids("/definitely/not/here.json"), set())

    def test_load_done_ids_shape(self):
        import json
        import tempfile

        from benchmarks.bfcl_eval import load_done_ids

        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as f:
            json.dump({"rows": [{"version": "v3", "id": "a"},
                                {"version": "v3", "id": "b"}]}, f)
            path = f.name
        try:
            self.assertEqual(load_done_ids(path), {"v3/a", "v3/b"})
        finally:
            import os

            os.unlink(path)


class TestBudgetGuard(unittest.TestCase):
    def test_small_case_under_budget(self):
        from benchmarks.bfcl_eval import PROMPT_TOKEN_BUDGET, estimate_case_tokens

        case = {"question": [[{"role": "user", "content": "hi"}]],
                "functions": [{"name": "f"}]}
        self.assertLess(estimate_case_tokens(case, 320), PROMPT_TOKEN_BUDGET)

    def test_huge_case_over_budget(self):
        from benchmarks.bfcl_eval import PROMPT_TOKEN_BUDGET, estimate_case_tokens

        case = {"question": [[{"role": "user", "content": "x" * 40000}]],
                "functions": [{"name": "f"}]}
        self.assertGreater(estimate_case_tokens(case, 320), PROMPT_TOKEN_BUDGET)

    def test_skipped_error_rows_summarize(self):
        from benchmarks.bfcl_eval import error_row, skipped_row, summarize

        case = {"id": "s1", "version": "v3", "category": "simple",
                "question": "hi"}
        rows = [skipped_row(case, "too big"),
                error_row(case, "boom", 0.5),
                {"id": "ok", "version": "v3", "category": "simple",
                 "tp": 1, "fp": 0, "fn": 0, "first_tp": 1,
                 "arg_valid": True, "retries": 0, "steps": 1,
                 "attempts": 1, "passed": True, "seconds": 2.0,
                 "expect_none": False}]
        s = summarize(rows)
        self.assertEqual(s["n"], 3)
        self.assertEqual(s["n_skipped"], 1)
        self.assertEqual(s["n_errors"], 1)
        # skipped/error count as failures (conservative), never as passes
        self.assertEqual(s["pass_at_k"], round(1 / 3, 3))


class TestNestedMatch(unittest.TestCase):
    def test_single_level_unwrap(self):
        from benchmarks.bfcl_eval import _val_match

        # SQL-style GT: columns: [["name"]] vs predicted "name"
        self.assertTrue(_val_match("name", [["name"]]))
        self.assertFalse(_val_match("other", [["name"]]))

    def test_plain_still_works(self):
        from benchmarks.bfcl_eval import _val_match

        self.assertTrue(_val_match("10", [10]))
        self.assertTrue(_val_match("True", [True]))
        self.assertFalse(_val_match("5", [4]))


class TestManifestLoading(unittest.TestCase):
    def test_load_plain_and_gz(self):
        import gzip
        import json
        import tempfile

        from benchmarks.bfcl_eval import load_manifest

        rows = [{"id": "a", "version": "v3"}, {"id": "b", "version": "v3"}]
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as f:
            json.dump(rows, f)
            plain = f.name
        gz = plain + ".jsonl.gz"
        try:
            with gzip.open(gz, "wt", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            self.assertEqual(load_manifest(plain), rows)
            self.assertEqual(load_manifest(gz), rows)
        finally:
            import os

            os.unlink(plain)
            os.unlink(gz)

    def test_load_wrapped_cases(self):
        import json
        import tempfile

        from benchmarks.bfcl_eval import load_manifest

        rows = [{"id": "a"}]
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as f:
            json.dump({"cases": rows}, f)
            path = f.name
        try:
            self.assertEqual(load_manifest(path), rows)
        finally:
            import os

            os.unlink(path)


class TestFullManifests(unittest.TestCase):
    """Committed full-dataset manifests: shape, counts, key uniqueness."""

    def _load(self, ver):
        import gzip
        import json

        with gzip.open(f"benchmarks/bfcl_{ver}_full.jsonl.gz",
                       "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_counts_match_official(self):
        v2 = self._load("v2")
        v3 = self._load("v3")
        self.assertEqual(len(v2), 1540)   # official V2 static total
        # official V3 static 2140 + 1351 live (1 GT-less orphan dropped)
        self.assertEqual(len(v3), 2140 + 1351 - 1)

    def test_keys_unique_per_version(self):
        import collections

        from benchmarks.bfcl_eval import case_key

        for ver in ("v2", "v3"):
            rows = self._load(ver)
            keys = [case_key(r) for r in rows]
            dupes = [k for k, n in collections.Counter(keys).items()
                     if n > 1]
            self.assertEqual(dupes, [], f"{ver} dupes: {dupes[:3]}")

    def test_gradeable_shape(self):
        from benchmarks.bfcl_eval import filter_cases

        for ver in ("v2", "v3"):
            rows = self._load(ver)
            for r in rows:
                self.assertIn(r["expected_kind"], ("ast", "none", "multiturn"))
                self.assertTrue(r["question"])
            # full-mode category set drops only multi_turn
            runnable = filter_cases(
                rows, categories=["simple", "multiple", "parallel",
                                  "parallel_multiple", "sql",
                                  "irrelevance", "chatable"])
            kinds = {r["expected_kind"] for r in runnable}
            self.assertNotIn("multiturn", kinds)

    def test_no_orphans(self):
        # every non-irrelevance/chatable single-turn case must carry GT
        # (builder drops GT-less entries loudly instead)
        for ver in ("v2", "v3"):
            for r in self._load(ver):
                if (r["category"] not in ("irrelevance", "chatable")
                        and r["expected_kind"] != "multiturn"):
                    self.assertIsNotNone(r["expected"], r["id"])
class TestResultFlag(unittest.TestCase):
    def test_tn_not_fn(self):
        from benchmarks.toolcall_eval import result_flag, score_case

        # chat case, model silent: scores 0/0/0 -> TN (was misprinted FN)
        tp, fp, fn, _ = score_case({"expected": None}, None, {})
        self.assertEqual((tp, fp, fn), (0, 0, 0))
        self.assertEqual(result_flag(tp, fp, fn), "TN ")

    def test_matrix(self):
        from benchmarks.toolcall_eval import result_flag

        self.assertEqual(result_flag(1, 0, 0), "OK ")
        self.assertEqual(result_flag(0, 1, 1), "fp ")
        self.assertEqual(result_flag(0, 0, 1), "FN ")


class TestGtString(unittest.TestCase):
    def test_quoted_and_bare(self):
        from benchmarks.bfcl_eval import parse_gt_string

        n, a = parse_gt_string("cd(folder='document')")
        self.assertEqual((n, a), ("cd", {"folder": "document"}))
        n, a = parse_gt_string("sort('final_report.pdf')")
        self.assertEqual((n, a), ("sort", {}))

    def test_garbage(self):
        from benchmarks.bfcl_eval import parse_gt_string

        self.assertEqual(parse_gt_string("not a call"), (None, {}))
        self.assertEqual(parse_gt_string(""), (None, {}))


if __name__ == "__main__":
    unittest.main()
