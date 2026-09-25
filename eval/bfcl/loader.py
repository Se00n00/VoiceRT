"""HF-only loaders -> normalized cases (full datasets, never samples).

Normalized case (matches the runner's expected shape):
  {id, version, category, file, index, question, functions,
   expected, expected_kind}
- single-turn: question = upstream value (str or message list);
  expected = list of {func: {param: [acceptable...]}} or None.
- multi-turn: question = list of turns; expected = list of per-turn GT
  (each a list of call strings or dicts); functions = resolved specs.
- expected_kind: "ast" | "none" (negative control) | "hand" (V1) | "multiturn".
"""
import json
import os

from eval.bfcl import registry
from eval.common import hf_download, read_json_lines


def _parse_json_maybe(value):
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _load_answers(path):
    out = {}
    for e in read_json_lines(path):
        if isinstance(e, dict) and "id" in e:
            out[e["id"]] = e.get("ground_truth")
    return out


def _load_func_docs(doc_dir):
    specs, file_of = {}, {}
    if not os.path.isdir(doc_dir):
        return specs, file_of
    for fn in sorted(os.listdir(doc_dir)):
        if not fn.endswith(".json"):
            continue
        full = os.path.join(doc_dir, fn)
        try:
            with open(full, encoding="utf-8") as f:
                text = f.read()
            try:
                payload = json.loads(text)
                items = payload if isinstance(payload, list) else [payload]
            except Exception:
                items = [json.loads(l) for l in text.splitlines() if l.strip()]
        except Exception:
            continue
        for spec in items:
            if isinstance(spec, dict) and spec.get("name"):
                specs.setdefault(spec["name"], spec)
                file_of.setdefault(spec["name"], fn)
    return specs, file_of


def _resolve_mt_specs(specs, file_of, involved):
    missing = [c for c in involved if c not in registry.CLASS_TO_FILES]
    assert not missing, f"unmapped involved_classes: {missing}"
    by_file = {}
    for name, fn in file_of.items():
        by_file.setdefault(fn, []).append(name)
    names = []
    for c in involved:
        for fn in registry.CLASS_TO_FILES[c]:
            names.extend(by_file.get(fn, []))
    return [specs[n] for n in names if n in specs]


def _fetch_prefix(repo, cache_dir, rev):
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo, repo_type="dataset",
                             revision=rev, cache_dir=cache_dir)


def load_v3(cache_dir, rev="main", categories=None):
    """Full gradeable V3 set from the official HF dataset. Returns (cases, meta)."""
    root = _fetch_prefix(registry.OFFICIAL_REPO, cache_dir, rev)
    func_root = os.path.join(root, registry.FUNC_DOC_DIR)
    specs, file_of = _load_func_docs(func_root)

    cases, orphans = [], []
    files_touched = []

    def data_path(remote):
        return os.path.join(root, remote)

    for remote, ans_remote, cat, kind in registry.V3_SINGLE:
        if categories and cat not in categories:
            continue
        entries = read_json_lines(data_path(remote))
        answers = _load_answers(data_path(ans_remote)) if ans_remote else {}
        files_touched.append(remote)
        for i, e in enumerate(entries):
            gt = answers.get(e.get("id")) if ans_remote else None
            if kind != "none" and gt is None:
                orphans.append((remote, e.get("id")))
                continue
            cases.append({
                "id": e.get("id") or f"{cat}_{i}",
                "version": "v3", "category": cat,
                "file": remote, "index": i,
                "question": e.get("question"),
                "functions": e.get("function", []) or [],
                "expected": gt,
                "expected_kind": "none" if gt is None else "ast",
            })

    if (not categories) or "multi_turn" in (categories or ["multi_turn"]):
        for remote, ans_remote in registry.V3_MULTI:
            entries = read_json_lines(data_path(remote))
            answers = _load_answers(data_path(ans_remote))
            files_touched.append(remote)
            for i, e in enumerate(entries):
                gt = answers.get(e.get("id"))
                if gt is None:
                    orphans.append((remote, e.get("id")))
                    continue
                involved = e.get("involved_classes") or []
                cases.append({
                    "id": e.get("id"), "version": "v3",
                    "category": "multi_turn", "file": remote, "index": i,
                    "question": e.get("question"),
                    "involved_classes": involved,
                    "functions": _resolve_mt_specs(specs, file_of, involved),
                    "expected": gt,
                    "expected_kind": "multiturn",
                })

    meta = {"repo": registry.OFFICIAL_REPO, "rev": rev,
            "files": files_touched, "orphans_dropped": len(orphans),
            "orphans_sample": orphans[:5]}
    return cases, meta


def _require_datasets():
    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "ABORT: V1/V2 mirror loaders need the 'datasets' + 'pyarrow' "
            "packages (parquet mirrors). On Kaggle: "
            "pip install datasets pyarrow. V3 works without them."
        ) from exc


def load_v2(cache_dir, categories=None):
    """Full V2 set from third-party HF parquet mirrors. Returns (cases, meta)."""
    _require_datasets()
    from datasets import load_dataset

    seen, cases = {}, []
    files_touched = []
    for split, repo in sorted(registry.V2_MIRROR_REPOS.items()):
        ds = load_dataset(repo, split="train", cache_dir=cache_dir)
        files_touched.append(f"{repo}#train:{len(ds)}")
        for i, row in enumerate(ds):
            cid = str(row.get("id") or f"{split}_{i}")
            if cid in seen:
                continue
            cat = str(row.get("test_category") or split).strip() or split
            if categories and cat not in categories:
                continue
            functions = _parse_json_maybe(row.get("functions")) or []
            if isinstance(functions, dict):
                functions = [functions]
            gt = _parse_json_maybe(row.get("ground_truth"))
            if gt is None and cat not in ("irrelevance", "chatable"):
                # relevance split rows without GT are negative controls
                kind = "none"
            else:
                kind = "none" if gt is None else "ast"
            question = row.get("question")
            if not question and row.get("dialog"):
                question = _parse_json_maybe(row.get("dialog"))
            seen[cid] = True
            cases.append({
                "id": cid, "version": "v2", "category": cat,
                "file": f"{repo}#{split}", "index": i,
                "question": question,
                "functions": functions,
                "expected": gt,
                "expected_kind": kind,
            })
    meta = {"mirrors": registry.V2_MIRROR_REPOS,
            "files": files_touched, "n": len(cases),
            "note": "third-party mirrors; verify test_category mix with --list-only"}
    return cases, meta


def load_v1(questions_path=None, grades_path=None):
    """V1 test has no machine GT on HF (upstream deleted it).

    Without explicit user-supplied files there is nothing gradeable to load;
    this returns ([], meta) explaining why, instead of silently scoring zero.
    With --v1-questions (JSONL of {id,question,function}) + --v1-grades
    (JSON {id: expected-AST}), cases are built with expected_kind "hand".
    """
    if not questions_path:
        return [], {"note": "V1 test has no machine answers on HF; "
                            "pass --v1-questions/--v1-grades to grade V1"}
    grades = {}
    if grades_path:
        with open(grades_path, encoding="utf-8") as f:
            grades = json.load(f)
    entries = read_json_lines(questions_path)
    cases = []
    for i, e in enumerate(entries):
        cid = e.get("id") or f"v1_{i}"
        fn = e.get("function", e.get("functions", []))
        cases.append({
            "id": cid, "version": "v1", "category": e.get("category", "simple"),
            "file": os.path.basename(questions_path), "index": i,
            "question": e.get("question"),
            "functions": [fn] if isinstance(fn, dict) else (fn or []),
            "expected": grades.get(cid),
            "expected_kind": "hand",
        })
    return cases, {"questions": questions_path,
                   "graded": sum(1 for c in cases if c["expected"] is not None),
                   "n": len(cases)}


def load_cases(versions, cache_dir, rev="main", categories=None,
               v1_questions=None, v1_grades=None):
    cases, metas = [], {}
    if "v3" in versions:
        c, m = load_v3(cache_dir, rev, categories)
        cases.extend(c)
        metas["v3"] = m
    if "v2" in versions:
        c, m = load_v2(cache_dir, categories)
        cases.extend(c)
        metas["v2"] = m
    if "v1" in versions:
        c, m = load_v1(v1_questions, v1_grades)
        cases.extend(c)
        metas["v1"] = m
    return cases, metas


def hf_file_bytes(*a, **k):  # re-export for CLI preflight
    return hf_download(*a, **k)
