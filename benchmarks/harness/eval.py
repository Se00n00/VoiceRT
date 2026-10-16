"""Harness suites: one Benchmark adapter + one evaluator, single entry point.

Contract: Benchmark vendors the GAIA (gated HF) and TB 2.1 (Harbor repo)
suites into harness manifests; HarnessEval runs one warmed VoiceAgent
over them as autonomous turns (auto-approving confirm; policy denies
still apply). GAIA rows grade reply_match locally; TB 2.1 rows are
verifier-only (trajectory recorded, passed=null, graded later under
harbor/docker). Dependencies: huggingface_hub + pandas (adapt gaia
only), git (adapt tb21 only), benchmarks.agent_eval_lib (runner),
benchmarks.eval_agent_tasks (grade_task, summarize). Called by
benchmarks/run_all_evals.sh and benchmarks/kaggle.md cells.

Usage:
  PYTHONPATH=. .venv/bin/python -u benchmarks/harness/eval.py --adapt all
  PYTHONPATH=. .venv/bin/python -u benchmarks/harness/eval.py \
      --suite gaia --limit 5 --tag smoke
  PYTHONPATH=. .venv/bin/python -u benchmarks/harness/eval.py \
      --suite tb21 --limit 25 --offset 25 --tag kaggle --backend bonsai
Results: benchmarks/results/harness_<suite>_<tag>_<ts>.json
(--suite all writes harness_all_<tag>_<ts>.json with per-suite summaries).
--limit/--offset apply per suite; --level filters GAIA levels (TB 2.1
tasks are all level 1).
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from benchmarks.agent_eval_lib import areanchor_workdir, make_workdir, run_task, warm_agent
from benchmarks.eval_agent_tasks import grade_task, summarize

HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
SUITES = {"gaia": "gaia.jsonl", "tb21": "tb21.jsonl"}


class Benchmark:
    """Vendored suite manifests: fetch + adapt + save per suite.

    ..
    """

    GAIA_REPO = "gaia-benchmark/GAIA"
    TB21_URL = "https://github.com/harbor-framework/terminal-bench-2-1.git"
    TB21_RANK = {"easy": 0, "medium": 1, "hard": 2}

    def __init__(self, root: str) -> None:
        """Point the adapter at the harness dir.

        ..
        """
        self.root = root

    def manifest_path(self, suite: str) -> str:
        """Harness manifest path for a suite. Pure.

        ..
        """
        return os.path.join(self.root, SUITES[suite])

    def fetch(self, suite: str) -> str:
        """Vendor a suite locally. Returns its source dir.

        ..
        """
        if suite == "tb21":
            root = os.path.join(self.root, "tb21")
            if os.path.isdir(os.path.join(root, ".git")):
                import subprocess

                subprocess.run(["git", "-C", root, "pull", "--ff-only"],
                               check=True)
            else:
                import subprocess

                subprocess.run(["git", "clone", "--depth", "1",
                                self.TB21_URL, root], check=True)
            return os.path.join(root, "tasks")
        from huggingface_hub import snapshot_download

        return snapshot_download(self.GAIA_REPO, repo_type="dataset",
                                 local_dir=os.path.join(self.root, "gaia"),
                                 allow_patterns=["2023/validation/*"])

    @staticmethod
    def _normalize_answer(text: str) -> str:
        """GAIA official rule: exact match after strip. Pure.

        ..
        """
        return str(text or "").strip()

    def adapt(self, suite: str, vendored: str = "") -> list:
        """Vendored source dir -> harness tasks. Pure-ish.

        ..
        """
        if suite == "tb21":
            return self._adapt_tb21(vendored or os.path.join(
                self.root, "tb21", "tasks"))
        return self._adapt_gaia(vendored or os.path.join(self.root, "gaia"))

    def _adapt_tb21(self, tasks_dir: str) -> list:
        """Harbor tasks dir -> harness tasks, easy-first. Pure-ish.

        ..
        """
        import tomllib

        tasks = []
        names = sorted(d for d in os.listdir(tasks_dir)
                       if os.path.isdir(os.path.join(tasks_dir, d)))
        for name in names:
            task_dir = os.path.join(tasks_dir, name)
            toml_path = os.path.join(task_dir, "task.toml")
            inst_path = os.path.join(task_dir, "instruction.md")
            if not os.path.isfile(toml_path) or not os.path.isfile(inst_path):
                continue
            with open(toml_path, "rb") as f:
                meta = tomllib.load(f)
            with open(inst_path, encoding="utf-8", errors="replace") as f:
                instruction = f.read().strip()
            if not instruction:
                continue
            env = meta.get("environment", {})
            agent = meta.get("agent", {})
            info = meta.get("metadata", {})
            image = str(env.get("docker_image", ""))
            tasks.append({
                "id": f"tb21-{name}",
                "level": 1,
                "prompt": instruction,
                "setup": {},
                "attachments": [],
                "oracle": [{"type": "tb21", "task": name,
                            "docker_image": image}],
                "category": str(info.get("category", "")),
                "difficulty": str(info.get("difficulty", "")),
                "docker_image": image,
                "agent_timeout": agent.get("timeout_sec", 900),
            })
        tasks.sort(key=lambda t: (self.TB21_RANK.get(t["difficulty"], 3),
                                  t["id"]))
        return tasks

    def _adapt_gaia(self, split_dir: str) -> list:
        """GAIA validation parquet -> harness tasks, file order kept.

        Schema-tolerant: validation parquet mirrors metadata.jsonl
        (task_id/Question/Level/Final answer/file_name), candidate keys
        cover renames. Attachment missing -> task kept, noted.
        ..
        """
        import pandas as pd

        df = pd.read_parquet(os.path.join(split_dir, "2023", "validation",
                                           "metadata.parquet"))
        cols = {c.lower().replace(" ", "_"): c for c in df.columns}

        def col(*names):
            for n in names:
                if n in cols:
                    return df[cols[n]]
            return None

        ids = col("task_id", "id")
        questions = col("question")
        levels = col("level")
        answers = col("final_answer", "final-answer", "answer")
        files = col("file_name", "filename", "file")
        tasks = []
        for i in range(len(df)):
            q = str(questions.iloc[i] or "").strip()
            if not q:
                continue
            try:
                level = int(str(levels.iloc[i] or "1").strip()[-1])
            except Exception:
                level = 1
            att, missing = [], []
            fname = str(files.iloc[i] or "").strip() if files is not None else ""
            if fname and fname.lower() != "none":
                rel = os.path.join("benchmarks", "harness", "gaia", "2023",
                                   "validation", os.path.basename(fname))
                full = os.path.join(self.root, "gaia", "2023", "validation",
                                    os.path.basename(fname))
                (att if os.path.isfile(full) else missing).append(rel)
            tasks.append({
                "id": f"gaia-{str(ids.iloc[i] or i)[:8]}",
                "level": level,
                "prompt": q,
                "setup": {},
                "attachments": att,
                "attachment_missing": missing,
                "oracle": [{"type": "reply_match",
                            "text": self._normalize_answer(
                                answers.iloc[i] if answers is not None else "")}],
            })
        return tasks

    def save(self, suite: str, tasks: list) -> str:
        """Harness tasks -> manifest file. Returns the path. Pure-ish.

        ..
        """
        out = self.manifest_path(suite)
        with open(out, "w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t, ensure_ascii=False, sort_keys=True) + "\n")
        return out

    def refresh(self, suite: str) -> str:
        """Fetch + adapt + save one suite. Returns the manifest path.

        ..
        """
        tasks = self.adapt(suite, self.fetch(suite))
        out = self.save(suite, tasks)
        extra = ""
        if suite == "tb21":
            by_diff: dict = {}
            for t in tasks:
                by_diff[t["difficulty"]] = by_diff.get(t["difficulty"], 0) + 1
            extra = f", difficulty {by_diff}"
        print(f"wrote {out}: {len(tasks)} tasks{extra}", flush=True)
        return out


class HarnessEval:
    """One warmed agent evaluating harness suites.

    ..
    """

    def __init__(self, args) -> None:
        """Keep run config; the agent warms on warm().
        ..
        """
        self.args = args
        self.agent = None

    async def warm(self) -> None:
        """Warm one LLM-leg agent for all suites.

        ..
        """
        self.agent = await warm_agent(model=self.args.model,
                                      backend=self.args.backend,
                                      max_tokens=self.args.max_tokens or None)

    @staticmethod
    def load(suite: str) -> list:
        """Harness manifest -> tasks (jsonl or json array). Pure-ish.

        ..
        """
        with open(os.path.join(HARNESS_DIR, SUITES[suite]),
                  encoding="utf-8") as f:
            raw = f.read()
        try:
            tasks = json.loads(raw)
            return tasks if isinstance(tasks, list) else [tasks]
        except Exception:
            return [json.loads(l) for l in raw.splitlines() if l.strip()]

    async def run_suite(self, suite: str, tasks: list) -> list:
        """Run one suite; rows carry suite + graded/pending verdicts.

        ..
        """
        rows = []
        for task in tasks:
            if self.args.level and int(task.get("level", 1)) not in self.args.level:
                continue
            workdir = make_workdir(task.get("setup"), task.get("attachments"))
            await areanchor_workdir(self.agent, workdir)
            res = await run_task(self.agent, task["prompt"], workdir,
                                 timeout_s=self.args.timeout,
                                 session_id=f"harness-{suite}-{task['id']}")
            checks = grade_task(workdir, res["reply"], task.get("oracle"))
            if checks and all(c["ok"] is None for c in checks):
                passed = None
            else:
                passed = bool(checks) and all(c["ok"] for c in checks)
            row = {"suite": suite, "id": task["id"],
                   "level": int(task.get("level", 1)),
                   "prompt": task["prompt"][:160],
                   "passed": passed, "checks": checks, **res}
            rows.append(row)
            mark = "PEND" if passed is None else ("OK " if passed else "FAIL")
            print(f"[{mark}] {suite}/{task['id']:32s} steps={res['steps']:2d} "
                  f"confirms={res['confirms']} {res['seconds']:6.1f}s "
                  f"reply={res['reply'][:70]!r}", flush=True)
        return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="harness suites (GAIA + TB 2.1)")
    ap.add_argument("--adapt", default="", choices=("", "all", "gaia", "tb21"),
                    help="regenerate manifest(s) instead of evaluating")
    ap.add_argument("--suite", default="all", choices=("all", "gaia", "tb21"))
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B",
                    help="HF id for local legs; model id for cloud legs")
    ap.add_argument("--backend", default="qwen")
    ap.add_argument("--tag", default="harness")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="per-step token budget (0 = backend floor; "
                         "reasoning legs need more, e.g. 2048)")
    ap.add_argument("--limit", type=int, default=0,
                    help="per-suite task cap (0 = full suite)")
    ap.add_argument("--offset", type=int, default=0,
                    help="per-suite task offset (chunked runs)")
    ap.add_argument("--level", type=int, nargs="*", default=[],
                    help="GAIA levels to run, e.g. --level 1 2 (empty = all)")
    args = ap.parse_args()
    if args.adapt:
        bench = Benchmark(HARNESS_DIR)
        suites = [args.adapt] if args.adapt != "all" else ["gaia", "tb21"]
        for suite in suites:
            bench.refresh(suite)
        return
    suites = [args.suite] if args.suite != "all" else ["gaia", "tb21"]
    evaluator = HarnessEval(args)
    asyncio.run(evaluator.warm())
    rows = []
    summaries = {}
    for suite in suites:
        tasks = HarnessEval.load(suite)
        if args.limit:
            tasks = tasks[args.offset:args.offset + args.limit]
        elif args.offset:
            tasks = tasks[args.offset:]
        suite_rows = asyncio.run(evaluator.run_suite(suite, tasks))
        rows.extend(suite_rows)
        summaries[suite] = summarize(suite_rows)
    summary = {"n": len(rows), "suites": summaries} if len(suites) > 1 \
        else summaries[suites[0]]
    print("---")
    print(json.dumps(summary, indent=2))
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = (f"benchmarks/results/harness_{args.suite}_{args.tag}_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": {"suite": args.suite, "model": args.model,
                            "backend": args.backend,
                            "timeout_s": args.timeout},
                   "summary": summary, "rows": rows}, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
