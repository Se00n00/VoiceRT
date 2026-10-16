# AGENTS.md — how to write any file here (local, NOT committed)

## Imports

Every import lives on the top of the file — never lazy imports
hiding inside functions. Standard lib, then third party, then
`src.*`, each group alphabetical. If a top import would create a
cycle or drag weights at import time, split the module instead of
hiding the import.

## File top

Every file starts with one docstring: what the file is for, plus the
metadata of the whole file (contract, dependencies, where it is used).

```python
"""
What this file serves (one or two lines). Plus file metadata:
contract, dependencies, who calls it. Never a changelog.
"""
```

## Functions

Every function carries its return type, and its docstring says what
the function is for in two lines, then `..`, then close.

```python
def fun(arg: str) -> str:
    """What this function is for, in two lines.

    ..
    """
```

## Classes

Related functions belong together in one class — never loose groups
of functions that share state or a theme. One class, one job.

## Try / except

Use very very little `try/except` — never one per line. Validate
with plain `if`s instead of catching your own coercions. The only
tries allowed are import guards and one boundary guard per tool
function (so it returns `error: ...` instead of raising).

## Commits

Never commit or push unless the user explicitly asks. And even then:
before committing, always propose the message first and wait for
approval — never commit on the same turn as writing code unasked.
Message shape: `Change_type [scope]: short description`, where type
is one of add (appending), change, fix, remove, chore. Before
staging, scan for secrets and stage only intended files.

## Speaking

Report like this, always: one short verdict line with the live
evidence, then the split (what works / what does not, per part),
then one concrete next-step question. Facts first, no preamble.

## Plan mode

Plan mode is read-only observe-and-plan, but keep it as minimal as
build mode: short verdict + findings + one question. Never a long
report — details wait for the build.

## Build mode

Build mode may edit, run, and verify — but stays as minimal as plan
mode: smallest diff that satisfies the order, then live evidence it
works. Never a long report.
