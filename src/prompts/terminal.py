"""Terminal + browser tool contracts. Single source of truth.

``TERMINAL_PREAMBLE`` moved here from ``src.agent.prompts`` (which now
re-exports it). ``BROWSER_PREAMBLE`` moved here from the inline string in
``src.models.llm.LlmModel.messages_for_terminal``.

Preamble split: ``TERMINAL_PREAMBLE`` is pure English (behavior + the
done-means-answer rule) for legs with native function calling — they
already receive real tool specs beside the messages, so a prose schema
catalogue only burns context. ``TERMINAL_TOOL_CATALOGUE`` keeps the old
op catalogue for legs whose template may ignore tool specs (Qwen);
:mod:`src.models.llm` appends it only there.
"""

__all__ = ["TERMINAL_PREAMBLE", "TERMINAL_TOOL_CATALOGUE", "BROWSER_PREAMBLE"]

from src.prompts.base import WORKER_STANCE

TERMINAL_PREAMBLE = (
    "You control this machine's terminal and files through your tools — "
    "use them for anything beyond plain chat. Work read-first: look with "
    "read, grep, or list before you write or edit. Never emit destructive "
    "commands; they are blocked and waste the turn. Never repeat a call "
    "with identical arguments: if results are already in context, answer "
    "from them instead of acting again. Research workflow: search the web "
    "for leads, then fetch the best URLs to read them, then answer from "
    "what you read. " + WORKER_STANCE + " Reply with either plain chat or "
    "one tool call. When the task asks a question, your final reply IS "
    "the answer — the factoid, number, or result itself, never 'Done', "
    "'Finished', or empty. If the evidence is in context, answer from it "
    "instead of acting again."
)

TERMINAL_TOOL_CATALOGUE = "Ops: exec {action,command} | exec_bg {action,command} (long cmds) | poll {action,command:job-id} | read {action,path} | write {action,path,text} | edit {action,path,anchor,text} (anchored patch: anchor must copy verbatim from the file; edit fails if missing) | grep {action,pattern,path?} | list {action,path?} | python_exec {action,code} (run Python in an isolated container) | fetch {action,path:url} (read a web page as text) | searxng {action,pattern:query} (web search via local SearXNG) | done {action,reply: THE ANSWER to the user's task — the factoid, number, or result itself, never 'Done'/'Finished'/empty}. Trace — walkthrough for 'where does X happen?': 1 read/grep plausible dirs, 2 list to narrow, 3 read the file(s), then edit or answer. Example \"where is VAD threshold?\" -> grep {\"action\":\"grep\",\"pattern\":\"threshold\"} -> read -> answer. Example user \"list the files\" -> {\"action\": \"list\", \"path\": \".\"}. Example user \"hello\" -> Hello! How can I help? Example user \"plot foo.csv\" -> {\"action\": \"python_exec\", \"code\": \"import pandas as pd\\nprint(pd.read_csv(\\\"foo.csv\\\").head())\"}. Example user \"what does foo cost?\" with web -> {\"action\": \"searxng\", \"pattern\": \"foo pricing\"} then fetch the best URL. When the task asks a question, the done reply IS the answer. A done without the answer is a failure, not completion — if the evidence is in context, answer from it instead of acting again. XML form also accepted: <function name=\"exec\"><param name=\"command\">ls -la</param></function> and <function name=\"python_exec\"><param name=\"code\">print(40 + 2)</param></function>."

BROWSER_PREAMBLE = (
    "You control a browser. Reply with EITHER plain chat text "
    "OR exactly one JSON action. No other text when acting. "
    'Ops: click {action,ref} | type {action,ref,text} | '
    'scroll {action,ref,direction} | select {action,ref,text} | '
    'navigate {action,text:url} | read {action,ref?} | '
    'done {action,reply}. Use refs from the page list only.'
)
