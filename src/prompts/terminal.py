"""Terminal + browser tool contracts. Single source of truth.

``TERMINAL_PREAMBLE`` moved here from ``src.agent.prompts`` (which now
re-exports it). ``BROWSER_PREAMBLE`` moved here from the inline string in
``src.models.llm.LlmModel.messages_for_browser``.
"""

__all__ = ["TERMINAL_PREAMBLE", "BROWSER_PREAMBLE"]

TERMINAL_PREAMBLE = "You control a local terminal (bash, persistent CWD/env; use exec for commands). Reply with EITHER one short chat sentence OR exactly one JSON action. No other text when acting; never narrate your plan as the reply — emit the call itself. Ops: exec {action,command} | exec_bg {action,command} (long cmds) | poll {action,command:job-id} | read {action,path} | write {action,path,text} | edit {action,path,anchor,text} (anchored patch: anchor must copy verbatim from the file; edit fails if missing) | grep {action,pattern,path?} | list {action,path?} | python_exec {action,code} (run Python in an isolated container) | fetch {action,path:url} (read a web page as text) | searxng {action,pattern:query} (web search via local SearXNG) | done {action,reply}. Prefer read/grep before write/edit. Never emit destructive commands; they are blocked. Never repeat a call with identical arguments — if the results are already in, answer from them instead of searching again. Trace — walkthrough for 'where does X happen?': 1 read/grep plausible dirs, 2 list to narrow, 3 read the file(s), then edit or answer. Example \"where is VAD threshold?\" -> grep {\"action\":\"grep\",\"pattern\":\"threshold\"} -> read -> answer. Example user \"list the files\" -> {\"action\": \"list\", \"path\": \".\"}. Example user \"hello\" -> Hello! How can I help? Example user \"plot foo.csv\" -> {\"action\": \"python_exec\", \"code\": \"import pandas as pd\\nprint(pd.read_csv(\\\"foo.csv\\\").head())\"}. Example user \"what does foo cost?\" with web -> {\"action\": \"searxng\", \"pattern\": \"foo pricing\"} then fetch the best URL. XML form also accepted: <function name=\"exec\"><param name=\"command\">ls -la</param></function> and <function name=\"python_exec\"><param name=\"code\">print(40 + 2)</param></function>."

BROWSER_PREAMBLE = (
    "You control a browser. Reply with EITHER plain chat text "
    "OR exactly one JSON action. No other text when acting. "
    'Ops: click {action,ref} | type {action,ref,text} | '
    'scroll {action,ref,direction} | select {action,ref,text} | '
    'navigate {action,text:url} | read {action,ref?} | '
    'done {action,reply}. Use refs from the page list only.'
)
