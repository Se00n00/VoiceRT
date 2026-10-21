"""L2/L3 memory injection headers for the long-running agent.

``PAST_EPISODES_HEADER`` frames recalled episodes, ``USER_FACTS_HEADER``
frames the facts block; both are appended to the system prompt by
``MemoryMiddleware``. Facts parsing itself stays in
``src.agent.facts`` — only the LRA-only headers live here.
"""

__all__ = ["PAST_EPISODES_HEADER", "USER_FACTS_HEADER"]

PAST_EPISODES_HEADER = """
[Past episodes — USE these to answer when relevant; prefer them over guessing]
"""

USER_FACTS_HEADER = """
[User facts]
"""
