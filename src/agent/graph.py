"""The voice-turn graph: vad -> {stt | silence} -> respond -> END.

Models are bound with :func:`functools.partial` at build time, so node
functions stay directly unit-testable with fakes. No checkpointer: turn
memory lives in the ``SessionStore`` passed to ``respond`` (RAM-only,
TTL-bound), which also keeps voice audio out of any persisted state.
"""
from functools import partial

from langgraph.graph import END, START, StateGraph

from src.agent import nodes as _nodes
from src.agent.state import AgentState

__all__ = ["build_turn_graph"]


def build_turn_graph(*, vad, stt, llm, tts, sessions=None,
                     trim_pad_s=0.15):
    """Compile one turn: VAD gate, STT, streaming LLM->TTS respond."""
    builder = StateGraph(AgentState)
    builder.add_node("vad", partial(_nodes.vad_node, vad=vad,
                                    trim_pad_s=trim_pad_s))
    builder.add_node("stt", partial(_nodes.stt_node, stt=stt))
    builder.add_node("respond", partial(_nodes.respond_node, llm=llm,
                                        tts=tts, sessions=sessions))
    builder.add_node("silence", _nodes.silence_node)
    builder.add_edge(START, "vad")
    builder.add_conditional_edges(
        "vad", _nodes.route_after_vad, {"stt": "stt", "silence": "silence"})
    builder.add_edge("stt", "respond")
    builder.add_edge("respond", END)
    builder.add_edge("silence", END)
    return builder.compile()
