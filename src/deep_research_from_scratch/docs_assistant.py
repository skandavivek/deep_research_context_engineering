
"""Docs Assistant Implementation.

This module implements a RAG-based documentation assistant that can
search indexed documentation and answer questions using retrieved context.
"""

from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
)
from langgraph.graph import END, START, StateGraph

from deep_research_from_scratch.rag_tools import qdrant_search
from deep_research_from_scratch.state_docs import (
    DocsAssistantOutputState,
    DocsAssistantState,
)
from deep_research_from_scratch.utils import get_today_str, think_tool

# ===== CONFIGURATION =====

tools = [qdrant_search, think_tool]
tools_by_name = {tool.name: tool for tool in tools}

model = init_chat_model(model="anthropic:claude-sonnet-4-6")
model_with_tools = model.bind_tools(tools)

# ===== PROMPTS =====

docs_assistant_prompt = """You are a documentation assistant. Your job is to answer questions about the indexed documentation using the qdrant_search tool.

Today's date is {date}.

<Task>
Use the qdrant_search tool to find relevant documentation and answer the user's question accurately.
After each search, use think_tool to assess whether you have enough information to provide a complete answer.
</Task>

<Instructions>
1. Read the user's question carefully to understand what they're asking
2. Search for relevant documentation using qdrant_search with appropriate queries
3. Use think_tool to reflect on the findings and assess if you have sufficient information
4. If needed, search with different or more specific queries to fill gaps
5. Provide a comprehensive, accurate answer based on the retrieved documentation
6. Always cite your sources by mentioning which documents the information came from
</Instructions>

<Hard Limits>
- Use 1-3 search queries maximum per question
- Stop searching when you have sufficient documentation to answer
- If the documentation doesn't contain the answer, say so clearly
- Always cite sources in your final response
</Hard Limits>
"""

# ===== AGENT NODES =====


def llm_call(state: DocsAssistantState):
    """Analyze current state and decide on next actions."""
    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content=docs_assistant_prompt.format(date=get_today_str()))]
                + list(state["messages"])
            )
        ]
    }


def tool_node(state: DocsAssistantState):
    """Execute all tool calls from the previous LLM response."""
    tool_calls = state["messages"][-1].tool_calls

    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observations.append(tool.invoke(tool_call["args"]))

    tool_outputs = [
        ToolMessage(
            content=observation, name=tool_call["name"], tool_call_id=tool_call["id"]
        )
        for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"messages": tool_outputs}


def compress_research(state: DocsAssistantState) -> dict:
    """Compress retrieved documentation into final answer."""
    # Extract raw notes from retrieval
    raw_notes = [
        str(m.content)
        for m in filter_messages(state["messages"], include_types=["tool", "ai"])
    ]

    # Get the final AI response as the answer
    final_message = state["messages"][-1]
    answer = str(final_message.content) if hasattr(final_message, "content") else ""

    return {
        "answer": answer,
        "retrieved_docs": "\n".join(raw_notes),
        "raw_notes": ["\n".join(raw_notes)],
    }


# ===== ROUTING LOGIC =====


def should_continue(state: DocsAssistantState) -> Literal["tool_node", "compress_research"]:
    """Determine whether to continue or provide final answer."""
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"
    return "compress_research"


# ===== GRAPH CONSTRUCTION =====

agent_builder = StateGraph(DocsAssistantState, output_schema=DocsAssistantOutputState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",
        "compress_research": "compress_research",
    },
)
agent_builder.add_edge("tool_node", "llm_call")
agent_builder.add_edge("compress_research", END)

docs_assistant = agent_builder.compile()
