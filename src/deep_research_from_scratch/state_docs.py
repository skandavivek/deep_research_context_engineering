
"""State Definitions for Docs Assistant.

This module defines the state objects for the RAG-based documentation
assistant workflow, including message history and retrieval metadata.
"""

import operator
from typing_extensions import Annotated, List, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class DocsAssistantState(TypedDict):
    """State for the docs assistant containing message history and retrieval metadata."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    retrieved_docs: str  # Compressed retrieved context
    raw_notes: Annotated[List[str], operator.add]


class DocsAssistantOutputState(TypedDict):
    """Output state for the docs assistant."""

    answer: str
    retrieved_docs: str
    raw_notes: Annotated[List[str], operator.add]
    messages: Annotated[Sequence[BaseMessage], add_messages]
