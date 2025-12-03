from dotenv import load_dotenv

# Load environment variables from .env before importing any models
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from deep_research_from_scratch.research_agent import researcher_agent
from deep_research_from_scratch.research_agent_full import agent
from deep_research_from_scratch.research_agent_mcp import agent_mcp


class ResearchRequest(BaseModel):
    """Request body for the deep research API."""

    query: str


class ResearchResponse(BaseModel):
    """Response body containing the final research report."""

    final_report: str


app = FastAPI(
    title="Deep Research API (Direct Graph Wrapper)",
    description=(
        "Simple FastAPI wrapper around the LangGraph deep research graph.\n\n"
        "- POST /deep_research: web search + compression (Tavily-based agent)\n"
        "- POST /deep_research_mcp: MCP filesystem + compression (local docs agent)\n"
        "- POST /deep_research_full: full multi-stage system (scope → research → report)\n\n"
        "All routes call the compiled LangGraph graphs in-process (no assistants/threads\n"
        "HTTP API), keeping the surface area minimal for teaching context engineering."
    ),
)


@app.post("/deep_research", response_model=ResearchResponse)
async def deep_research(request: ResearchRequest) -> ResearchResponse:
    """Run the single research agent and return its compressed research summary.

    This uses the middle-stage research agent (web search + compression) defined
    in `research_agent.py`, which is often the most useful "context engineering"
    stage to showcase as a standalone API.
    """
    result = await researcher_agent.ainvoke(
        {
            "researcher_messages": [HumanMessage(content=request.query)],
            "tool_call_iterations": 0,
            "research_topic": request.query,
            "compressed_research": "",
            "raw_notes": [],
        },
        config={
            "configurable": {
                "thread_id": "api-deep-research",
                "recursion_limit": 5,
            }
        },
    )

    # `compressed_research` is the synthesized summary of all tool calls.
    final_report = result.get("compressed_research") or ""
    return ResearchResponse(final_report=final_report)


@app.post("/deep_research_mcp", response_model=ResearchResponse)
async def deep_research_mcp(request: ResearchRequest) -> ResearchResponse:
    """Run the MCP-backed research agent and return its compressed research summary.

    This uses the MCP variant of the research agent (`research_agent_mcp.py`),
    which integrates with an MCP filesystem server for local document research.
    """
    result = await agent_mcp.ainvoke(
        {
            "researcher_messages": [HumanMessage(content=request.query)],
            "tool_call_iterations": 0,
            "research_topic": request.query,
            "compressed_research": "",
            "raw_notes": [],
        },
        config={
            "configurable": {
                "thread_id": "api-deep-research-mcp",
                "recursion_limit": 5,
            }
        },
    )

    final_report = result.get("compressed_research") or ""
    return ResearchResponse(final_report=final_report)


@app.post("/deep_research_full", response_model=ResearchResponse)
async def deep_research_full(request: ResearchRequest) -> ResearchResponse:
    """Run the full multi-stage deep research graph and return the final report."""
    # The full graph expects a LangChain-style message history.
    result = await agent.ainvoke(
        {
            "messages": [HumanMessage(content=request.query)],
        },
        config={
            "configurable": {
                # Higher recursion limit for complex research, as in the notebook.
                "thread_id": "api-deep-research",
                "recursion_limit": 50,
            }
        },
    )

    final_report = result.get("final_report") or ""
    return ResearchResponse(final_report=final_report)



