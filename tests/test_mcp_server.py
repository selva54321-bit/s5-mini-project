import asyncio

from app.mcp_server import mcp


def test_mcp_server_registers_project_intelligence_tools():
    async def list_tool_names() -> set[str]:
        return {tool.name for tool in await mcp.list_tools()}

    names = asyncio.run(list_tool_names())
    assert {
        "list_projects",
        "get_project_overview",
        "query_project_graph",
        "get_repository_evidence",
        "create_project",
        "ingest_repository",
        "ingest_project_documents",
    } <= names
