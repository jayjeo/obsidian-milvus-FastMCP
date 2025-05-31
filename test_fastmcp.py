#!/usr/bin/env python3
"""FastMCP Test File to verify tool registration"""

from mcp.server.fastmcp import FastMCP
import json

# Create FastMCP instance
mcp = FastMCP("test-server")

# Define a simple tool
@mcp.tool()
async def test_tool_1(message: str) -> dict:
    """Test tool 1"""
    return {"message": f"Test 1: {message}"}

@mcp.tool()
async def test_tool_2(number: int) -> dict:
    """Test tool 2"""
    return {"result": number * 2}

@mcp.tool()
def test_tool_3(name: str) -> dict:
    """Test tool 3 (sync)"""
    return {"greeting": f"Hello, {name}!"}

# Test if tools are registered
if __name__ == "__main__":
    print("FastMCP Test")
    print("="*50)
    
    # Check registered tools
    if hasattr(mcp, '_tool_manager'):
        print(f"Tool manager found: {mcp._tool_manager}")
        if hasattr(mcp._tool_manager, 'tools'):
            print(f"Registered tools: {list(mcp._tool_manager.tools.keys())}")
    elif hasattr(mcp, 'tools'):
        print(f"Direct tools found: {list(mcp.tools.keys())}")
    elif hasattr(mcp, '_tools'):
        print(f"_tools found: {list(mcp._tools.keys())}")
    else:
        print("No tools attribute found")
        
    # Try to find tools in different ways
    print("\nSearching for tools in mcp object:")
    for attr in dir(mcp):
        if 'tool' in attr.lower():
            print(f"  - {attr}: {type(getattr(mcp, attr))}")
    
    print("\nMCP object attributes:")
    for attr in dir(mcp):
        if not attr.startswith('_'):
            print(f"  - {attr}")
    
    # List tools using list_tools method
    print("\nTrying to list tools:")
    try:
        import asyncio
        async def list_registered_tools():
            tools = await mcp.list_tools()
            print(f"Registered tools: {tools}")
            return tools
        
        tools = asyncio.run(list_registered_tools())
        print(f"\nTool count: {len(tools) if tools else 0}")
        if tools:
            for tool in tools:
                print(f"  - {tool}")
    except Exception as e:
        print(f"Error listing tools: {e}")
        import traceback
        traceback.print_exc()
