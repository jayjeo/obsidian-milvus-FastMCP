#!/usr/bin/env python3
"""Simplified MCP Server Test to diagnose tool registration issues"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

# Set environment variables to suppress output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from mcp.server.fastmcp import FastMCP
import asyncio

# Create FastMCP instance
mcp = FastMCP("test-obsidian")

# Test simple tools first
@mcp.tool()
async def simple_search(query: str) -> dict:
    """Simple search test"""
    return {"query": query, "results": ["result1", "result2"]}

@mcp.tool()
def get_stats() -> dict:
    """Get simple stats"""
    return {"documents": 100, "status": "ok"}

# Import config but catch any errors
try:
    import config
    print(f"Config loaded successfully. Collection: {config.COLLECTION_NAME}")
except Exception as e:
    print(f"Config error: {e}")
    config = None

# Test with conditional imports
try:
    # Import the actual mcp_server to see if tools are registered
    print("\nImporting mcp_server...")
    import mcp_server
    print("mcp_server imported successfully")
    
    # Check if mcp instance exists
    if hasattr(mcp_server, 'mcp'):
        print(f"Found mcp instance: {mcp_server.mcp}")
        test_mcp = mcp_server.mcp
    else:
        print("No mcp instance found in mcp_server")
        test_mcp = mcp
        
except Exception as e:
    print(f"Error importing mcp_server: {e}")
    import traceback
    traceback.print_exc()
    test_mcp = mcp

# Check registered tools
async def check_tools():
    """Check which tools are registered"""
    print("\n" + "="*50)
    print("CHECKING REGISTERED TOOLS")
    print("="*50)
    
    try:
        # Check simple test tools
        simple_tools = await mcp.list_tools()
        print(f"\nSimple test MCP tools: {len(simple_tools)}")
        for tool in simple_tools:
            print(f"  - {tool.name}")
        
        # Check actual mcp_server tools
        if test_mcp != mcp:
            actual_tools = await test_mcp.list_tools()
            print(f"\nActual mcp_server tools: {len(actual_tools)}")
            for i, tool in enumerate(actual_tools[:10]):
                print(f"  {i+1}. {tool.name}")
            if len(actual_tools) > 10:
                print(f"  ... and {len(actual_tools) - 10} more")
            
            # Check detailed info for first tool
            print("\nDetailed info for first tool:")
            if actual_tools:
                first_tool = actual_tools[0]
                print(f"Name: {first_tool.name}")
                print(f"Description: {first_tool.description}")
                print(f"Input Schema: {first_tool.inputSchema}")
                
            # Check if there's an async_wrapper tool
            print("\nSearching for 'async_wrapper' tool:")
            async_wrapper_found = False
            for tool in actual_tools:
                if 'async' in tool.name.lower() or 'wrapper' in tool.name.lower():
                    print(f"Found: {tool.name} - {tool.description}")
                    async_wrapper_found = True
            if not async_wrapper_found:
                print("No 'async_wrapper' tool found in registered tools")
        
    except Exception as e:
        print(f"Error listing tools: {e}")
        import traceback
        traceback.print_exc()

# Main execution
if __name__ == "__main__":
    asyncio.run(check_tools())
