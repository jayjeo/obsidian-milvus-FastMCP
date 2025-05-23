try:
    from mcp.server.fastmcp import FastMCP
    print('[OK] FastMCP module found')
except Exception as e:
    print(f'[ERROR] FastMCP module issue: {e}')
