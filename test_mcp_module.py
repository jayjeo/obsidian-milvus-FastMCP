try:
    import mcp
    print('[OK] mcp module found')
except Exception as e:
    print(f'[ERROR] mcp module missing: {e}')
