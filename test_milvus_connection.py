try:
    import config
    from milvus_manager import MilvusManager
    print('[TEST] Initializing Milvus manager...')
    manager = MilvusManager()
    print('[OK] Milvus connection successful')
except Exception as e:
    print(f'[WARNING] Milvus connection issue: {e}')
    print('Please check if Milvus is running.')
