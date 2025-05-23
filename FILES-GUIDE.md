# 📋 Project Files Guide

## 🔧 Configuration Files

### Core Configuration Files
- **`config.py`** - Main configuration file (only file you need to modify!)
- **`SIMPLE-SETUP.md`** - Simple setup guide
- **`QUICKSTART.md`** - Quick start guide

### Optional Files
- **`.env.example`** - Environment variable examples (for advanced users)
- **`README-SETUP.md`** - Detailed setup guide
- **`check-config.bat`** - Windows configuration validation tool

## 🚀 Main Program Files

- **`main.py`** - Main executable file
- **`mcp_server.py`** - MCP server (for Claude Desktop connection)
- **`milvus_manager.py`** - Milvus database management
- **`obsidian_processor.py`** - Obsidian file processing
- **`embeddings.py`** - Embedding model management
- **`search_engine.py`** - Basic search engine
- **`enhanced_search_engine.py`** - Advanced search engine

## 📁 Auto-Generated Folders

- **`volumes/`** - Milvus database file storage
- **`__pycache__/`** - Python cache files
- **`.mypy_cache/`** - Type checking cache

## 🔧 Container Configuration Files

- **`milvus-podman-compose.yml`** - Podman configuration
- **`claude_desktop_config.json`** - Claude Desktop configuration

## 📊 Monitoring & Utilities

- **`progress_monitor_cmd.py`** - Progress monitoring
- **`watcher.py`** - File change detection
- **`check_gpu.py`** - GPU status checking

---

## 🎯 What Regular Users Should Touch

**Only one thing**: Modify `OBSIDIAN_VAULT_PATH` at the top of `config.py`!

Don't touch the other files. 😊
