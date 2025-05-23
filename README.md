# 🚀 Obsidian-Milvus-FastMCP

**A powerful, production-ready system that connects your Obsidian vault to Claude Desktop via FastMCP, leveraging Milvus vector database for intelligent document search and retrieval.**

## ✨ Features

- 🔍 **Intelligent Search**: Advanced vector similarity search with Milvus
- 🧠 **Claude Integration**: Seamless connection to Claude Desktop via FastMCP
- 📚 **Knowledge Graph**: Explore connections between your notes
- ⚡ **GPU Acceleration**: Optimized for high-performance embedding generation
- 🔄 **Real-time Sync**: Automatic file change detection and reindexing
- 🌐 **Multilingual Support**: Works with notes in multiple languages
- 🎯 **Advanced RAG**: Sophisticated retrieval-augmented generation

## 🚀 Quick Start

### 1. Edit Configuration
Open `config.py` and modify the vault path:

```python
# 🗂️ Obsidian Vault Path (REQUIRED!)
# Change the path below to your Obsidian vault path
OBSIDIAN_VAULT_PATH = "C:\\Users\\YourName\\Documents\\My Obsidian Vault"  # ← Change this!
```

### 2. Verify Setup
```bash
python config.py
```

### 3. Run the Program
```bash
python main.py
```

### 4. Start MCP Server
From the menu, select `1. Start MCP Server`

**That's it!** No complex environment setup required. 🎉

## 📋 Requirements

- **Python 3.8+**
- **Podman** (for Milvus containers)
- **CUDA-compatible GPU** (optional, for acceleration)
- **Obsidian vault** with markdown files

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd obsidian-milvus-FastMCP
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Podman**
   - Windows: Download from [Podman.io](https://podman.io/getting-started/installation)
   - Mac: `brew install podman`
   - Linux: Check your distribution's package manager

4. **Configure paths**
   - Edit `config.py` and set your Obsidian vault path
   - Run `python config.py` to verify

5. **Interactive Setup & Testing**
   ```bash
   python test_mcp.py
   ```
   
   This interactive script will guide you through:
   1. **Package Installation** - Installs required dependencies
   2. **Milvus Setup** - Automatically deploys Milvus using Podman with persistent data storage [SAFE_MILVUS_README.md](SAFE_MILVUS_README.md)
   3. **Collection Testing** - Verifies database operations
   4. **MCP Server Validation** - Checks server files
   5. **Claude Desktop Integration** - Configures Claude Desktop automatically

6. **Document Processing & Embedding**
   ```bash
   python main.py
   # Select option 2 or 3 to start embedding your documents
   ```

7. **Launch MCP Server**
   ```bash
   python main.py
   # Select option 1 and keep this running
   ```


## 🎮 Usage
##### Once setup is complete:
- Keep MCP Server Running: python main.py (option 1)
- Open Claude Desktop: Your Obsidian vault is now searchable
- Start Searching: Use natural language queries in Claude Desktop


## 🎯 Path Configuration Examples

### Windows
```python
OBSIDIAN_VAULT_PATH = "C:\\Users\\JohnDoe\\Documents\\My Obsidian Vault"
```

### macOS
```python
OBSIDIAN_VAULT_PATH = "/Users/johndoe/Documents/My Obsidian Vault"
```

### Linux
```python
OBSIDIAN_VAULT_PATH = "/home/johndoe/Documents/obsidian-vault"
```

## 🔍 Advanced Configuration

For advanced users, you can use environment variables by creating a `.env` file:

```bash
cp .env.example .env
# Edit .env with your settings
```

## 📊 Performance Optimization

- **GPU Acceleration**: Automatically detected and enabled
- **Batch Processing**: Optimized for large vaults
- **Memory Management**: Intelligent resource usage
- **Caching**: Smart caching for faster retrieval

## 🧠 AI Features

- **Vector Embeddings**: Multi-language support with state-of-the-art models
- **Semantic Search**: Find related content by meaning, not just keywords
- **Knowledge Graph**: Discover connections between your notes
- **Advanced RAG**: Context-aware retrieval for better AI responses

## 📁 File Structure

```
obsidian-milvus-FastMCP/
├── config.py                 # 🔧 Main configuration (edit this!)
├── QUICKSTART.md             # ⚡ 30-second setup guide
├── SIMPLE-SETUP.md           # 📖 Detailed setup instructions
├── main.py                   # 🚀 Main program
├── mcp_server.py            # 🔌 MCP server for Claude
├── test_mcp.py              # 🧪 Interactive setup & testing
├── volumes/                  # 📁 Auto-created data storage
└── ...                       # Other program files
```

## 🔧 Available Commands

From the main menu:
1. **Start MCP Server** - Connect to Claude Desktop
2. **Full Embedding** - Reindex all files (first-time setup)
3. **Incremental Embedding** - Update only changed files
4. **Exit** - Quit the program

## 🚨 Troubleshooting

### "Obsidian vault path does not exist"
→ Check and correct the path in `config.py`

### "Podman not found"
→ Install Podman or set `PODMAN_PATH` in `config.py`

### GPU not detected
→ Set `USE_GPU = False` in `config.py` to use CPU mode

### Performance issues
→ Adjust `BATCH_SIZE` and `EMBEDDING_BATCH_SIZE` in `config.py`

## 🎯 Claude Desktop Integration

After starting the MCP server, add this to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "obsidian-assistant": {
      "command": "python",
      "args": ["path/to/mcp_server.py"],
      "env": {}
    }
  }
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Milvus**: High-performance vector database
- **Claude**: Advanced AI assistant
- **FastMCP**: Efficient MCP implementation
- **Sentence Transformers**: Multilingual embedding models

---

## 💡 Tips for Best Results

1. **Organize your vault**: Well-structured notes yield better results
2. **Use descriptive titles**: Helps with semantic search
3. **Add tags**: Enhances filtering and discovery
4. **Regular updates**: Run incremental embedding after adding new notes
5. **GPU acceleration**: Significant performance boost for large vaults

**Happy knowledge exploring!** 🚀📚
