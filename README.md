# 🚀 Obsidian-Milvus-FastMCP

**A powerful, production-ready system that connects your Obsidian vault to Claude Desktop via FastMCP, leveraging Milvus vector database for intelligent document search and retrieval.**

## ✨ Features

### 🔍 **Core Search Capabilities**
- **Hybrid Search Engine**: Advanced vector similarity + keyword search fusion
- **Intelligent Semantic Search**: High-precision meaning-based document retrieval
- **Adaptive Query Processing**: Automatic parameter adjustment based on query complexity
- **Multi-modal Search**: Integrated text and attachment file search
- **Contextual Expansion**: Related document discovery and context-aware retrieval

### 🧠 **Advanced AI & RAG Features**
- **Hierarchical Retrieval**: Document → Section → Chunk progressive search
- **Multi-query Fusion**: Intelligent combination of multiple search queries with weighted averaging, maximum value, and reciprocal rank fusion
- **Adaptive Chunk Retrieval**: Dynamic chunk size adjustment based on query complexity
- **Knowledge Graph Exploration**: Vector similarity-based connection discovery with BFS traversal and graph centrality ranking
- **Temporal-aware Search**: Balance between relevance and recency with time-weighted scoring

### 🏷️ **Advanced Metadata Filtering**
- **Complex Tag Logic**: AND/OR/NOT combinations for sophisticated tag-based filtering
- **Time Range Filtering**: Precise temporal document filtering
- **File Type & Quality Filtering**: Content quality assessment and file type categorization
- **Multi-dimensional Filtering**: Simultaneous application of multiple filter criteria

### ⚡ **Performance Optimization**
- **GPU/CPU Auto-selection**: Hardware-optimized index selection (GPU_IVF_FLAT, GPU_CAGRA, HNSW)
- **HNSW Index Optimization**: Dynamic parameter tuning (ef, nprobe) based on collection size
- **Real-time Performance Monitoring**: Live performance tracking and analysis
- **Adaptive Search Parameters**: Result quality-based parameter dynamic adjustment
- **Batch Processing Optimization**: Efficient large-scale search operations

### 📊 **System Intelligence**
- **Auto-tuning**: Collection size-based automatic parameter optimization
- **Performance Benchmarking**: Multi-strategy search performance comparison
- **Smart Recommendations**: Automatic optimization suggestions based on usage patterns
- **Resource Management**: Intelligent memory and GPU utilization

### 🌐 **Integration & Connectivity**
- **Claude Desktop Integration**: Seamless connection via FastMCP protocol
- **Real-time File Monitoring**: Automatic change detection and incremental reindexing
- **Multilingual Support**: Advanced embedding models for global language support
- **Container-based Deployment**: Reliable Podman-based Milvus deployment


## 📋 Requirements

- **Python 3.8+**
- **Podman** (for Milvus containers)
- **CUDA-compatible GPU** (optional, for acceleration)
- **Obsidian vault** with markdown files

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/jayjeo/obsidian-milvus-FastMCP
   cd obsidian-milvus-fastmcp
   ```

2. **Configure paths**
   - Edit `config.py` and set your Obsidian vault path
   - Edit mamba path and conda path to install required packages
     - Run `detect-paths.bat` to find paths automatically

3. **Install dependencies**
   ```bash
   CMD at your directory, 
   conda install -c conda-forge -y python pip
   conda run -n base pip install pymilvus mcp fastmcp sentence-transformers torch
   conda run -n base pip install PyPDF2 markdown beautifulsoup4 python-dotenv watchdog psutil colorama pyyaml tqdm requests
   ```

4. **Install Podman**
   - Windows: Download from [Podman.io](https://podman.io/getting-started/installation)
   - Mac: `brew install podman`
   - Linux: Check your distribution's package manager
   **Run podman** : If you cannot find the path, run `find-podman-desktop.bat`

5. **Interactive Setup & Testing**
   ```bash
   run-setup.bat
   ```
   
   This interactive script will guide you through:
   1. **Package Installation** - Installs required dependencies
   2. **Milvus Setup** - Automatically deploys Milvus using Podman with persistent data storage [SAFE_MILVUS_README.md](SAFE_MILVUS_README.md)
   3. **Collection Testing** - Verifies database operations
   4. **MCP Server Validation** - Checks server files
   5. **Claude Desktop Integration** - Configures Claude Desktop automatically

6. **Document Processing & Embedding**
   ```bash
   run-main.bat
   # Select option 2 or 3 to start embedding your documents
   ```

7. **Launch MCP Server**
   ```bash
   run-main.bat
   # Select option 1 and keep this running
   ```

## 🎮 Usage

##### Once setup is complete:
- **Keep MCP Server Running**: `python main.py` (option 1)
- **Open Claude Desktop**: Your Obsidian vault is now searchable
- **Start Searching**: Use natural language queries in Claude Desktop
- **Advanced Features**: Access all search modes and filtering options through Claude

##### Backup & Restore Milvus Data (Embedding Data)

- **Backup**: Run `backup-all-data.bat`
- **Restore**: Run `restore-backup.bat` 
- **Reset Container**: Run `safe-container-reset.bat` (This is safe for preserving Milvus data)  

### ⚠️ **EMERGENCY RESET**

**If you encounter container conflicts or system issues**, you can use the emergency reset script:

```bash
complete-reset.bat  # Windows
```

**⚠️ CRITICAL WARNING**: This script will:
- **Kill ALL Podman containers** (not just Milvus)
- **Remove ALL Podman containers, pods, volumes, and networks**
- **Delete local MilvusData and volumes directories**
- **Permanently destroy all container data system-wide**

**Use ONLY if:**
- You have container name conflicts
- Milvus services fail to start properly
- You need a complete clean state
- **You don't have other important Podman containers running**

**Before running**: Make sure you don't have other Docker/Podman projects running that you want to preserve. This reset affects the **entire Podman system**, not just this project.

**Alternative**: Use `emergency-reset.bat` for a more targeted cleanup that focuses primarily on Milvus containers while being safer for other containers.


## 🎯 Advanced Search Capabilities

### **Available Search Modes**
- **Intelligent Search**: Adaptive strategy selection with context expansion
- **Power Search**: Maximum optimization with GPU acceleration
- **Multi-query Fusion**: Combine multiple queries for enhanced accuracy
- **Knowledge Graph Search**: Explore semantic connections between documents
- **Hierarchical Search**: Progressive document exploration

### **Filtering Options**
- **Temporal Filtering**: Time range-based document selection
- **Tag-based Filtering**: Complex boolean logic for tag combinations
- **Content Quality Filtering**: Quality score-based document ranking
- **File Type Filtering**: Specific format-based search restriction

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


## 🚨 Troubleshooting

### "Obsidian vault path does not exist"
→ Check and correct the path in `config.py`

### "Podman not found"
→ Install Podman or set `PODMAN_PATH` in `config.py`

### GPU not detected
→ Set `USE_GPU = False` in `config.py` to use CPU mode

### Performance issues
→ Adjust `BATCH_SIZE` and `EMBEDDING_BATCH_SIZE` in `config.py`

### Search quality issues
→ Try different search strategies or adjust similarity thresholds

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

- **Milvus**: High-performance vector database with advanced indexing
- **Claude**: Advanced AI assistant with MCP protocol support
- **FastMCP**: Efficient MCP implementation framework
- **Sentence Transformers**: State-of-the-art multilingual embedding models
- **HNSW Algorithm**: Hierarchical Navigable Small World graphs for fast similarity search

---

**Contact**: [acubens555@gmail.com](mailto:acubens555@gmail.com)  
**Author**: Jay Jeong, Ph.D. in Economics, Research Fellow at KCTDI
