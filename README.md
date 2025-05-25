# 🚀 Obsidian-Milvus-FastMCP

**A powerful, production-ready system that connects your Obsidian vault to Claude Desktop via FastMCP, leveraging Milvus vector database for intelligent document search and retrieval.**

## ✨ Features

### 📝 **Video Showcase**
- [https://youtu.be/wPFiG9mC7e8?si=uF-TJrgG-guC33JG](https://youtu.be/wPFiG9mC7e8?si=uF-TJrgG-guC33JG)

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

3. **Install dependencies**
   - Install conda([anaconda](https://www.anaconda.com/download))
   - Do the following:
    ```bash
    Open Anaconda Prompt with administrator privileges
    cd to your directory
    conda install -c conda-forge -y python pip
    conda run -n base pip install pymilvus mcp fastmcp sentence-transformers torch
    conda run -n base pip install PyPDF2 markdown beautifulsoup4 python-dotenv watchdog psutil colorama pyyaml tqdm requests
    ```

4. **Install Podman**
   - From CMD: winget install RedHat.Podman
   - Open PowerShell as Administrator and run (Enable Virtual Machine)
     ```
     dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
     ```
   - Open PowerShell as Administrator and run (Linux kernel update package)
     ```
     wsl.exe --install
     ```
   - Set WSL 2 as your default version
     ```
     wsl --set-default-version 2
     ```
   - Install your Linux distribution of choice
     - Ubuntu 18.04 LTS
     - Ubuntu 20.04 LTS
     - Ubuntu 22.04 LTS
     - and so on
   - Restart after setting up WSL Linux system

5. **Configure paths**
   - Edit `config.py` and set your Obsidian vault path
   - Edit `config.py` and set your podman path
     - Find podman path using `find_podman_path.bat`

6. **Interactive Setup & Testing**
   ```bash
   run-setup.bat
   ```

   - (0) **Podman auto launch at startup** - Sets up automatic Podman startup at Windows boot
   - (1) **Package Installation** - Installs required dependencies
   - (2) **Milvus Setup** - Automatically deploys Milvus using Podman with persistent data storage
   - (3) **Collection Testing** - Verifies database operations
   - (4) **MCP Server Validation** - Checks server files
   - (5) **Claude Desktop Integration** - Configures Claude Desktop automatically
   - (8) **Safe Server Restart** (Preserve All Data. Use this if MCP server has launching issues)
   - (9) **Emergency: Complete Data Reset** (DELETE All Data)

7. **Keep MCP Server Running**: 
  - There are two ways for doing this. 
    - (1) manually run `run-main.bat`, and select option 1
    - (2) run `setup_windows_startup.bat`: This will set up automatic startup at Windows boot, so you don't have to manually run `run-main.bat`



## 🚨 GPU not detected issue even though my GPU exists and supports CUDA
- Make sure set `USE_GPU = True` in `config.py` to use GPU mode
- Pytorch has two different versions. One is for CPU and the other is for GPU. Make sure you have the correct version installed. If you arn't sure, run `pytorch_gpu_installer.bat` to install the correct version.

## 🎮 Daily Use

##### Once setup is complete:
- **Start Embedding (Indexing) your Obsidian vault**: Run `run-main.bat` and select option 2 or 3 
- **Open Claude Desktop**: Your Obsidian vault is now searchable
- **Start Searching**: Use natural language queries in Claude Desktop
- **Advanced Features**: Access all search modes and filtering options through Claude

## Backup & Restore Milvus Data (Embedding Data)
- **Backup**: Run `backup-all-data.bat`
- **Restore**: Run `restore-backup.bat` 


## MCP Search Tools that you can see at Claude Desktop
##### Basic Search
- **search_documents**: Enhanced Obsidian document search
- **get_document_content**: Retrieve complete content of specific document
- **get_similar_documents**: Find similar documents

##### Intelligent Search
- **auto_search_mode_decision**: Automatic search mode determination
- **intelligent_search**: Advanced intelligent search
- **intelligent_search_enhanced**: Enhanced intelligent search
- **comprehensive_search_all**: Comprehensive search across entire collection

##### Advanced Search
- **batch_search_with_pagination**: Batch search with pagination
- **advanced_filter_search**: Advanced metadata filter search
- **multi_query_fusion_search**: Multi-query fusion search
- **milvus_power_search**: Milvus optimized power search

##### Knowledge Graph
- **knowledge_graph_exploration**: Knowledge graph exploration
- **milvus_knowledge_graph_builder**: Milvus-based knowledge graph construction

##### Tag Search
- **search_by_tags**: Search documents by tags
- **list_available_tags**: List available tags

##### System Management
- **get_collection_stats**: Collection statistics information
- **performance_optimization_analysis**: Performance optimization analysis
- **milvus_system_optimization_report**: Comprehensive system optimization report

## Use Project feature in Claude Desktop for better search results
- For example, I use a project called "obsidian" to search my Obsidian vault. 
- My project instructions are as follows. 
  - Use only information from the obsidian assistant MCP
  - For general searches, always use the auto_search_mode_decision function
  - Tell me how many md or pdf files were searched and provide the list
  - Summarize the content comprehensively
  - Only save md files to obsidian when specifically requested
  - If saving, the location should be G:\jayjeo
  - When saving, create the note list in a clickable format for obsidian
  - Create a list of referenced md and pdf files in clickable obsidian format at the top of the note

## Use Filesystem MCP to save markdown notes to Obsidian
  - This MCP is amazingly helpful for saving and reading any files in your system. 
  - This includes saving markdown notes to your Obsidian vault
  - [https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem)


## ⚠️ **EMERGENCY RESET**

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

**Before running**: Make sure you don't have other Podman projects running that you want to preserve. This reset affects the **entire Podman system**, not just this project.



## 🎯 Claude Desktop Integration

- Claude Desktop configuration for this program is as follows. 
- However, this should be already set automatically by using `run-setup.bat` (option 5).

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


## Project Tree Structure
```
obsidian-milvus-FastMCP/
├── Main Application Files
│   ├── main.py                           # Main entry point with CLI menu system
│   ├── mcp_server.py                     # FastMCP server with advanced search tools
│   ├── config.py                         # Central configuration settings
│   ├── milvus_manager.py                 # Milvus database operations manager  
│   ├── obsidian_processor.py             # Obsidian file processing and embedding
│   ├── search_engine.py                  # Basic hybrid search engine
│   └── embeddings.py                     # Advanced embedding model with GPU optimization
│
├── Advanced Search Components
│   ├── enhanced_search_engine.py         # Advanced semantic and contextual search
│   ├── advanced_rag.py                   # RAG engine with knowledge graphs
│   ├── hnsw_optimizer.py                 # HNSW index optimization
│   └── watcher.py                        # Real-time file monitoring
│
├── Setup and Utilities
│   ├── setup.py                          # Interactive setup and testing system
│   ├── requirements.txt                  # Python dependencies
│   ├── README.md                         # Comprehensive documentation
│   └── LICENSE                           # MIT license file
│
├── Batch Scripts (Windows)
│   ├── run-main.bat                      # Launch main application
│   ├── run-setup.bat                     # Interactive setup wizard
│   ├── start-milvus.bat                  # Start Milvus containers
│   ├── stop-milvus.bat                   # Stop Milvus containers
│   └── [20+ other utility scripts]       # Various management and utility scripts
│
├── Container Configuration
│   ├── milvus-podman-compose.yml         # Podman compose for Milvus
│   └── milvus-docker-compose.yml         # Docker compose alternative
│
├── Data Directories (Created at runtime)
│   ├── MilvusData/                       # Persistent embedding data
│   └── volumes/                          # Container runtime data
│
└── Support Files
    ├── .env.example                      # Environment variables template
    ├── .gitignore                        # Git ignore patterns
    └── [Various helper scripts]          # Additional utilities
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
 
**Author**: Jay Jeong, Ph.D. in Economics, Research Fellow at KCTDI, [acubens555@gmail.com](mailto:acubens555@gmail.com) 
