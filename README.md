# 🚀 Obsidian-Milvus-FastMCP

**A powerful, production-ready system that connects your Obsidian vault to Claude Desktop via FastMCP, leveraging Milvus vector database for intelligent document search and retrieval.**

## ✨ Features

### 📝 **Video Showcase**
- [https://youtu.be/wPFiG9mC7e8?si=uF-TJrgG-guC33JG](https://youtu.be/wPFiG9mC7e8?si=uF-TJrgG-guC33JG)

### 🔍 **Core Search Capabilities**
- <span style="color: green">**Hybrid Search Engine**</span> : Advanced vector similarity + keyword search fusion
- <span style="color: green">**Intelligent Semantic Search**</span> : High-precision meaning-based document retrieval
- <span style="color: green">**Adaptive Query Processing**</span> : Automatic parameter adjustment based on query complexity
- <span style="color: green">**Multi-modal Search**</span> : Integrated text and attachment file search
- <span style="color: green">**Contextual Expansion**</span> : Related document discovery and context-aware retrieval

### 🧠 **Advanced AI & RAG Features**
- <span style="color: green">**Hierarchical Retrieval**</span> : Document → Section → Chunk progressive search
- <span style="color: green">**Multi-query Fusion**</span> : Intelligent combination of multiple search queries with weighted averaging, maximum value, and reciprocal rank fusion
- <span style="color: green">**Adaptive Chunk Retrieval**</span> : Dynamic chunk size adjustment based on query complexity
- <span style="color: green">**Knowledge Graph Exploration**</span> : Vector similarity-based connection discovery with BFS traversal and graph centrality ranking
- <span style="color: green">**Temporal-aware Search**</span> : Balance between relevance and recency with time-weighted scoring

### 🏷️ **Advanced Metadata Filtering**
- <span style="color: green">**Complex Tag Logic**</span> : AND/OR/NOT combinations for sophisticated tag-based filtering
- <span style="color: green">**Time Range Filtering**</span> : Precise temporal document filtering
- <span style="color: green">**File Type & Quality Filtering**</span> : Content quality assessment and file type categorization
- <span style="color: green">**Multi-dimensional Filtering**</span> : Simultaneous application of multiple filter criteria

### ⚡ **Performance Optimization**
- <span style="color: green">**GPU/CPU Auto-selection**</span> : Hardware-optimized index selection (GPU_IVF_FLAT, GPU_CAGRA, HNSW)
- <span style="color: green">**HNSW Index Optimization**</span> : Dynamic parameter tuning (ef, nprobe) based on collection size
- <span style="color: green">**Real-time Performance Monitoring**</span> : Live performance tracking and analysis
- <span style="color: green">**Adaptive Search Parameters**</span> : Result quality-based parameter dynamic adjustment
- <span style="color: green">**Batch Processing Optimization**</span> : Efficient large-scale search operations

### 📊 **System Intelligence**
- <span style="color: green">**Auto-tuning**</span> : Collection size-based automatic parameter optimization
- <span style="color: green">**Performance Benchmarking**</span> : Multi-strategy search performance comparison
- <span style="color: green">**Smart Recommendations**</span> : Automatic optimization suggestions based on usage patterns
- <span style="color: green">**Resource Management**</span> : Intelligent memory and GPU utilization

### 🌐 **Integration & Connectivity**
- <span style="color: green">**Claude Desktop Integration**</span> : Seamless connection via FastMCP protocol
- <span style="color: green">**Real-time File Monitoring**</span> : Automatic change detection and incremental reindexing
- <span style="color: green">**Multilingual Support**</span> : Advanced embedding models for global language support
- <span style="color: green">**Container-based Deployment**</span> : Reliable Podman-based Milvus deployment


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

6. **Podman auto launch at startup**
   - Follow instructions in [Podman auto launch.md](https://share.note.sx/r6kx06pj#78CIGnxLJYkJG+ZrQKYQhU35gtl+nKa47ZllwEyfUE0)

7. **Milvus Server auto launch at startup**
   - Follow instructions in [Milvus auto launch.md](https://share.note.sx/y9vrzgj6#zr1aL4s1WFBK/A4WvqvkP6ETVMC4sKcAwbqAt4NyZhk)
   - When Windows starts, a CMD will pop up, but you may close it
   - A manual execution alternative is available, though this method proves operationally cumbersome
     - `run-main.bat`, and select option 1
     - You have to keep this CMD opened. Otherwise, the server will be terminated

8.  **Interactive Setup & Testing**
   ```bash
   run-setup.bat
   ```
   - (1) **Package Installation** - Installs required dependencies
   - (2) **Milvus Setup** - Automatically deploys Milvus using Podman with persistent data storage
   - (3) **Collection Testing** - Verifies database operations
   - (4) **MCP Server Validation** - Checks server files
   - (5) **Claude Desktop Integration** - Configures Claude Desktop automatically
   - (8) **Safe Server Restart** (Preserve All Data. Use this if MCP server has launching issues)
   - (9) **Emergency: Complete Data Reset** (DELETE All Data)

## 🎮 Daily Use

##### Once setup is complete:
- **Start Embedding (Indexing) your Obsidian vault**: Run `run-main.bat` and select option 2 or 3 
- **Open Claude Desktop**: Your Obsidian vault is now searchable
- **Start Searching**: Use natural language queries in Claude Desktop

## 🚨 GPU not detected issue even though my GPU exists and supports CUDA
- Make sure set `USE_GPU = True` in `config.py` to use GPU mode
- Pytorch has two different versions. One is for CPU and the other is for GPU. Make sure you have the correct version installed. If you arn't sure, run `pytorch_gpu_installer.bat` to install the correct version.


## Backup & Restore Milvus Data (Embedding Data)
- <span style="color: green">**Backup**</span> : Run `backup-all-data.bat`
- <span style="color: green">**Restore**</span> : Run `restore-backup.bat` 


## MCP Search Tools that you can see at Claude Desktop
##### Basic Search
- <span style="color: green">**search_documents**</span> : Enhanced Obsidian document search
- <span style="color: green">**get_document_content**</span> : Retrieve complete content of specific document
- <span style="color: green">**get_similar_documents**</span> : Find similar documents

##### Intelligent Search
- <span style="color: green">**auto_search_mode_decision**</span> : Automatic search mode determination
- <span style="color: green">**intelligent_search**</span> : Advanced intelligent search
- <span style="color: green">**intelligent_search_enhanced**</span> : Enhanced intelligent search
- <span style="color: green">**comprehensive_search_all**</span> : Comprehensive search across entire collection

##### Advanced Search
- <span style="color: green">**batch_search_with_pagination**</span> : Batch search with pagination
- <span style="color: green">**advanced_filter_search**</span> : Advanced metadata filter search
- <span style="color: green">**multi_query_fusion_search**</span> : Multi-query fusion search
- <span style="color: green">**milvus_power_search**</span> : Milvus optimized power search

##### Knowledge Graph
- <span style="color: green">**knowledge_graph_exploration**</span> : Knowledge graph exploration
- <span style="color: green">**milvus_knowledge_graph_builder**</span> : Milvus-based knowledge graph construction

##### Tag Search
- <span style="color: green">**search_by_tags**</span> : Search documents by tags
- <span style="color: green">**list_available_tags**</span> : List available tags

##### System Management
- <span style="color: green">**get_collection_stats**</span> : Collection statistics information
- <span style="color: green">**performance_optimization_analysis**</span> : Performance optimization analysis
- <span style="color: green">**milvus_system_optimization_report**</span> : Comprehensive system optimization report

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


## <span style="color: red">**EMERGENCY RESET**</span>

**If you encounter container conflicts or system issues**, you can use the emergency reset script:

```bash
complete-reset.bat  # Windows
```

<span style="color: red">**CRITICAL WARNING**</span>: This script will:
- **Kill ALL Podman containers** (not just Milvus)
- **Remove ALL Podman containers, pods, volumes, and networks**
- **Delete local MilvusData and volumes directories**
- **Permanently destroy all container data system-wide**

<span style="color: red">**Use ONLY if**</span>
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

- <span style="color: blue">**Milvus**</span>: High-performance vector database with advanced indexing
- <span style="color: blue">**Claude**</span>: Advanced AI assistant with MCP protocol support
- <span style="color: blue">**FastMCP**</span>: Efficient MCP implementation framework
- <span style="color: blue">**Sentence Transformers**</span>: State-of-the-art multilingual embedding models
- <span style="color: blue">**HNSW Algorithm**</span>: Hierarchical Navigable Small World graphs for fast similarity search

---
 
<span style="color: green">**Author**</span>: Jay Jeong, Ph.D. in Economics, Research Fellow at KCTDI, [acubens555@gmail.com](mailto:acubens555@gmail.com) 
