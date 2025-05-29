# 🚀 Obsidian-Milvus-FastMCP

**A powerful, production-ready system that connects your Obsidian vault to Claude Desktop via FastMCP, leveraging Milvus vector database for intelligent document search and retrieval.** 

**Conventional MCPs (Notion MCP, Obsidian MCP) provide fast embedding calculations and convenient question-answering results. However, these results are limited to a level of convenience suitable for daily life. If a user has built a large volume of materials in obsidian and wants to read numerous notes and PDFs stored in obsidian through a single question to conduct in-depth analysis, conventional note program-based MCPs are not suitable. My obsidian-milvus-FastMCP was created to address this need. As a Ph.D. in Economics myself, I store a large amount of research materials in Obsidian and required comprehensive analytical results that correspond to my inquiries.**

**Warning! This program is extremely heavy and requires high memory (RAM) usage due to the need to keep the Milvus server running constantly, even when not performing embedding tasks. Therefore, it is not suitable for PCs with limited memory or laptops where power consumption needs to be conserved. Additionally, this program only supports Windows 10 and 11.**

## ✨ Features

### 📝 **Video Showcase**
- [https://youtu.be/wPFiG9mC7e8?si=uF-TJrgG-guC33JG](https://youtu.be/wPFiG9mC7e8?si=uF-TJrgG-guC33JG)

### 🔍 **Core Search Capabilities**
- 🟢 **Hybrid Search Engine** : Advanced vector similarity + keyword search fusion
- 🟢 **Intelligent Semantic Search** : High-precision meaning-based document retrieval
- 🟢 **Adaptive Query Processing** : Automatic parameter adjustment based on query complexity
- 🟢 **Multi-modal Search** : Integrated text and attachment file search
- 🟢 **Contextual Expansion** : Related document discovery and context-aware retrieval

### 🧠 **Advanced AI & RAG Features**
- 🟢 **Hierarchical Retrieval** : Document → Section → Chunk progressive search
- 🟢 **Multi-query Fusion** : Intelligent combination of multiple search queries with weighted averaging, maximum value, and reciprocal rank fusion
- 🟢 **Adaptive Chunk Retrieval** : Dynamic chunk size adjustment based on query complexity
- 🟢 **Knowledge Graph Exploration** : Vector similarity-based connection discovery with BFS traversal and graph centrality ranking
- 🟢 **Temporal-aware Search** : Balance between relevance and recency with time-weighted scoring

### 🏷️ **Advanced Metadata Filtering**
- 🟢 **Complex Tag Logic** : AND/OR/NOT combinations for sophisticated tag-based filtering
- 🟢 **Time Range Filtering** : Precise temporal document filtering
- 🟢 **File Type & Quality Filtering** : Content quality assessment and file type categorization
- 🟢 **Multi-dimensional Filtering** : Simultaneous application of multiple filter criteria

### ⚡ **Performance Optimization**
- 🟢 **GPU/CPU Auto-selection** : Hardware-optimized index selection (GPU_IVF_FLAT, GPU_CAGRA, HNSW)
- 🟢 **HNSW Index Optimization** : Dynamic parameter tuning (ef, nprobe) based on collection size
- 🟢 **Real-time Performance Monitoring** : Live performance tracking and analysis
- 🟢 **Adaptive Search Parameters** : Result quality-based parameter dynamic adjustment
- 🟢 **Batch Processing Optimization** : Efficient large-scale search operations

### 📊 **System Intelligence**
- 🟢 **Auto-tuning** : Collection size-based automatic parameter optimization
- 🟢 **Performance Benchmarking** : Multi-strategy search performance comparison
- 🟢 **Smart Recommendations** : Automatic optimization suggestions based on usage patterns
- 🟢 **Resource Management** : Intelligent memory and GPU utilization

### 🌐 **Integration & Connectivity**
- 🟢 **Claude Desktop Integration** : Seamless connection via FastMCP protocol
- 🟢 **Real-time File Monitoring** : Automatic change detection and incremental reindexing
- 🟢 **Multilingual Support** : Advanced embedding models for global language support
- 🟢 **Container-based Deployment** : Reliable Podman-based Milvus deployment


## 📋 Requirements

- **Python 3.8+**
- **Podman** (for Milvus containers)
- **CUDA-compatible GPU** (optional, for acceleration)
- **Obsidian vault** with markdown files

## 🔧 Installation
### Recommended Installation

1. Make sure git and anaconda (or miniconda) is installed

2. Download an installer from [here](https://www.dropbox.com/scl/fo/l4qtgiuajhrcqk6ejatdt/ANv-uIw78Q9hvbUzp5FEWHM?rlkey=8apdv9cwzptivbjaat39uwcfp&st=9esvs2et), and run it
   
   ```bash
   Obsidian_Milvus_Installer_AMD64.exe
   ```

3. 🚨 GPU not detected issue even though my GPU exists and supports CUDA
- If you do not have a CUDA supported GPU, you may skip this step
- Pytorch has two different versions. One is for CPU and the other is for GPU. Make sure you have the correct version installed. 
- If you arn't sure, follow the instruction below:
  - Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
  - Run the following commands:
  
    ```bash
    pip uninstall torch torchvision torchaudio -y
    pip install torch==2.7.0+cu118 torchvision==0.22.0+cu118 torchaudio==2.7.0+cu118 --index-url https://download.pytorch.org/whl/cu118
    ```

### Manual Installation

1. Make sure git is installed
   
2. **Clone the repository**
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
    pip uninstall -y numpy sentence-transformers
    pip install "numpy>2.0.0"
    pip install --no-deps transformers==4.52.3
    pip install sentence-transformers==4.1.0 tqdm filelock fsspec
    pip install PyPDF2 markdown beautifulsoup4 python-dotenv watchdog psutil colorama pyyaml tqdm requests pymilvus mcp fastmcp torch nvidia-ml-py 
    ```

    - Sentence Transformers Library (sentence-transformers): The v3.1.1 release announced in September 2024 removed the numpy<2 constraint that was previously set to prevent conflicts in Windows environments on GitHub. This means that the latest version (≥3.1.1) of Sentence Transformers officially supports NumPy 2.x, allowing users to freely choose between NumPy 1.x and 2.x.
    - In conclusion, the paraphrase-multilingual-mpnet-base-v2 model provided through the Sentence Transformers framework can operate normally in NumPy 2.x environments. This model is implemented based on Hugging Face Transformers and PyTorch, and as mentioned earlier, the combination of the latest Sentence Transformers version and PyTorch 2.5.1 has resolved compatibility issues with NumPy 2.x.
    - In fact, since the Sentence Transformers library officially supports NumPy 2.x on GitHub, and PyTorch 2.5.1 was also built to accommodate NumPy 2.x on GitHub, embedding extraction and other operations during model inference proceed without any additional errors. No differences based on NumPy version have been reported in either CPU environments or CUDA (GPU) accelerated environments, and since NumPy 2.x itself only affects CPU operations, CUDA usage is irrelevant to compatibility.
    - However, one thing to note is that the official requirements of the Hugging Face Transformers package still point to NumPy 1.x, which may cause warnings or conflicts during pip installation on GitHub. For example, when installing transformers via pip while NumPy 2.x is already installed, dependency conflict warnings may appear. However, this is merely an installation constraint and does not mean that the paraphrase-multilingual-mpnet-base-v2 model malfunctions due to NumPy 2.x at runtime. The model itself operates normally with the latest compatible library combinations, and there are no reports of embedding quality or accuracy changing based on NumPy version.
    - The dependency warnings/conflicts that occur when installing Hugging Face Transformers via pip can be bypassed. First, if you attempt pip install transformers while maintaining NumPy 2.x, warnings will appear, but you can skip dependency checks using the `--no-deps` option.


1. **Install Podman**
   - From CMD: winget install RedHat.Podman
   - Open PowerShell as Administrator and run (Enable Virtual Machine)

     ```powershell
     dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
     ```

   - Open PowerShell as Administrator and run (Linux kernel update package)

     ```powershell
     wsl.exe --install
     ```

   - Set WSL 2 as your default version

     ```powershell
     wsl --set-default-version 2
     ```

   - Install your Linux distribution of choice
     - Ubuntu 18.04 LTS
     - Ubuntu 20.04 LTS
     - Ubuntu 22.04 LTS
     - and so on
   - Restart after setting up WSL Linux system
   - `pip install podman-compose` using CMD at your path

2. **Configure paths**
   - Edit `config.py` and set your Obsidian vault path
   - Edit `config.py` and set your podman path
     - Find podman path using `find_podman_path.bat`

3. **Initialize Podman Container**

   ```
   complete-podman-reset.bat
   ```

4. **Initialize Milvus Server**

   ```
   start_mcp_with_encoding_fix.bat
   ```   

5. **Podman auto launch at startup**
   - Follow instructions in [Podman auto launch.md](https://share.note.sx/r6kx06pj#78CIGnxLJYkJG+ZrQKYQhU35gtl+nKa47ZllwEyfUE0)
   - When Windows starts, nothing will pop up unless there is an error
     - If you want to figure out the error, see `podman_startup.log`

6.  **Milvus Server auto launch at startup**
   - Follow instructions in [Milvus auto launch.md](https://share.note.sx/y9vrzgj6#zr1aL4s1WFBK/A4WvqvkP6ETVMC4sKcAwbqAt4NyZhk)
   - When Windows starts, nothing will pop up unless there is an error
     - If you want to figure out the error, see `auto_startup_mcp.log` and `vbs_startup.log`
     - Launching the server takes approximately 5~7 minites depending on your PC performance. Wait for it to finish before you start Claude Desktop
   - A manual execution alternative is available, though this method proves operationally cumbersome
     - `run-main.bat`, and select option 1
     - You have to keep this CMD opened. Otherwise, the server will be terminated

7.   **Interactive Setup & Testing**

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

8.  🚨 GPU not detected issue even though my GPU exists and supports CUDA
- If you do not have a CUDA supported GPU, you may skip this step
- Pytorch has two different versions. One is for CPU and the other is for GPU. Make sure you have the correct version installed. 
- If you arn't sure, follow the instruction below:
  - Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
  - Run the following commands:

    ```bash
    pip uninstall torch torchvision torchaudio -y
    pip install torch==2.7.0+cu118 torchvision==0.22.0+cu118 torchaudio==2.7.0+cu118 --index-url https://download.pytorch.org/whl/cu118
    ```


## 🎮 Daily Use

##### Once setup is complete:
- **Start Embedding (Indexing) your Obsidian vault**: Run `run-main.bat` and select option 2 or 3 
- **Open Claude Desktop**: Your Obsidian vault is now searchable
- **Start Searching**: Use natural language queries in Claude Desktop



## Backup & Restore Milvus Data (Embedding Data)
- 🟢 **Backup** : Run `backup-all-data.bat`
- 🟢 **Restore** : Run `restore-backup.bat` 


## MCP Search Tools that you can see at Claude Desktop
##### Basic Search
- 🟢 **search_documents** : Enhanced Obsidian document search
- 🟢 **get_document_content** : Retrieve complete content of specific document
- 🟢 **get_similar_documents** : Find similar documents

##### Intelligent Search
- 🟢 **auto_search_mode_decision** : Automatic search mode determination
- 🟢 **intelligent_search** : Advanced intelligent search
- 🟢 **intelligent_search_enhanced** : Enhanced intelligent search
- 🟢 **comprehensive_search_all** : Comprehensive search across entire collection

##### Advanced Search
- 🟢 **batch_search_with_pagination** : Batch search with pagination
- 🟢 **advanced_filter_search** : Advanced metadata filter search
- 🟢 **multi_query_fusion_search** : Multi-query fusion search
- 🟢 **milvus_power_search** : Milvus optimized power search

##### Knowledge Graph
- 🟢 **knowledge_graph_exploration** : Knowledge graph exploration
- 🟢 **milvus_knowledge_graph_builder** : Milvus-based knowledge graph construction

##### Tag Search
- 🟢 **search_by_tags** : Search documents by tags
- 🟢 **list_available_tags** : List available tags

##### System Management
- 🟢 **get_collection_stats** : Collection statistics information
- 🟢 **performance_optimization_analysis** : Performance optimization analysis
- 🟢 **milvus_system_optimization_report** : Comprehensive system optimization report

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


## 🔴 **EMERGENCY RESET**

**If you encounter container conflicts or system issues**, you can use the emergency reset script:

```bash
complete-reset.bat  # Windows
```

🔴 **CRITICAL WARNING**: This script will:
- **Kill ALL Podman containers** (not just Milvus)
- **Remove ALL Podman containers, pods, volumes, and networks**
- **Delete local MilvusData and volumes directories**
- **Permanently destroy all container data system-wide**

🔴 **Use ONLY if**
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


## Obsidian-Milvus-FastMCP Project Structure and Module Descriptions
##### Project Dependency Tree

```
📁 obsidian-milvus-FastMCP/
│
├── 🎯 main.py (Core Entry Point #1)
│   ├── logger.py ← Centralized logging system
│   ├── config.py ← Configuration management
│   ├── milvus_manager.py ← Milvus database operations
│   │   ├── logger.py (reused)
│   │   ├── config.py (reused)
│   │   └── embeddings.py ← AI embedding model management
│   │       ├── warning_suppressor.py ← Warning suppression utility
│   │       └── config.py (reused)
│   ├── obsidian_processor.py ← Obsidian note processing
│   │   ├── logger.py (reused)
│   │   ├── config.py (reused)
│   │   ├── embeddings.py (reused)
│   │   └── progress_monitor_cmd.py ← Progress monitoring for CLI
│   ├── watcher.py ← File system monitoring
│   │   ├── logger.py (reused)
│   │   └── config.py (reused)
│   └── robust_incremental_embedding.py ← Incremental embedding processing
│       ├── logger.py (reused)
│       └── config.py (reused)
│
├── 🔧 setup.py (Core Entry Point #2 - Testing & Configuration)
│   └── config.py (reused)
│
├── 📦 installer/installer_ui.py (Core Entry Point #3 - Windows Installer)
│   └── (PyQt5-based GUI installer)
│
├── 🌐 mcp_server.py ← MCP Server for Claude Desktop integration
│   ├── mcp_server_helpers.py ← Helper functions for MCP
│   ├── config.py (reused)
│   ├── milvus_manager.py (reused)
│   ├── obsidian_processor.py (reused)
│   ├── enhanced_search_engine.py ← Advanced search functionality
│   │   ├── embeddings.py (reused)
│   │   └── config.py (reused)
│   └── search_engine.py ← Basic search functionality
│       ├── embeddings.py (reused)
│       └── config.py (reused)
│
├── 🚀 Startup Scripts
│   ├── run-main.py ← Python wrapper for main.py
│   ├── run-main.bat ← Windows batch launcher
│   ├── auto_start_mcp_server.vbs ← Windows auto-startup script
│   ├── start-milvus.bat ← Milvus container startup
│   └── Various other .bat/.vbs startup utilities
│
└── 📄 Configuration Files
    ├── .env / .env.example ← Environment variables
    ├── requirements.txt ← Python dependencies
    ├── milvus-podman-compose.yml ← Podman container configuration
    └── milvus-docker-compose.yml ← Docker container configuration
```

##### Module Descriptions
- Core Entry Points
1. main.py - Primary Application Entry

```
Purpose: Main command-line interface for the Obsidian-Milvus integration
Key Functions:

Initializes the entire system (Milvus connection, Obsidian processor)
Provides interactive menu for operations:

Start MCP Server for Claude Desktop
Full embedding (reindex all files)
Incremental embedding with cleanup
Cleanup deleted files


Manages embedding progress and monitoring
Handles system resource management
```


2. setup.py - Interactive Test & Configuration Tool

```
Purpose: System setup, testing, and troubleshooting
Key Functions:

Tests Milvus connection and operations
Manages Podman/Docker containers
Configures Claude Desktop integration
Provides safe server restart functionality
Handles auto-startup configuration
```


3. installer/installer_ui.py - Windows GUI Installer

```
Purpose: Automated installation wizard for Windows users
Key Functions:

PyQt5-based graphical interface
Clones repository from GitHub
Installs Python dependencies
Sets up Podman and WSL
Configures system for first use
```


- Core Modules
4. config.py - Configuration Management

```
Purpose: Central configuration hub for all settings
Key Functions:

Manages paths (Obsidian vault, storage, Podman)
Sets embedding model parameters
Configures batch sizes and performance limits
Handles GPU/CPU settings
Auto-detects system paths
```


5. logger.py - Centralized Logging System

```
Purpose: Unified logging across all modules
Key Functions:

Module-specific loggers with consistent formatting
File and console output
Log rotation and size management
Error tracking and debugging support
```


6. milvus_manager.py - Milvus Database Interface

```
Purpose: Manages all interactions with Milvus vector database
Key Functions:

Connection management and health checking
Collection creation and management
Vector insertion and deletion
Search operations (with GPU optimization)
Container lifecycle management
Batch operations with intelligent sizing
```


7. obsidian_processor.py - Document Processing Engine

```
Purpose: Processes Obsidian notes for embedding
Key Functions:

Extracts text from Markdown and PDF files
Chunks documents intelligently
Manages embedding generation with batch optimization
Tracks processing progress
Handles incremental updates
Cleans up deleted files
```


8. embeddings.py - AI Embedding Model Management

```
Purpose: Manages sentence transformer models for text embeddings
Key Features:

Hardware profiling and optimization
Dynamic batch size optimization
GPU/CPU automatic selection
Memory management and caching
Support for multiple embedding models
Performance monitoring
```


9. watcher.py - File System Monitor

```
Purpose: Monitors Obsidian vault for changes
Key Functions:

Real-time file change detection
Triggers incremental processing
Handles file creation, modification, deletion
Efficient directory watching
```


10. mcp_server.py - MCP Server for Claude Desktop

```
Purpose: Provides search interface for Claude Desktop
Key Functions:

FastMCP-based server implementation
Multiple search tools exposed to Claude
Document retrieval and content access
Advanced search with metadata filtering
```


- Helper Modules
11. enhanced_search_engine.py - Advanced Search Features

```
Purpose: Provides sophisticated search capabilities
Key Functions:

Intelligent search with context expansion
Multi-query fusion search
Knowledge graph exploration
Performance optimization analysis
```


12. search_engine.py - Basic Search Implementation

```
Purpose: Core search functionality
Key Functions:

Vector similarity search
Metadata filtering
Result ranking and formatting
```


13. progress_monitor_cmd.py - CLI Progress Display

```
Purpose: Real-time progress monitoring in terminal
Key Functions:

Live progress bars and statistics
System resource monitoring
Error logging display
ETA calculations
```


14. robust_incremental_embedding.py - Incremental Processing

```
Purpose: Handles incremental updates efficiently
Key Functions:

Detects changed files
Processes only updates
Maintains consistency
```


- Utility Scripts
15. warning_suppressor.py - Warning Management

```
Purpose: Suppresses unnecessary warnings from libraries
Key Functions:

Filters TensorFlow/PyTorch warnings
Cleans console output
```

##### Data Flow

```
Initialization: main.py → config.py → milvus_manager.py → obsidian_processor.py
File Processing: watcher.py detects changes → obsidian_processor.py → embeddings.py → milvus_manager.py
Search Operations: mcp_server.py → enhanced_search_engine.py → milvus_manager.py
Progress Monitoring: All operations → progress_monitor_cmd.py → Terminal display
```


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- 🔵 **Milvus**: High-performance vector database with advanced indexing
- 🔵 **Claude**: Advanced AI assistant with MCP protocol support
- 🔵 **FastMCP**: Efficient MCP implementation framework
- 🔵 **Sentence Transformers**: State-of-the-art multilingual embedding models
- 🔵 **HNSW Algorithm**: Hierarchical Navigable Small World graphs for fast similarity search

---
 
🟢 **Author**: Jay Jeong, Ph.D. in Economics, Research Fellow at KCTDI, [acubens555@gmail.com](mailto:acubens555@gmail.com) 
