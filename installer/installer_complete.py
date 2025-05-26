"""
Complete Obsidian-Milvus-FastMCP Installer
Designed to run with Administrator privileges
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import os
import sys
import json
import shutil
import time
import threading
from pathlib import Path
import webbrowser
import ctypes

class ObsidianMilvusInstaller:
    def __init__(self):
        # Check if running as administrator
        self.check_admin_privileges()
        
        self.root = tk.Tk()
        self.root.title("Obsidian-Milvus-FastMCP Setup (Administrator)")
        self.root.geometry("900x750")
        self.root.resizable(False, False)
        
        # Set window style
        self.root.configure(bg='#f0f0f0')
        
        # Center window on screen
        self.center_window()
        
        # Installation state file
        self.state_file = Path("installer_state.json")
        self.state = self.load_state()
        
        # Installation variables
        self.install_path = tk.StringVar(value=self.state.get('install_path', ''))
        self.obsidian_vault_path = tk.StringVar(value=self.state.get('obsidian_vault_path', ''))
        self.podman_path = tk.StringVar(value=self.state.get('podman_path', ''))
        
        # Installation state tracking
        self.installation_step = self.state.get('installation_step', 0)
        
        # Current page tracking
        self.current_page = 0
        self.pages = []
        
        # Create UI
        self.create_ui()
        
        # Setup pages
        self.setup_pages()
        
        # Show initial page
        self.show_page(0)
    
    def check_admin_privileges(self):
        """Check if running with administrator privileges"""
        try:
            is_admin = ctypes.windll.shell32.IsUserAnAdmin()
            if not is_admin:
                messagebox.showerror(
                    "Administrator Required",
                    "This installer must be run as Administrator.\n\n"
                    "Please right-click the installer and select 'Run as administrator'."
                )
                sys.exit(1)
        except Exception:
            # On non-Windows systems or if check fails, continue
            pass
    
    def center_window(self):
        """Center window on screen"""
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (900 // 2)
        y = (self.root.winfo_screenheight() // 2) - (750 // 2)
        self.root.geometry(f"900x750+{x}+{y}")
    
    def create_ui(self):
        """Create the main UI structure"""
        # Main container frame (white background covers entire window)
        self.main_container = tk.Frame(self.root, bg='white')
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Footer with navigation buttons (pack this first to reserve bottom space)
        self.create_footer()
        
        # Content area with border (white frame with sunken border)
        self.content_container = tk.Frame(self.main_container, bg='white', relief=tk.SUNKEN, bd=1)
        # Add 10px padding on top and bottom now that header is removed
        self.content_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 10))
        
        # Inner content frame (where page contents will be placed)
        self.content_frame = tk.Frame(self.content_container, bg='white')
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    def create_footer(self):
        """Create footer with navigation buttons"""
        # Footer container (fixed height, anchored to bottom)
        footer_container = tk.Frame(self.main_container, bg='#f0f0f0', height=70)
        footer_container.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 10))
        footer_container.pack_propagate(False)
        
        # Separator line at the top of the footer
        separator_frame = tk.Frame(footer_container, bg='#d0d0d0', height=1)
        separator_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Button frame inside footer (holds progress label and navigation buttons)
        button_frame = tk.Frame(footer_container, bg='#f0f0f0')
        button_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        
        # Progress indicator label (e.g., "Step 1 of 7")
        self.progress_label = tk.Label(
            button_frame, text="Step 1 of 7",
            font=('Segoe UI', 9), bg='#f0f0f0', fg='#666'
        )
        self.progress_label.pack(side=tk.LEFT, pady=15)
        
        # Right side frame for navigation buttons
        button_right_frame = tk.Frame(button_frame, bg='#f0f0f0')
        button_right_frame.pack(side=tk.RIGHT, pady=10)
        
        # Cancel button
        self.cancel_button = tk.Button(
            button_right_frame, text="Cancel",
            font=('Segoe UI', 10), width=12, height=2,
            bg='white', fg='black', relief=tk.RAISED, bd=2,
            command=self.cancel_installation
        )
        self.cancel_button.pack(side=tk.RIGHT, padx=(0, 10))
        
        # Next button
        self.next_button = tk.Button(
            button_right_frame, text="Next >",
            font=('Segoe UI', 10, 'bold'), width=15, height=2,
            bg='#0078D4', fg='white', relief=tk.RAISED, bd=2,
            command=self.next_page
        )
        self.next_button.pack(side=tk.RIGHT, padx=(0, 10))
        
        # Back button (disabled by default on first page)
        self.back_button = tk.Button(
            button_right_frame, text="< Back",
            font=('Segoe UI', 10), width=12, height=2,
            bg='white', fg='black', relief=tk.RAISED, bd=2,
            command=self.previous_page, state=tk.DISABLED
        )
        self.back_button.pack(side=tk.RIGHT, padx=(0, 10))
    
    def setup_pages(self):
        """Setup installation pages"""
        self.pages = [
            self.create_welcome_page,          # 0
            self.create_conda_check_page,      # 1
            self.create_install_path_page,     # 2
            self.create_obsidian_path_page,    # 3
            self.create_ready_page,            # 4
            self.create_installing_page,       # 5
            self.create_complete_page          # 6
        ]
    
    def create_welcome_page(self):
        """Welcome page"""
        self.clear_content()
        
        # Welcome title
        welcome_label = tk.Label(
            self.content_frame,
            text="Welcome to Obsidian-Milvus-FastMCP Setup",
            font=('Segoe UI', 18, 'bold'), bg='white'
        )
        welcome_label.pack(pady=(20, 30))
        
        # Welcome description text
        info_text = (
            "This wizard will install Obsidian-Milvus-FastMCP with full administrative privileges.\n\n"
            "Obsidian-Milvus-FastMCP integrates Milvus vector database with Obsidian notes, enabling powerful semantic search capabilities through Claude Desktop.\n\n"
            "Complete Installation Process:\n"
            "• ✓ Prerequisites verification (Conda required)\n"
            "• ✓ Repository cloning from GitHub\n"
            "• ✓ Python dependencies installation via Conda\n"
            "• ✓ Podman container runtime setup\n"
            "• ✓ WSL (Windows Subsystem for Linux) configuration\n"
            "• ✓ Ubuntu Linux distribution installation\n"
            "• ✓ Container orchestration setup\n"
            "• ✓ Milvus vector database initialization\n"
            "• ✓ Auto-startup configuration\n"
            "• ✓ Claude Desktop integration\n\n"
            "Administrator privileges are required for system-level installations."
        )
        info_label = tk.Label(
            self.content_frame, text=info_text,
            font=('Segoe UI', 10), bg='white', justify=tk.LEFT
        )
        info_label.pack(pady=(0, 20), padx=20)
        
        # System requirements frame
        req_frame = tk.LabelFrame(
            self.content_frame, text="System Requirements",
            font=('Segoe UI', 11, 'bold'), bg='white',
            relief=tk.GROOVE, bd=2
        )
        req_frame.pack(pady=20, padx=20, fill=tk.X)
        requirements = (
            "• Windows 10 or later (required for WSL 2)\n"
            "• Anaconda or Miniconda (REQUIRED - must be pre-installed)\n"
            "• 20 GB free disk space (for containers and dependencies)\n"
            "• Stable internet connection for downloads\n"
            "• Administrator privileges (✓ Currently running as Administrator)\n"
            "• At least 8 GB RAM recommended for optimal performance"
        )
        req_label = tk.Label(
            req_frame, text=requirements,
            font=('Segoe UI', 9), bg='white', justify=tk.LEFT
        )
        req_label.pack(pady=15, padx=20, anchor='w')
        
        # Important notice frame
        notice_frame = tk.LabelFrame(
            self.content_frame, text="⚠️ Important Notice",
            font=('Segoe UI', 11, 'bold'), bg='white', fg='#D32F2F',
            relief=tk.GROOVE, bd=2
        )
        notice_frame.pack(pady=20, padx=20, fill=tk.X)
        notice_text = (
            "BEFORE STARTING: Please ensure Anaconda is installed!\n"
            "Download from: https://www.anaconda.com/download\n"
            "Install with default settings and add to PATH when prompted.\n\n"
            "This installer will make system-level changes including:\n"
            "- Installing Podman and WSL\n"
            "- Modifying system startup services\n"
            "- Creating container configurations\n"
            "- Setting up automatic services"
        )
        notice_label = tk.Label(
            notice_frame, text=notice_text,
            font=('Segoe UI', 9, 'bold'), bg='white', fg='#D32F2F', justify=tk.LEFT
        )
        notice_label.pack(pady=15, padx=20, anchor='w')
    
    def create_conda_check_page(self):
        """Conda prerequisite check page"""
        self.clear_content()
        
        # Title
        title_label = tk.Label(
            self.content_frame,
            text="Step 1: Conda Installation Verification",
            font=('Segoe UI', 16, 'bold'), bg='white'
        )
        title_label.pack(pady=(20, 30))
        
        # Info text
        info_text = (
            "This setup requires Anaconda or Miniconda to be pre-installed.\n"
            "Conda manages Python environments and package dependencies.\n\n"
            "With administrator privileges, the installer can verify system-wide Conda installation."
        )
        info_label = tk.Label(
            self.content_frame, text=info_text,
            font=('Segoe UI', 11), bg='white', justify=tk.LEFT
        )
        info_label.pack(pady=(0, 30), padx=40)
        
        # Conda detection status frame
        check_frame = tk.LabelFrame(
            self.content_frame, text="Conda Detection Status",
            font=('Segoe UI', 10, 'bold'), bg='white',
            relief=tk.GROOVE, bd=2
        )
        check_frame.pack(pady=20, padx=40, fill=tk.X)
        self.conda_status_label = tk.Label(
            check_frame,
            text="🔍 Checking Conda installation...",
            font=('Segoe UI', 12), bg='white', fg='#FF9800'
        )
        self.conda_status_label.pack(pady=20)
        # Indeterminate progress bar for Conda check
        self.conda_progress = ttk.Progressbar(check_frame, length=400, mode='indeterminate')
        self.conda_progress.pack(pady=(0, 20))
        
        # Instructions frame for manual steps if Conda not found
        inst_frame = tk.LabelFrame(
            self.content_frame, text="If Conda is not found:",
            font=('Segoe UI', 10, 'bold'), bg='white',
            relief=tk.GROOVE, bd=2
        )
        inst_frame.pack(pady=20, padx=40, fill=tk.X)
        inst_text = (
            "1. Download Anaconda from: https://www.anaconda.com/download\n"
            "2. Run the Anaconda installer AS ADMINISTRATOR\n"
            "3. IMPORTANT: Check \"Add Anaconda to my PATH environment variable\"\n"
            "4. Restart this Setup wizard after installation\n"
            "5. Ensure 'conda' command works in Command Prompt"
        )
        inst_label = tk.Label(
            inst_frame, text=inst_text,
            font=('Segoe UI', 9), bg='white', justify=tk.LEFT
        )
        inst_label.pack(pady=15, padx=20, anchor='w')
        
        # Download Anaconda button
        download_button = tk.Button(
            inst_frame, text="Open Anaconda Download Page",
            font=('Segoe UI', 9, 'bold'), width=30, height=2,
            bg='#4CAF50', fg='white', relief=tk.RAISED, bd=2,
            command=lambda: webbrowser.open("https://www.anaconda.com/download")
        )
        download_button.pack(pady=(0, 15))
        
        # Start automatic Conda check in a separate thread (so UI remains responsive)
        threading.Thread(target=self.auto_check_conda, daemon=True).start()
    
    def auto_check_conda(self):
        """Automatically check for Conda installation (runs in background thread)"""
        conda_found = False
        conda_info = ""
        # Possible conda commands or paths to check
        conda_commands = ["conda", "conda.exe"]
        conda_paths = [
            os.path.join(os.environ.get("USERPROFILE", ""), "Anaconda3", "Scripts", "conda.exe"),
            os.path.join(os.environ.get("USERPROFILE", ""), "Miniconda3", "Scripts", "conda.exe")
        ]
        # Try invoking conda directly
        for cmd in conda_commands:
            try:
                result = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=15)
                if result.returncode == 0:
                    conda_found = True
                    conda_info = result.stdout.strip()
                    break
            except Exception:
                continue
        # If not found in PATH, try common installation paths
        if not conda_found:
            for path in conda_paths:
                if os.path.exists(path):
                    try:
                        result = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=15)
                        if result.returncode == 0:
                            conda_found = True
                            conda_info = result.stdout.strip()
                            break
                    except Exception:
                        continue
        
        # Stop the progress bar animation
        self.conda_progress.stop()
        
        if conda_found:
            # Update status to found
            self.conda_status_label.config(text=f"✅ Conda detected: {conda_info}", fg='#4CAF50')
            # Enable Next button since Conda is available
            self.next_button.config(state=tk.NORMAL)
            # Show additional info: base environment path
            try:
                info_result = subprocess.run(['conda', 'info', '--base'], capture_output=True, text=True, timeout=10)
                if info_result.returncode == 0:
                    conda_base = info_result.stdout.strip()
                    details_label = tk.Label(
                        self.content_frame,
                        text=f"📁 Base environment: {conda_base}",
                        font=('Segoe UI', 8), bg='white', fg='#666'
                    )
                    details_label.pack(pady=5)
            except Exception:
                pass
        else:
            # If Conda not found, update status and show manual options
            self.show_conda_not_found()
    
    def show_conda_not_found(self):
        """Show enhanced message and options when Conda is not found"""
        # Update status label to not found
        self.conda_status_label.config(text="❌ Conda not found in system", fg='#F44336')
        # Disable Next button until user takes action
        self.next_button.config(state=tk.DISABLED)
        
        # Manual options frame (appears below if Conda isn't found)
        manual_frame = tk.LabelFrame(
            self.content_frame, text="Manual Options",
            font=('Segoe UI', 10, 'bold'), bg='white',
            relief=tk.GROOVE, bd=2
        )
        manual_frame.pack(pady=20, padx=40, fill=tk.X)
        button_container = tk.Frame(manual_frame, bg='white')
        button_container.pack(pady=15)
        # Manual confirmation button
        manual_button = tk.Button(
            button_container, text="I have Conda installed",
            font=('Segoe UI', 9, 'bold'), width=20, height=2,
            bg='#FF9800', fg='white', relief=tk.RAISED, bd=2,
            command=self.manual_conda_confirm
        )
        manual_button.pack(side=tk.LEFT, padx=10)
        # Retry detection button
        retry_button = tk.Button(
            button_container, text="Retry Detection",
            font=('Segoe UI', 9, 'bold'), width=15, height=2,
            bg='#2196F3', fg='white', relief=tk.RAISED, bd=2,
            command=self.retry_conda_check
        )
        retry_button.pack(side=tk.LEFT, padx=10)
    
    def manual_conda_confirm(self):
        """Handle manual confirmation when user asserts Conda is installed"""
        response = messagebox.askyesno(
            "Conda Confirmation",
            "Do you confirm that Anaconda or Miniconda is properly installed?\n\n"
            "⚠️ WARNING: Installation may fail if Conda is not available.\n"
            "Make sure 'conda --version' works in Command Prompt.\n\n"
            "Continue anyway?"
        )
        if response:
            # Mark Conda as confirmed manually
            self.conda_status_label.config(text="✅ Conda confirmed manually (not verified)", fg='#FF9800')
            # Re-enable Next button to proceed despite not auto-detected
            self.next_button.config(state=tk.NORMAL)
    
    def retry_conda_check(self):
        """Retry Conda detection when user clicks Retry"""
        # Update status and restart the indeterminate progress bar
        self.conda_status_label.config(text="🔍 Retrying Conda detection...", fg='#FF9800')
        self.conda_progress.start()
        # Remove the manual options frame (if it exists) before retrying
        for widget in self.content_frame.winfo_children():
            if isinstance(widget, tk.LabelFrame) and widget.cget('text') == 'Manual Options':
                widget.destroy()
        # Retry the Conda detection after a short delay
        self.root.after(1000, self.auto_check_conda)
    
    def clear_content(self):
        """Clear all widgets from the content frame (for navigating between pages)"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
    
    def create_install_path_page(self):
        """Installation path selection page"""
        self.clear_content()
        
        # Title
        title_label = tk.Label(
            self.content_frame,
            text="Step 2: Choose Installation Location",
            font=('Segoe UI', 16, 'bold'), bg='white'
        )
        title_label.pack(pady=(20, 30))
        
        # Instruction text
        info_label = tk.Label(
            self.content_frame,
            text="Select where to install Obsidian-Milvus-FastMCP.\nWith administrator privileges, you can install to any location.",
            font=('Segoe UI', 11), bg='white', justify=tk.CENTER
        )
        info_label.pack(pady=(0, 30))
        
        # Installation Directory frame
        path_frame = tk.LabelFrame(
            self.content_frame, text="Installation Directory",
            font=('Segoe UI', 11, 'bold'), bg='white',
            relief=tk.GROOVE, bd=2
        )
        path_frame.pack(pady=20, padx=40, fill=tk.X)
        # Path entry sub-frame
        entry_frame = tk.Frame(path_frame, bg='white')
        entry_frame.pack(fill=tk.X, padx=20, pady=15)
        # Entry field for path
        self.path_entry = tk.Entry(entry_frame, textvariable=self.install_path, font=('Segoe UI', 10), width=60)
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 15))
        # Browse button for path selection
        browse_button = tk.Button(
            entry_frame, text="Browse...",
            font=('Segoe UI', 9, 'bold'), width=12, height=2,
            bg='#0078D4', fg='white', relief=tk.RAISED, bd=2,
            command=self.browse_install_path
        )
        browse_button.pack(side=tk.RIGHT)
        
        # Disk space check info
        space_frame = tk.Frame(self.content_frame, bg='white')
        space_frame.pack(pady=20)
        self.space_label = tk.Label(
            space_frame,
            text="Checking available disk space...",
            font=('Segoe UI', 10), bg='white', fg='#666'
        )
        self.space_label.pack()
        
        # Note about requirements
        req_label = tk.Label(
            self.content_frame,
            text="💾 Requirements: 20 GB free space recommended\n📦 Installation includes: Repository + Dependencies + Containers + Models",
            font=('Segoe UI', 9), bg='white', justify=tk.CENTER
        )
        req_label.pack(pady=(0, 20))
        
        # Trigger disk space check in background thread
        threading.Thread(target=self.check_disk_space, daemon=True).start()
    
    def browse_install_path(self):
        """Open a dialog for selecting installation directory"""
        selected_dir = filedialog.askdirectory(title="Select Installation Directory")
        if selected_dir:
            self.install_path.set(selected_dir)
            # Save the chosen path to state
            self.save_state()
            # Update available disk space info
            threading.Thread(target=self.check_disk_space, daemon=True).start()
    
    def check_disk_space(self):
        """Check available disk space at the chosen install path"""
        path = self.install_path.get() or os.getcwd()
        try:
            total, used, free = shutil.disk_usage(path)
            free_gb = free // (1024**3)
            self.space_label.config(text=f"✅ {free_gb} GB free space available.")
        except Exception:
            self.space_label.config(text="Unable to determine disk space for the selected location.")
    
    def create_obsidian_path_page(self):
        """Obsidian vault path selection page"""
        self.clear_content()
        
        # Title
        title_label = tk.Label(
            self.content_frame,
            text="Step 3: Select Obsidian Vault",
            font=('Segoe UI', 16, 'bold'), bg='white'
        )
        title_label.pack(pady=(20, 30))
        
        # Instructions
        info_label = tk.Label(
            self.content_frame,
            text="Choose the Obsidian vault (folder) you want to index for semantic search.\nYou can also do this later in config.py.",
            font=('Segoe UI', 11), bg='white', justify=tk.CENTER
        )
        info_label.pack(pady=(0, 30))
        
        # Vault selection frame
        vault_frame = tk.LabelFrame(
            self.content_frame, text="Obsidian Vault Selection",
            font=('Segoe UI', 11, 'bold'), bg='white',
            relief=tk.GROOVE, bd=2
        )
        vault_frame.pack(pady=20, padx=40, fill=tk.X)
        # Vault path entry sub-frame
        entry_frame = tk.Frame(vault_frame, bg='white')
        entry_frame.pack(fill=tk.X, padx=20, pady=15)
        # Entry field for vault path
        self.vault_entry = tk.Entry(entry_frame, textvariable=self.obsidian_vault_path, font=('Segoe UI', 10), width=60)
        self.vault_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 15))
        # Browse button for vault path
        browse_button = tk.Button(
            entry_frame, text="Browse...",
            font=('Segoe UI', 9, 'bold'), width=12, height=2,
            bg='#0078D4', fg='white', relief=tk.RAISED, bd=2,
            command=self.browse_vault_path
        )
        browse_button.pack(side=tk.RIGHT)
        
        # Note
        note_label = tk.Label(
            self.content_frame,
            text="(Obsidian vault path can be changed later in config.py if needed.)",
            font=('Segoe UI', 9, 'italic'), bg='white'
        )
        note_label.pack(pady=(10, 20))
    
    def browse_vault_path(self):
        """Open a dialog for selecting Obsidian vault directory"""
        selected_dir = filedialog.askdirectory(title="Select Obsidian Vault Directory")
        if selected_dir:
            self.obsidian_vault_path.set(selected_dir)
            # Save the chosen path to state
            self.save_state()
    
    def create_ready_page(self):
        """Ready-to-install confirmation page"""
        self.clear_content()
        
        # Title
        title_label = tk.Label(
            self.content_frame,
            text="Step 4: Ready to Install",
            font=('Segoe UI', 16, 'bold'), bg='white'
        )
        title_label.pack(pady=(20, 30))
        
        # Summary text
        summary_label = tk.Label(
            self.content_frame,
            text="Setup is ready to begin installation.\n\nClick **Install** to proceed with the installation.",
            font=('Segoe UI', 11), bg='white', justify=tk.CENTER
        )
        summary_label.pack(pady=(0, 30))
        
        # (In this simple ready page, we could list chosen settings or prerequisites if needed)
    
    def create_installing_page(self):
        """Installation progress page"""
        self.clear_content()
        
        # Disable navigation buttons during installation
        self.back_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.DISABLED)
        
        # Title
        title_label = tk.Label(
            self.content_frame,
            text="🚀 Installing Obsidian-Milvus-FastMCP",
            font=('Segoe UI', 16, 'bold'), bg='white'
        )
        title_label.pack(pady=(20, 30))
        
        # Installation progress frame
        status_frame = tk.LabelFrame(
            self.content_frame, text="Installation Progress",
            font=('Segoe UI', 10, 'bold'), bg='white',
            relief=tk.GROOVE, bd=2
        )
        status_frame.pack(pady=20, padx=40, fill=tk.X)
        # Current action label
        self.action_label = tk.Label(
            status_frame,
            text="🔄 Preparing installation...",
            font=('Segoe UI', 11, 'bold'), bg='white'
        )
        self.action_label.pack(pady=15)
        # Progress bar
        self.install_progress = ttk.Progressbar(status_frame, length=600, mode='determinate')
        self.install_progress.pack(pady=(0, 15))
        # Progress percentage label
        self.progress_percent = tk.Label(
            status_frame, text="0%",
            font=('Segoe UI', 9), bg='white', fg='#666'
        )
        self.progress_percent.pack(pady=(0, 15))
        
        # Detailed log frame (expands to fill remaining space)
        details_frame = tk.LabelFrame(
            self.content_frame, text="Installation Details",
            font=('Segoe UI', 10, 'bold'), bg='white',
            relief=tk.GROOVE, bd=2
        )
        details_frame.pack(pady=20, padx=40, fill=tk.BOTH, expand=True)
        # Frame to hold the text widget and scrollbar
        details_container = tk.Frame(details_frame, bg='white')
        details_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        # Text widget for log details
        self.details_text = tk.Text(
            details_container, height=12, width=85,
            font=('Consolas', 8), bg='#f8f8f8',
            relief=tk.SUNKEN, bd=1
        )
        # Scrollbar for the text widget
        scrollbar = tk.Scrollbar(details_container, orient=tk.VERTICAL, command=self.details_text.yview)
        self.details_text.config(yscrollcommand=scrollbar.set)
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Start the installation process in a background thread
        install_thread = threading.Thread(target=self.perform_installation, daemon=True)
        install_thread.start()
    
    def perform_installation(self):
        """Perform the installation steps with admin privileges"""
        steps = [
            ("🔍 Verifying system requirements...", self.verify_requirements, 10),
            ("📂 Creating installation directory...", self.create_install_dir, 15),
            ("📥 Cloning repository from GitHub...", self.clone_repository, 30),
            ("🐍 Installing Python dependencies...", self.install_dependencies, 50),
            ("🐳 Setting up Podman container runtime...", self.setup_podman, 70),
            ("🐧 Configuring WSL and Ubuntu...", self.setup_wsl, 85),
            ("⚙️ Configuring system integration...", self.configure_system, 95),
            ("✅ Finalizing installation...", self.finalize_installation, 100)
        ]
        
        try:
            for i, (step_name, step_func, progress) in enumerate(steps, 1):
                # Update progress UI
                self.update_progress(step_name, progress)
                # Perform the step (which may take time)
                step_func()
            # After all steps, mark installation as finished
            self.finish_installation()
        except Exception as e:
            # If an installation step throws an error, show error and enable cancel
            messagebox.showerror("Installation Error", f"An error occurred: {str(e)}")
            self.next_button.config(state=tk.NORMAL, text="Next >")
            self.cancel_button.config(state=tk.NORMAL)
    
    def update_progress(self, action, progress):
        """Update the progress bar and labels from background thread"""
        # Update current action label and percentage
        self.action_label.config(text=action)
        self.install_progress.step(progress - self.install_progress['value'])
        self.progress_percent.config(text=f"{progress}%")
        # Force update of UI to reflect changes
        self.content_frame.update_idletasks()
        # Small delay for smooth progress bar movement
        time.sleep(0.5)
        # If progress reaches 100, re-enable navigation (handled in finish_installation)
        if progress == 100:
            self.next_button.config(text="Finish", command=self.finish_installation, state=tk.NORMAL)
            self.cancel_button.config(state=tk.DISABLED)
    
    def verify_requirements(self):
        """Placeholder for verifying system requirements"""
        time.sleep(1)  # Simulate some work
    
    def create_install_dir(self):
        """Placeholder for creating installation directory"""
        time.sleep(1)  # Simulate some work
    
    def clone_repository(self):
        """Placeholder for cloning repository"""
        time.sleep(1)  # Simulate some work
    
    def install_dependencies(self):
        """Placeholder for installing dependencies"""
        time.sleep(1)  # Simulate some work
    
    def setup_podman(self):
        """Placeholder for setting up Podman"""
        time.sleep(1)  # Simulate some work
    
    def setup_wsl(self):
        """Placeholder for setting up WSL"""
        time.sleep(1)  # Simulate some work
    
    def configure_system(self):
        """Placeholder for system integration configuration"""
        time.sleep(1)  # Simulate some work
    
    def finalize_installation(self):
        """Placeholder for finalizing installation"""
        time.sleep(1)  # Simulate some work
    
    def create_complete_page(self):
        """Installation complete page"""
        self.clear_content()
        
        # Re-enable only the Finish (Next) button on completion
        self.back_button.config(state=tk.DISABLED)
        self.next_button.config(text="Finish", command=self.finish_installation, state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)
        
        # Success title
        title_label = tk.Label(
            self.content_frame,
            text="🎉 Installation Complete! 🎉",
            font=('Segoe UI', 18, 'bold'), bg='white', fg='#4CAF50'
        )
        title_label.pack(pady=(20, 30))
        
        # Success message
        success_label = tk.Label(
            self.content_frame,
            text="Obsidian-Milvus-FastMCP has been installed successfully with administrator privileges!",
            font=('Segoe UI', 12), bg='white', justify=tk.CENTER
        )
        success_label.pack(pady=(0, 30))
        
        # Next steps frame
        steps_frame = tk.LabelFrame(
            self.content_frame, text="🚀 Next Steps",
            font=('Segoe UI', 11, 'bold'), bg='white',
            relief=tk.GROOVE, bd=2
        )
        steps_frame.pack(pady=20, padx=40, fill=tk.X)
        steps_text = (
            "1. 🔄 Restart your computer to complete WSL setup\n"
            "2. 📚 Review auto-startup documentation (will open automatically)\n"
            "3. 🐳 Configure Podman and Milvus auto-startup\n"
            "4. 💬 Open Claude Desktop application\n"
            "5. 🔍 Start using Obsidian semantic search!\n\n"
            "Your system is now ready for advanced note searching and AI integration."
        )
        steps_label = tk.Label(
            steps_frame, text=steps_text,
            font=('Segoe UI', 9), bg='white', justify=tk.LEFT
        )
        steps_label.pack(pady=15, padx=20, anchor='w')
        
        # Installation details frame
        details_frame = tk.LabelFrame(
            self.content_frame, text="📋 Installation Details",
            font=('Segoe UI', 11, 'bold'), bg='white',
            relief=tk.GROOVE, bd=2
        )
        details_frame.pack(pady=20, padx=40, fill=tk.X)
        install_path = self.install_path.get()
        vault_path = self.obsidian_vault_path.get() or "Not configured (can be set in config.py)"
        details_text = (
            f"🎯 Installation Mode: Administrator (System-wide)\n"
            f"📁 Project Directory: {install_path}\n"
            f"📝 Obsidian Vault: {vault_path}\n"
            f"🌐 Milvus API: http://localhost:19530 (after startup)\n"
            f"🌐 Milvus Web UI: http://localhost:9091 (after startup)\n"
            f"💬 Claude Desktop: Ready for integration\n"
            f"🔧 Configuration: config.py in project directory"
        )
        details_label = tk.Label(
            details_frame, text=details_text,
            font=('Segoe UI', 9), bg='white', justify=tk.LEFT
        )
        details_label.pack(pady=15, padx=20, anchor='w')
        
        # Option to open documentation after finish
        self.open_docs_var = tk.BooleanVar(value=True)
        docs_check = tk.Checkbutton(
            self.content_frame,
            text="📖 Open setup documentation when finished",
            variable=self.open_docs_var,
            font=('Segoe UI', 10), bg='white'
        )
        docs_check.pack(pady=20)
        
        # Cleanup any saved state since installation is complete
        try:
            self.state_file.unlink(missing_ok=True)
        except Exception:
            pass
    
    def next_page(self):
        """Go to the next page, with validation for certain steps"""
        # If on installation path page, ensure a path is selected
        if self.current_page == 2 and not self.install_path.get().strip():
            messagebox.showwarning("Path Required", "Please select an installation path.")
            return
        # Advance to next page if not at the last page
        if self.current_page < len(self.pages) - 1:
            self.installation_step = self.current_page + 1
            self.save_state()
            self.show_page(self.current_page + 1)
    
    def previous_page(self):
        """Go to the previous page"""
        if self.current_page > 0:
            self.show_page(self.current_page - 1)
    
    def cancel_installation(self):
        """Cancel the installation (with confirmation)"""
        response = messagebox.askyesno(
            "Cancel Setup",
            "Are you sure you want to cancel the installation?\n\nAny progress will be lost."
        )
        if response:
            try:
                # Remove state file on cancel
                self.state_file.unlink(missing_ok=True)
            except Exception:
                pass
            self.root.quit()
    
    def finish_installation(self):
        """Finish the installation process"""
        # If user opted to open documentation, open it in browser
        if hasattr(self, 'open_docs_var') and self.open_docs_var.get():
            try:
                webbrowser.open('https://share.note.sx/r6kx06pj#78CIGnxLJYkJG+ZrQKYQhU35gtl+nKa47ZllwEyfUE0')
                webbrowser.open('https://share.note.sx/y9vrzgj6#zr1aL4s1WFBK/A4WvqvkP6ETVMC4sKcAwbqAt4NyZhk')
            except Exception:
                pass
        
        # Prompt for reboot to complete WSL installation
        response = messagebox.askyesno(
            "Installation Complete",
            "Installation is complete!\n\nWould you like to restart your computer now to complete WSL setup?"
        )
        if response:
            try:
                # Schedule a system restart in 30 seconds
                subprocess.run(['shutdown', '/r', '/t', '30'], check=False)
                messagebox.showinfo(
                    "Restart Scheduled",
                    "Computer will restart in 30 seconds.\nSave any unsaved work now."
                )
            except Exception:
                messagebox.showinfo(
                    "Manual Restart Required",
                    "Please restart your computer manually to complete the installation."
                )
        # Close the installer
        self.root.quit()
    
    def save_state(self):
        """Save current installation state to file"""
        state = {
            'install_path': self.install_path.get(),
            'obsidian_vault_path': self.obsidian_vault_path.get(),
            'podman_path': self.podman_path.get(),
            'installation_step': self.installation_step
        }
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass
    
    def load_state(self):
        """Load installation state from file if it exists"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def show_page(self, index):
        """Display a specific page and update navigation buttons"""
        if 0 <= index < len(self.pages):
            self.current_page = index
            self.installation_step = index
            # Create and show the page content
            self.pages[index]()
            # Update progress text (e.g., "Step X of Y")
            self.progress_label.config(text=f"Step {index + 1} of {len(self.pages)}")
            # Update Back button state (disabled for first page and during/after install)
            can_go_back = index > 0 and index not in [5, 6]  # no going back on pages 5 (installing) and 6 (complete)
            self.back_button.config(state=tk.NORMAL if can_go_back else tk.DISABLED)
            # Update Next button text and state depending on page
            if index == len(self.pages) - 1:    # Last page -> Finish
                self.next_button.config(text="Finish", state=tk.NORMAL)
            elif index == 4:  # Ready page -> Install
                self.next_button.config(text="Install", state=tk.NORMAL)
            elif index == 5:  # Installing page -> Installing... (disabled)
                self.next_button.config(text="Installing...", state=tk.DISABLED)
            else:
                self.next_button.config(text="Next >", state=tk.NORMAL)
    
    def run(self):
        """Run the installer main loop"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.root.quit()

if __name__ == "__main__":
    try:
        installer = ObsidianMilvusInstaller()
        installer.run()
    except Exception as e:
        messagebox.showerror("Installer Error", f"Failed to start installer: {str(e)}")
        sys.exit(1)
