#!/usr/bin/env python3
"""
Obsidian-Milvus-FastMCP Installer (PyQt5)
Single-file GUI installer script.
Requires: PyQt5, Windows 10/11 with admin privileges, Conda pre-installed.
"""
import sys
import os
import json
import subprocess
import shutil
import threading
import time
from pathlib import Path
from PyQt5 import QtWidgets, QtGui, QtCore

class ObsidianMilvusInstaller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Verify administrator privileges (Windows only)
        self.is_admin = True
        try:
            import ctypes
            if not ctypes.windll.shell32.IsUserAnAdmin():
                raise PermissionError
        except Exception:
            # Not running as admin or check failed
            QtWidgets.QMessageBox.critical(None, "Administrator Required",
                                          "This installer must be run as Administrator.\n"
                                          "Right-click the script or EXE and choose 'Run as administrator'.")
            sys.exit(1)
        # Determine if running as a frozen executable (compiled .exe via PyInstaller)
        self.is_compiled = getattr(sys, 'frozen', False)
        # State file to store progress (for reboot resume)
        self.state_file = Path("installer_state.json")
        self.state = {}
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    self.state = json.load(f)
            except Exception:
                self.state = {}
        # Load any saved state variables (installation path, vault path, etc.)
        self.install_path = self.state.get('install_path', "")
        self.obsidian_vault_path = self.state.get('obsidian_vault_path', "")
        self.podman_path = self.state.get('podman_path', "")
        # Check if we are resuming after a reboot (phase 1 completed)
        self.resuming = (self.state.get('phase') == 1)
        # Build the UI elements and pages
        self.init_ui()
        # If resuming, jump directly to configuration page; else start at welcome
        if self.resuming:
            self.stacked_pages.setCurrentIndex(self.pages_index_map['config'])
            self.current_page_name = 'config'
        else:
            self.stacked_pages.setCurrentIndex(self.pages_index_map['welcome'])
            self.current_page_name = 'welcome'
        self.update_progress_label()

    def init_ui(self):
        """Initialize the main window and all wizard pages."""
        self.setWindowTitle("Obsidian-Milvus-FastMCP Setup (Administrator)")
        self.resize(900, 750)
        self.setFixedSize(900, 750)  # Fixed size to match design
        app_font = QtGui.QFont("Segoe UI", 9)
        QtWidgets.QApplication.instance().setFont(app_font)

        central_widget = QtWidgets.QWidget(self)
        central_layout = QtWidgets.QVBoxLayout(central_widget)
        central_layout.setContentsMargins(10, 10, 10, 10)
        central_layout.setSpacing(0)
        self.setCentralWidget(central_widget)

        # Header section with title on left and admin shield on right (blue background)
        header_frame = QtWidgets.QFrame()
        header_frame.setStyleSheet("background-color: #0078D4;")
        header_frame.setFixedHeight(80)
        header_layout = QtWidgets.QHBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 0, 20, 0)
        title_label = QtWidgets.QLabel("Obsidian-Milvus-FastMCP Setup Wizard [Administrator]")
        title_label.setFont(QtGui.QFont("Segoe UI", 16, QtGui.QFont.Bold))
        title_label.setStyleSheet("color: white;")
        admin_label = QtWidgets.QLabel("🛡️ ADMIN")
        admin_label.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.Bold))
        admin_label.setStyleSheet("color: yellow;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(admin_label)
        central_layout.addWidget(header_frame)

        # Content container (white background with slight border for pages)
        content_container = QtWidgets.QFrame()
        content_container.setStyleSheet("background-color: white; border: 1px solid #C0C0C0;")
        content_layout = QtWidgets.QVBoxLayout(content_container)
        content_layout.setContentsMargins(20, 20, 20, 20)
        # Stacked widget to hold different pages of the wizard
        self.stacked_pages = QtWidgets.QStackedWidget()
        content_layout.addWidget(self.stacked_pages)
        central_layout.addWidget(content_container, 1)

        # Footer with progress label and navigation buttons
        footer_frame = QtWidgets.QFrame()
        footer_frame.setStyleSheet("background-color: #f0f0f0;")
        footer_frame.setFixedHeight(70)
        footer_layout = QtWidgets.QVBoxLayout(footer_frame)
        footer_layout.setContentsMargins(20, 0, 20, 10)
        # Separator line
        sep_line = QtWidgets.QFrame()
        sep_line.setFrameShape(QtWidgets.QFrame.HLine)
        sep_line.setFrameShadow(QtWidgets.QFrame.Plain)
        sep_line.setStyleSheet("background-color: #d0d0d0;")
        sep_line.setFixedHeight(1)
        footer_layout.addWidget(sep_line)
        # Bottom row with progress label and buttons
        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        # Progress label (e.g., "Step 1 of 7")
        self.progress_label = QtWidgets.QLabel("")
        self.progress_label.setFont(QtGui.QFont("Segoe UI", 9))
        self.progress_label.setStyleSheet("color: #666666;")
        bottom_row.addWidget(self.progress_label)
        bottom_row.addStretch()
        # Cancel button
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setFont(QtGui.QFont("Segoe UI", 10))
        self.cancel_button.setFixedSize(100, 40)
        self.cancel_button.setStyleSheet("background-color: white; color: black; border: 2px solid #d0d0d0;")
        self.cancel_button.clicked.connect(self.cancel_installation)
        bottom_row.addWidget(self.cancel_button)
        # Back button
        self.back_button = QtWidgets.QPushButton("< Back")
        self.back_button.setFont(QtGui.QFont("Segoe UI", 10))
        self.back_button.setFixedSize(100, 40)
        self.back_button.setStyleSheet("background-color: white; color: black; border: 2px solid #d0d0d0;")
        self.back_button.clicked.connect(self.previous_page)
        self.back_button.setDisabled(True)  # Initially disabled (on welcome page)
        bottom_row.addWidget(self.back_button)
        # Next/Install/Finish button
        self.next_button = QtWidgets.QPushButton("Next >")
        self.next_button.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.Bold))
        self.next_button.setFixedSize(120, 40)
        self.next_button.setStyleSheet("background-color: #0078D4; color: white;")
        self.next_button.clicked.connect(self.next_page)
        bottom_row.addWidget(self.next_button)
        footer_layout.addLayout(bottom_row)
        central_layout.addWidget(footer_frame)

        # --- Define each page in the wizard ---

        # Page 0: Welcome/Introduction
        welcome_page = QtWidgets.QWidget()
        wl_layout = QtWidgets.QVBoxLayout(welcome_page)
        wl_layout.setAlignment(QtCore.Qt.AlignTop)
        welcome_label = QtWidgets.QLabel("Welcome to Obsidian-Milvus-FastMCP Setup")
        welcome_label.setFont(QtGui.QFont("Segoe UI", 18, QtGui.QFont.Bold))
        welcome_label.setAlignment(QtCore.Qt.AlignCenter)
        wl_layout.addWidget(welcome_label)
        info_text = (
            "This wizard will install Obsidian-Milvus-FastMCP with full administrative privileges.\n\n"
            "Obsidian-Milvus-FastMCP integrates the Milvus vector database with Obsidian notes, enabling powerful semantic search through Claude Desktop.\n\n"
            "Complete Installation Process:\n"
            "• ✓ Prerequisites verification (Conda required)\n"
            "• ✓ Repository cloning from GitHub\n"
            "• ✓ Python dependencies installation via Conda\n"
            "• ✓ Podman container runtime setup\n"
            "• ✓ WSL (Windows Subsystem for Linux) configuration\n"
            "• ✓ Ubuntu Linux distribution installation\n"
            "• ✓ Container orchestration setup (Milvus)\n"
            "• ✓ Milvus vector database initialization\n"
            "• ✓ Auto-startup configuration\n"
            "• ✓ Claude Desktop integration\n\n"
            "Administrator privileges are required for system-level installation."
        )
        info_label = QtWidgets.QLabel(info_text)
        info_label.setFont(QtGui.QFont("Segoe UI", 10))
        info_label.setWordWrap(True)
        wl_layout.addWidget(info_label)
        # System requirements
        req_group = QtWidgets.QGroupBox("System Requirements")
        req_group.setFont(QtGui.QFont("Segoe UI", 11, QtGui.QFont.Bold))
        req_group.setStyleSheet("QGroupBox { background-color: white; }")
        req_layout = QtWidgets.QVBoxLayout(req_group)
        req_text = (
            "• Windows 10 or later (required for WSL 2)\n"
            "• Anaconda or Miniconda (REQUIRED - must be pre-installed)\n"
            "• 20 GB free disk space (for containers and dependencies)\n"
            "• Stable internet connection for downloads\n"
            "• Administrator privileges (✓ Running as Administrator)\n"
            "• 8 GB RAM or more recommended"
        )
        req_label = QtWidgets.QLabel(req_text)
        req_label.setFont(QtGui.QFont("Segoe UI", 9))
        req_label.setWordWrap(True)
        req_layout.addWidget(req_label)
        wl_layout.addWidget(req_group)
        # Important notice
        notice_group = QtWidgets.QGroupBox("⚠️ Important Notice")
        notice_group.setFont(QtGui.QFont("Segoe UI", 11, QtGui.QFont.Bold))
        notice_group.setStyleSheet("QGroupBox { background-color: white; color: #D32F2F; }")
        notice_layout = QtWidgets.QVBoxLayout(notice_group)
        notice_text = (
            "BEFORE STARTING: Please ensure Anaconda (Conda) is installed!\n"
            "Download from: https://www.anaconda.com/download\n"
            "Install with default settings and **add Anaconda to PATH** when prompted.\n\n"
            "This installer will make system-level changes including:\n"
            "- Installing Podman and WSL\n"
            "- Modifying system startup services\n"
            "- Creating container configurations\n"
            "- Setting up automatic services"
        )
        notice_label = QtWidgets.QLabel(notice_text)
        notice_label.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold))
        notice_label.setWordWrap(True)
        notice_layout.addWidget(notice_label)
        wl_layout.addWidget(notice_group)
        self.stacked_pages.addWidget(welcome_page)

        # Page 1: Conda Installation Verification
        conda_page = QtWidgets.QWidget()
        conda_layout = QtWidgets.QVBoxLayout(conda_page)
        conda_layout.setAlignment(QtCore.Qt.AlignTop)
        conda_label = QtWidgets.QLabel("Step 1: Conda Installation Verification")
        conda_label.setFont(QtGui.QFont("Segoe UI", 16, QtGui.QFont.Bold))
        conda_layout.addWidget(conda_label)
        conda_info = QtWidgets.QLabel(
            "Anaconda (Conda) must be installed before proceeding. If you haven't installed Conda, please do so now."
        )
        conda_info.setFont(QtGui.QFont("Segoe UI", 10))
        conda_info.setWordWrap(True)
        conda_layout.addWidget(conda_info)
        download_link = QtWidgets.QLabel('<a href="https://www.anaconda.com/download">Download Anaconda</a>')
        download_link.setOpenExternalLinks(True)
        download_link.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.Bold))
        conda_layout.addWidget(download_link)
        self.conda_status_label = QtWidgets.QLabel("Checking for Conda installation...")
        self.conda_status_label.setFont(QtGui.QFont("Segoe UI", 10))
        conda_layout.addWidget(self.conda_status_label)
        # Indeterminate progress bar for checking Conda
        self.conda_progress = QtWidgets.QProgressBar()
        self.conda_progress.setRange(0, 0)  # 0,0 = infinite/progress animation
        self.conda_progress.setFixedWidth(300)
        conda_layout.addWidget(self.conda_progress)
        # Manual confirmation button (hidden until needed)
        self.conda_manual_button = QtWidgets.QPushButton("I have installed Conda")
        self.conda_manual_button.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold))
        self.conda_manual_button.setVisible(False)
        self.conda_manual_button.clicked.connect(self.check_conda)
        conda_layout.addWidget(self.conda_manual_button)
        self.stacked_pages.addWidget(conda_page)

        # Page 2: Installation Path selection
        path_page = QtWidgets.QWidget()
        path_layout = QtWidgets.QVBoxLayout(path_page)
        path_layout.setAlignment(QtCore.Qt.AlignTop)
        path_label = QtWidgets.QLabel("Step 2: Choose Installation Path")
        path_label.setFont(QtGui.QFont("Segoe UI", 16, QtGui.QFont.Bold))
        path_layout.addWidget(path_label)
        path_info = QtWidgets.QLabel(
            "Select the folder where the Obsidian-Milvus-FastMCP repository will be installed."
        )
        path_info.setFont(QtGui.QFont("Segoe UI", 10))
        path_info.setWordWrap(True)
        path_layout.addWidget(path_info)
        path_select_layout = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setText(self.install_path)  # prefill if loaded from state
        path_select_layout.addWidget(self.path_edit)
        browse_btn = QtWidgets.QPushButton("Browse...")
        def browse_folder():
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Installation Folder", "")
            if folder:
                self.path_edit.setText(folder)
        browse_btn.clicked.connect(browse_folder)
        path_select_layout.addWidget(browse_btn)
        path_layout.addLayout(path_select_layout)
        self.stacked_pages.addWidget(path_page)

        # Page 3: Summary (Ready to install)
        ready_page = QtWidgets.QWidget()
        ready_layout = QtWidgets.QVBoxLayout(ready_page)
        ready_layout.setAlignment(QtCore.Qt.AlignTop)
        ready_title = QtWidgets.QLabel("Setup is ready to begin installation.")
        ready_title.setFont(QtGui.QFont("Segoe UI", 12))
        ready_title.setAlignment(QtCore.Qt.AlignCenter)
        ready_layout.addWidget(ready_title)
        summary_group = QtWidgets.QGroupBox("📋 Installation Summary")
        summary_group.setFont(QtGui.QFont("Segoe UI", 11, QtGui.QFont.Bold))
        summary_layout = QtWidgets.QVBoxLayout(summary_group)
        self.summary_label = QtWidgets.QLabel("")  # content filled in dynamically
        self.summary_label.setFont(QtGui.QFont("Segoe UI", 9))
        self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label)
        ready_layout.addWidget(summary_group)
        warning_group = QtWidgets.QGroupBox("⚠️ Important")
        warning_group.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.Bold))
        warning_group.setStyleSheet("QGroupBox { color: #D32F2F; }")
        warning_layout = QtWidgets.QVBoxLayout(warning_group)
        warning_text = (
            "System restart will be required after WSL installation. "
            "The installer will handle this automatically and resume after restart."
        )
        warning_label = QtWidgets.QLabel(warning_text)
        warning_label.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold))
        warning_label.setWordWrap(True)
        warning_layout.addWidget(warning_label)
        ready_layout.addWidget(warning_group)
        self.stacked_pages.addWidget(ready_page)

        # Page 4: Installation Progress (executing tasks)
        install_page = QtWidgets.QWidget()
        install_layout = QtWidgets.QVBoxLayout(install_page)
        install_layout.setAlignment(QtCore.Qt.AlignTop)
        install_title = QtWidgets.QLabel("🚀 Installing Obsidian-Milvus-FastMCP")
        install_title.setFont(QtGui.QFont("Segoe UI", 16, QtGui.QFont.Bold))
        install_title.setAlignment(QtCore.Qt.AlignCenter)
        install_layout.addWidget(install_title)
        status_group = QtWidgets.QGroupBox("Installation Progress")
        status_group.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.Bold))
        status_layout = QtWidgets.QVBoxLayout(status_group)
        self.action_label = QtWidgets.QLabel("🔄 Preparing installation...")
        self.action_label.setFont(QtGui.QFont("Segoe UI", 11, QtGui.QFont.Bold))
        self.action_label.setAlignment(QtCore.Qt.AlignCenter)
        status_layout.addWidget(self.action_label)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximum(100)
        status_layout.addWidget(self.progress_bar)
        self.progress_percent = QtWidgets.QLabel("0%")
        self.progress_percent.setFont(QtGui.QFont("Segoe UI", 9))
        self.progress_percent.setAlignment(QtCore.Qt.AlignCenter)
        status_layout.addWidget(self.progress_percent)
        install_layout.addWidget(status_group)
        details_group = QtWidgets.QGroupBox("Installation Details")
        details_group.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.Bold))
        details_layout = QtWidgets.QVBoxLayout(details_group)
        self.details_text = QtWidgets.QPlainTextEdit()
        self.details_text.setFont(QtGui.QFont("Consolas", 8))
        self.details_text.setReadOnly(True)
        details_layout.addWidget(self.details_text)
        install_layout.addWidget(details_group)
        self.stacked_pages.addWidget(install_page)

        # Page 5: Configuration (post-reboot Obsidian vault & Podman paths)
        config_page = QtWidgets.QWidget()
        config_layout = QtWidgets.QVBoxLayout(config_page)
        config_layout.setAlignment(QtCore.Qt.AlignTop)
        config_label = QtWidgets.QLabel("Configure Obsidian Vault and Podman Path")
        config_label.setFont(QtGui.QFont("Segoe UI", 14, QtGui.QFont.Bold))
        config_layout.addWidget(config_label)
        vault_label = QtWidgets.QLabel("Obsidian Vault Path:")
        vault_label.setFont(QtGui.QFont("Segoe UI", 10))
        config_layout.addWidget(vault_label)
        vault_layout = QtWidgets.QHBoxLayout()
        self.vault_edit = QtWidgets.QLineEdit()
        self.vault_edit.setText(self.obsidian_vault_path)
        vault_layout.addWidget(self.vault_edit)
        vault_browse_btn = QtWidgets.QPushButton("Browse...")
        def browse_vault():
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Obsidian Vault Folder", "")
            if folder:
                self.vault_edit.setText(folder)
        vault_browse_btn.clicked.connect(browse_vault)
        vault_layout.addWidget(vault_browse_btn)
        config_layout.addLayout(vault_layout)
        podman_label = QtWidgets.QLabel("Podman Installation Path (optional, leave blank for auto-detect):")
        podman_label.setFont(QtGui.QFont("Segoe UI", 10))
        config_layout.addWidget(podman_label)
        podman_layout = QtWidgets.QHBoxLayout()
        self.podman_edit = QtWidgets.QLineEdit()
        self.podman_edit.setText(self.podman_path)
        podman_layout.addWidget(self.podman_edit)
        podman_browse_btn = QtWidgets.QPushButton("Browse...")
        def browse_podman():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Podman Executable", "", "Executables (*.exe)")
            if path:
                self.podman_edit.setText(path)
        podman_browse_btn.clicked.connect(browse_podman)
        podman_layout.addWidget(podman_browse_btn)
        config_layout.addLayout(podman_layout)
        note_label = QtWidgets.QLabel("* If Podman path is left empty, it will be auto-detected.")
        note_label.setFont(QtGui.QFont("Segoe UI", 8))
        config_layout.addWidget(note_label)
        self.stacked_pages.addWidget(config_page)

        # Page 6: Completion (final screen with links)
        complete_page = QtWidgets.QWidget()
        comp_layout = QtWidgets.QVBoxLayout(complete_page)
        comp_layout.setAlignment(QtCore.Qt.AlignTop)
        comp_title = QtWidgets.QLabel("🎉 Installation Complete! 🎉")
        comp_title.setFont(QtGui.QFont("Segoe UI", 18, QtGui.QFont.Bold))
        comp_title.setStyleSheet("color: #4CAF50;")
        comp_title.setAlignment(QtCore.Qt.AlignCenter)
        comp_layout.addWidget(comp_title)
        comp_msg = QtWidgets.QLabel("Obsidian-Milvus-FastMCP has been installed successfully.")
        comp_msg.setFont(QtGui.QFont("Segoe UI", 12))
        comp_msg.setAlignment(QtCore.Qt.AlignCenter)
        comp_layout.addWidget(comp_msg)
        next_steps_group = QtWidgets.QGroupBox("🚀 Next Steps")
        next_steps_group.setFont(QtGui.QFont("Segoe UI", 11, QtGui.QFont.Bold))
        next_steps_layout = QtWidgets.QVBoxLayout(next_steps_group)
        next_label = QtWidgets.QLabel(
            "You can now start using Obsidian semantic search. For additional setup or to verify installation, use the links below:"
        )
        next_label.setFont(QtGui.QFont("Segoe UI", 10))
        next_label.setWordWrap(True)
        next_steps_layout.addWidget(next_label)
        podman_link = QtWidgets.QLabel(
            '<a href="https://share.note.sx/r6kx06pj#78CIGnxLJYkJG+ZrQKYQhU35gtl+nKa47ZllwEyfUE0">Podman Auto-Launch Guide</a>'
        )
        podman_link.setOpenExternalLinks(True)
        podman_link.setFont(QtGui.QFont("Segoe UI", 10))
        next_steps_layout.addWidget(podman_link)
        milvus_link = QtWidgets.QLabel(
            '<a href="https://share.note.sx/y9vrzgj6#zr1aL4s1WFBK/A4WvqvkP6ETVMC4sKcAwbqAt4NyZhk">Milvus Auto-Launch Guide</a>'
        )
        milvus_link.setOpenExternalLinks(True)
        milvus_link.setFont(QtGui.QFont("Segoe UI", 10))
        next_steps_layout.addWidget(milvus_link)
        comp_layout.addWidget(next_steps_group)
        self.stacked_pages.addWidget(complete_page)

        # Map page widgets to names for reference
        self.pages_index_map = {
            'welcome': 0,
            'conda': 1,
            'install_path': 2,
            'ready': 3,
            'installing': 4,
            'config': 5,
            'complete': 6
        }

        # Trigger automatic Conda detection shortly after showing the conda page
        QtCore.QTimer.singleShot(1500, self.auto_check_conda)

    def update_progress_label(self):
        """Update the progress label text (Step X of Y)."""
        # Define total steps for each mode (fresh install vs resume)
        if self.resuming:
            # When resuming, only consider config, installing, and complete pages as steps
            total_steps = 3
            # Determine current step index among those pages
            if self.current_page_name == 'config':
                current_step = 1
            elif self.current_page_name == 'installing':
                current_step = 2
            else:
                current_step = 3
        else:
            # In fresh install, all pages including final complete count as steps
            total_steps = 7  # welcome(1), conda(2), install_path(3), ready(4), installing(5), config(6), complete(7)
            # Determine current page index in order
            current_step = {
                'welcome': 1, 'conda': 2, 'install_path': 3,
                'ready': 4, 'installing': 5, 'config': 6, 'complete': 7
            }.get(self.current_page_name, 1)
        self.progress_label.setText(f"Step {current_step} of {total_steps}")

    def show_summary(self):
        """Populate the summary page with chosen installation options."""
        install_path = self.path_edit.text().strip()
        vault_path = self.vault_edit.text().strip() or "Not configured (can be set later)"
        summary_text = (
            f"🎯 Installation Mode: Administrator (System-wide installation)\n\n"
            f"📁 Installation Path:\n{install_path}\n\n"
            f"📝 Obsidian Vault:\n{vault_path}\n\n"
            "🔧 Components to install:\n"
            "• Repository cloning from GitHub\n"
            "• Python dependencies via Conda (base environment)\n"
            "• Podman container runtime (for containers)\n"
            "• WSL and Ubuntu 22.04 (Linux subsystem)\n"
            "• Milvus vector database (containers via Podman)\n"
            "• Auto-startup services for Podman/Milvus\n"
            "• Claude Desktop integration\n\n"
            "⚠️ System Changes:\n"
            "• WSL feature will be enabled on Windows\n"
            "• Podman will be installed\n"
            "• System startup tasks will be created\n"
            "• Container runtime and database will be configured"
        )
        self.summary_label.setText(summary_text)

    def auto_check_conda(self):
        """Automatically check if Conda is installed on PATH."""
        conda_found = False
        conda_info = ""
        possible_commands = ['conda', 'conda.exe']
        possible_paths = [
            rf"C:\ProgramData\Anaconda3\Scripts\conda.exe",
            rf"C:\Users\{os.environ.get('USERNAME','')}\Anaconda3\Scripts\conda.exe",
            rf"C:\Users\{os.environ.get('USERNAME','')}\Miniconda3\Scripts\conda.exe"
        ]
        # Try direct commands
        for cmd in possible_commands:
            try:
                result = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    conda_found = True
                    conda_info = result.stdout.strip()
                    break
            except Exception:
                continue
        # Try specific known paths if not found via PATH
        if not conda_found:
            for path in possible_paths:
                if os.path.exists(path):
                    try:
                        result = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10)
                        if result.returncode == 0:
                            conda_found = True
                            conda_info = result.stdout.strip()
                            break
                    except Exception:
                        continue
        # Stop spinner and update status
        self.conda_progress.setRange(0, 100)
        self.conda_progress.setValue(100 if conda_found else 0)
        if conda_found:
            self.conda_status_label.setText(f"✅ Conda detected: {conda_info}")
            self.conda_status_label.setStyleSheet("color: #4CAF50;")
            self.next_button.setDisabled(False)
        else:
            self.conda_status_label.setText("❌ Conda not found. Please install Anaconda before continuing.")
            self.conda_status_label.setStyleSheet("color: #F44336;")
            self.conda_manual_button.setVisible(True)
            self.next_button.setDisabled(True)

    def check_conda(self):
        """Re-check Conda availability (when user clicks 'I have installed Conda')."""
        self.conda_status_label.setText("Re-checking Conda installation...")
        self.conda_status_label.setStyleSheet("color: black;")
        self.conda_progress.setRange(0, 0)
        QtCore.QTimer.singleShot(1500, self.auto_check_conda)

    def next_page(self):
        """Handle the Next/Install button logic for moving between pages."""
        current_idx = self.stacked_pages.currentIndex()
        # Validate inputs on certain pages
        if current_idx == self.pages_index_map['install_path']:
            chosen_path = self.path_edit.text().strip()
            if not chosen_path:
                QtWidgets.QMessageBox.warning(self, "Path Required", "Please select an installation path.")
                return
            # If directory exists and is not empty, warn the user
            if os.path.isdir(chosen_path) and os.listdir(chosen_path):
                reply = QtWidgets.QMessageBox.question(
                    self, "Directory Not Empty",
                    "The selected directory is not empty. The repository will be cloned into this folder and existing files may be overwritten.\n\nContinue?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                )
                if reply != QtWidgets.QMessageBox.Yes:
                    return
        # Populate summary before showing the ready page
        if current_idx == self.pages_index_map['install_path']:
            self.install_path = self.path_edit.text().strip()
            # (Vault path can be set later, we include current value if any for summary)
            self.show_summary()
        # If moving from Ready page to actual installation (Install clicked)
        if current_idx == self.pages_index_map['ready']:
            # Save state (phase 1 complete pending reboot)
            self.save_state(phase=1)
            # Disable navigation during installation
            self.back_button.setDisabled(True)
            self.next_button.setDisabled(True)
            # Switch to installing page and start background installation thread (Phase 1)
            self.stacked_pages.setCurrentIndex(self.pages_index_map['installing'])
            self.current_page_name = 'installing'
            self.update_progress_label()
            threading.Thread(target=self.perform_installation_phase1, daemon=True).start()
            return
        # If moving from Configuration page to finalize installation (after reboot)
        if current_idx == self.pages_index_map['config']:
            vault_path = self.vault_edit.text().strip()
            if not vault_path:
                QtWidgets.QMessageBox.warning(self, "Vault Path Required", "Please select your Obsidian vault path to continue.")
                return
            self.obsidian_vault_path = vault_path
            self.podman_path = self.podman_edit.text().strip()
            # Save updated state (phase 2, finalizing)
            self.save_state(phase=2)
            # Switch to installing page and start final tasks thread (Phase 2)
            self.stacked_pages.setCurrentIndex(self.pages_index_map['installing'])
            self.current_page_name = 'installing'
            # Update UI elements for final phase
            self.action_label.setText("🔄 Finalizing installation...")
            self.progress_bar.setValue(0)
            self.progress_percent.setText("0%")
            self.details_text.clear()
            self.back_button.setDisabled(True)
            self.next_button.setDisabled(True)
            self.update_progress_label()
            threading.Thread(target=self.perform_installation_phase2, daemon=True).start()
            return
        # Otherwise, just go to the next page in sequence
        self.stacked_pages.setCurrentIndex(current_idx + 1)
        # Determine the new current page name
        for name, idx in self.pages_index_map.items():
            if idx == current_idx + 1:
                self.current_page_name = name
                break
        # Enable/disable Back button as needed
        self.back_button.setDisabled(self.stacked_pages.currentIndex() == 0 or self.current_page_name in ['installing', 'complete'])
        # Update Next button text for certain pages
        if self.current_page_name == 'ready':
            self.next_button.setText("Install")
        elif self.current_page_name == 'complete':
            self.next_button.setText("Finish")
            # On completion page, enable Finish button and disable Back
            self.back_button.setDisabled(True)
        else:
            self.next_button.setText("Next >")
        self.update_progress_label()

    def previous_page(self):
        """Handle the Back button to return to the previous page."""
        current_idx = self.stacked_pages.currentIndex()
        if current_idx > 0:
            self.stacked_pages.setCurrentIndex(current_idx - 1)
            # Update the page name tracker
            for name, idx in self.pages_index_map.items():
                if idx == current_idx - 1:
                    self.current_page_name = name
                    break
            # Adjust Back button enabled state
            self.back_button.setDisabled(self.stacked_pages.currentIndex() == 0)
            # Adjust Next button text if we moved back from Install/Finish
            if self.current_page_name == 'ready':
                self.next_button.setText("Install")
            else:
                self.next_button.setText("Next >")
            self.update_progress_label()

    def save_state(self, phase=0):
        """Save current installation state to file for resuming after reboot."""
        state = {
            'install_path': self.path_edit.text().strip(),
            'obsidian_vault_path': self.vault_edit.text().strip(),
            'podman_path': self.podman_edit.text().strip(),
            'phase': phase
        }
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    # --- Installation Phases ---

    def perform_installation_phase1(self):
        """Execute phase 1 installation tasks (up to WSL setup) on a background thread."""
        steps = [
            ("🔍 Verifying system requirements...", self.verify_requirements, 5),
            ("📂 Creating installation directory...", self.create_install_dir, 10),
            ("📥 Cloning repository from GitHub...", self.clone_repository, 30),
            ("🐍 Installing Python dependencies...", self.install_dependencies, 60),
            ("🐳 Installing Podman container runtime...", self.install_podman, 75),
            ("🐧 Configuring WSL and Ubuntu...", self.setup_wsl, 90)
        ]
        for desc, func, prog in steps:
            self.update_action(desc, prog)
            try:
                func()
            except Exception as e:
                self.log_detail(f"❌ Error: {e}")
                # On error, enable Next to allow retry or cancel
                self.next_button.setDisabled(False)
                return
        # After completing phase 1 tasks:
        if self.is_compiled:
            # For compiled installer, schedule auto-reboot and resume
            self.log_detail("\n⚠️ A system reboot is required to continue installation.")
            # Register this installer to run once after reboot (Windows RunOnce)
            exe_path = sys.argv[0]
            try:
                reg_cmd = [
                    'REG', 'ADD', r'HKCU\Software\Microsoft\Windows\CurrentVersion\RunOnce',
                    '/v', 'ObsidianMilvusInstaller', '/t', 'REG_SZ', '/d', f'"{exe_path}"', '/f'
                ]
                subprocess.run(reg_cmd, check=True, stdout=subprocess.DEVNULL)
                self.log_detail("✔️ Installer will resume automatically after reboot.")
            except Exception as e:
                self.log_detail(f"❌ Failed to register RunOnce: {e}")
            # Initiate reboot in 30 seconds to allow user to read logs
            try:
                subprocess.run(['shutdown', '/r', '/t', '30'], check=False)
                self.log_detail("🔃 System will reboot in 30 seconds. Save any unsaved work.")
            except Exception as e:
                self.log_detail(f"❌ Failed to initiate reboot: {e}")
            # (The application will be closed by system reboot)
        else:
            # If running as script (not compiled), allow continuing without reboot
            self.log_detail("\n⚠️ Reboot skipped (developer mode). Continuing installation without reboot...")
            # Directly proceed with phase 2 tasks in the same session
            self.perform_installation_phase2()

    def perform_installation_phase2(self):
        """Execute phase 2 installation tasks (post-reboot) on a background thread."""
        steps = [
            ("🔧 Installing podman-compose...", self.install_podman_compose, 15),
            ("⚙️ Updating configuration files...", self.update_config_file, 30),
            ("🔍 Detecting Podman path...", self.find_podman_path, 50),
            ("♻️ Resetting Podman environment...", self.reset_podman, 60),
            ("🚀 Starting FastMCP services...", self.start_mcp_services, 70),
            ("🛠️ Running final setup script...", self.run_setup_script, 90)
        ]
        for desc, func, prog in steps:
            self.update_action(desc, prog)
            try:
                func()
            except Exception as e:
                self.log_detail(f"❌ Error during final setup: {e}")
                self.next_button.setDisabled(False)
                return
        # Final wrap-up
        self.update_action("✅ Finalizing installation...", 100)
        self.log_detail("\n🎉 Installation completed successfully!")
        # Installation done: remove state file
        try:
            if self.state_file.exists():
                self.state_file.unlink()
        except Exception:
            pass
        # Enable Finish button and disable Cancel on completion
        self.next_button.setText("Finish")
        self.next_button.setDisabled(False)
        self.cancel_button.setDisabled(True)

    # --- Individual step functions ---

    def update_action(self, text, progress):
        """Update the current action label and progress bar/percentage."""
        QtCore.QMetaObject.invokeMethod(
            self.action_label, "setText", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, text)
        )
        QtCore.QMetaObject.invokeMethod(
            self.progress_bar, "setValue", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(int, progress)
        )
        QtCore.QMetaObject.invokeMethod(
            self.progress_percent, "setText", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, f"{progress}%")
        )
        self.log_detail(f"\n{text}")

    def log_detail(self, message):
        """Append a message line to the details log box (thread-safe)."""
        QtCore.QMetaObject.invokeMethod(
            self.details_text, "appendPlainText", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, message)
        )

    def verify_requirements(self):
        """Verify basic system requirements (Conda, Git, etc.)."""
        self.log_detail("Verifying system requirements...")
        # Ensure git is available for cloning
        if not shutil.which("git"):
            raise Exception("git is not installed or not in PATH")
        # (Conda was already verified in earlier step)
        self.log_detail("System requirements check passed.")

    def create_install_dir(self):
        """Create or verify the installation directory."""
        target_dir = Path(self.install_path)
        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
        self.log_detail(f"Using installation directory: {target_dir}")

    def clone_repository(self):
        """Clone the Obsidian-Milvus-FastMCP repository into the chosen path."""
        repo_url = "https://github.com/jayjeo/obsidian-milvus-FastMCP.git"
        target_dir = Path(self.install_path)
        self.log_detail(f"Cloning repository from GitHub into {target_dir} ...")
        result = subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target_dir)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            self.log_detail(result.stdout.strip())
            raise Exception("Git clone failed")
        self.log_detail("✅ Repository cloned successfully")

    def install_dependencies(self):
        """Install required Python packages using Conda/Pip."""
        packages = [
            "pymilvus", "mcp", "fastmcp", "sentence-transformers", "torch",
            "PyPDF2", "markdown", "beautifulsoup4", "python-dotenv", "watchdog",
            "psutil", "colorama", "pyyaml", "tqdm", "requests"
        ]
        # Use pip to install packages into the base conda environment
        pip_cmd_base = [sys.executable, "-m", "pip", "install"] if not self.is_compiled else ["pip", "install"]
        total = len(packages)
        for idx, pkg in enumerate(packages, start=1):
            self.log_detail(f"Installing {pkg}... ({idx}/{total})")
            result = subprocess.run(pip_cmd_base + [pkg], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if result.returncode != 0:
                self.log_detail(result.stdout.strip())
                raise Exception(f"Failed to install {pkg}")
        self.log_detail("✅ Python dependencies installed successfully")

    def install_podman(self):
        """Install Podman using Windows Package Manager (winget)."""
        self.log_detail("Installing Podman via winget...")
        cmd = [
            "winget", "install", "--id=RedHat.Podman", "-e",
            "--accept-package-agreements", "--accept-source-agreements", "--silent"
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            self.log_detail(result.stdout.strip())
            raise Exception("Podman installation failed")
        self.log_detail("✅ Podman installed successfully")

    def setup_wsl(self):
        """Enable WSL 2 and install Ubuntu 22.04 distribution."""
        self.log_detail("Enabling Windows Subsystem for Linux (WSL)...")
        # Enable required Windows features for WSL2 (no immediate restart)
        subprocess.run(
            ["dism.exe", "/online", "/enable-feature", "/featurename:VirtualMachinePlatform", "/all", "/norestart"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        subprocess.run(
            ["dism.exe", "/online", "/enable-feature", "/featurename:Microsoft-Windows-Subsystem-Linux", "/all", "/norestart"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        # Set default WSL version to 2 (ignore errors if any)
        subprocess.run(["wsl", "--set-default-version", "2"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.log_detail("WSL features enabled. Installing Ubuntu 22.04 LTS via winget...")
        # Install Ubuntu 22.04 LTS distribution through winget
        cmd = [
            "winget", "install", "--id=Canonical.Ubuntu.2204", "-e",
            "--accept-package-agreements", "--accept-source-agreements", "--silent"
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            self.log_detail("⚠️ Ubuntu 22.04 LTS installation will complete after reboot.")
        else:
            self.log_detail("✅ Ubuntu 22.04 LTS distribution installed (WSL)")

    def install_podman_compose(self):
        """Install podman-compose Python package (to manage containers with Podman)."""
        self.log_detail("Installing podman-compose via pip...")
        result = subprocess.run(["pip", "install", "podman-compose"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            self.log_detail(result.stdout.strip())
            raise Exception("podman-compose installation failed")
        self.log_detail("✅ podman-compose installed successfully")

    def update_config_file(self):
        """Write user-provided Obsidian vault path and Podman path into config.py."""
        config_path = Path(self.install_path) / "config.py"
        if not config_path.exists():
            self.log_detail(f"⚠️ config.py not found in {self.install_path}, skipping configuration update")
            return
        text = config_path.read_text(encoding='utf-8')
        vault_path_str = self.obsidian_vault_path.replace("\\", "\\\\")
        podman_path_str = self.podman_path.replace("\\", "\\\\")
        # Replace the vault path line
        import re
        text = re.sub(r'OBSIDIAN_VAULT_PATH\s*=\s*".*"', f'OBSIDIAN_VAULT_PATH = "{vault_path_str}"', text)
        # Replace the podman path line if provided, otherwise leave it empty
        if podman_path_str:
            text = re.sub(r'PODMAN_PATH\s*=\s*".*"', f'PODMAN_PATH = "{podman_path_str}"', text)
        config_path.write_text(text, encoding='utf-8')
        self.log_detail("✔️ config.py updated with Obsidian vault and Podman path")

    def find_podman_path(self):
        """Run the find_podman_path.bat script to auto-detect Podman installation path if needed."""
        if self.podman_path:
            # If user manually provided Podman path, skip auto-detection
            self.log_detail(f"Using user-provided Podman path: {self.podman_path}")
            return
        script_path = Path(self.install_path) / "find_podman_path.bat"
        self.log_detail("Executing find_podman_path.bat to detect Podman path...")
        if not script_path.exists():
            self.log_detail("⚠️ find_podman_path.bat not found, skipping Podman path detection")
            return
        result = subprocess.run(str(script_path), cwd=self.install_path,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)
        output = result.stdout.strip()
        if output:
            self.log_detail(output)
        if result.returncode != 0:
            self.log_detail("⚠️ Podman path detection failed or Podman not found in default locations.")

    def reset_podman(self):
        """Run script to reset Podman environment (remove existing containers, if any)."""
        script_path = Path(self.install_path) / "complete-podman-reset.bat"
        self.log_detail("Resetting Podman environment...")
        if not script_path.exists():
            self.log_detail("⚠️ complete-podman-reset.bat not found, skipping reset")
            return
        result = subprocess.run(str(script_path), cwd=self.install_path,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)
        if result.stdout:
            self.log_detail(result.stdout.strip())

    def start_mcp_services(self):
        """Run script to start FastMCP services (with encoding fix)."""
        script_path = Path(self.install_path) / "start_mcp_with_encoding_fix.bat"
        self.log_detail("Starting FastMCP services (with encoding fix)...")
        if not script_path.exists():
            self.log_detail("⚠️ start_mcp_with_encoding_fix.bat not found, skipping service start")
            return
        result = subprocess.run(str(script_path), cwd=self.install_path,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)
        if result.stdout:
            self.log_detail(result.stdout.strip())

    def run_setup_script(self):
        """Execute the interactive run-setup.bat script and auto-select options 1-5."""
        self.log_detail("Running final setup script (run-setup.bat) to configure components...")
        script_path = Path(self.install_path) / "run-setup.bat"
        if not script_path.exists():
            self.log_detail("⚠️ run-setup.bat not found, skipping final setup steps")
            return
        # Start the batch file process
        process = subprocess.Popen(["cmd", "/c", str(script_path)], cwd=self.install_path,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # Automatically send selections 1 through 5, then send an exit (option 9 or Enter)
        try:
            for choice in ['1', '2', '3', '4', '5']:
                self.log_detail(f"Selecting option {choice}...")
                process.stdin.write(choice + "\n")
                process.stdin.flush()
                time.sleep(1)  # wait a bit for each operation to start
            # Attempt to exit the menu (option 9 assumed to be exit, adjust if different)
            process.stdin.write("9\n")
            process.stdin.flush()
        except Exception as e:
            self.log_detail(f"❌ Error sending input to run-setup.bat: {e}")
        # Read all remaining output from the process
        output, _ = process.communicate(timeout=120)
        if output:
            self.log_detail(output.strip())
        if process.returncode and process.returncode != 0:
            raise Exception("run-setup.bat did not complete successfully")

    def cancel_installation(self):
        """Confirm cancellation and exit the installer."""
        reply = QtWidgets.QMessageBox.question(
            self, "Cancel Setup",
            "Are you sure you want to cancel the installation?\n\nAny progress will be lost.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            try:
                if self.state_file.exists():
                    self.state_file.unlink()
            except Exception:
                pass
            QtWidgets.QApplication.quit()

# Run the installer if this script is executed directly
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    installer = ObsidianMilvusInstaller()
    installer.show()
    sys.exit(app.exec_())
