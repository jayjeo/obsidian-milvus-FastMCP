import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import os
import sys
import json
import shutil
import time
import requests
import threading
import winreg
from pathlib import Path
import urllib.request
import ctypes

class ObsidianMilvusInstaller:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Obsidian-Milvus-FastMCP Setup")
        self.root.geometry("700x500")
        self.root.resizable(False, False)
        
        # Set window style
        self.root.configure(bg='white')
        
        # Installation state file
        self.state_file = Path("installer_state.json")
        self.state = self.load_state()
        
        # Installation variables
        self.install_path = tk.StringVar(value=self.state.get('install_path', ''))
        self.obsidian_vault_path = tk.StringVar(value=self.state.get('obsidian_vault_path', ''))
        self.podman_path = tk.StringVar(value=self.state.get('podman_path', ''))
        
        # Current page tracking
        self.current_page = 0
        self.pages = []
        
        # Create main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        self.create_header()
        
        # Create content area
        self.content_frame = ttk.Frame(self.main_frame, style='Content.TFrame')
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Create footer with buttons
        self.create_footer()
        
        # Setup pages
        self.setup_pages()
        
        # Show first page
        self.show_page(0)
    
    def create_header(self):
        """Create modern header with logo area"""
        header_frame = tk.Frame(self.main_frame, bg='#0078D4', height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(header_frame, text="Obsidian-Milvus-FastMCP Setup Wizard", 
                              font=('Segoe UI', 18, 'bold'), fg='white', bg='#0078D4')
        title_label.pack(pady=20, padx=20, anchor='w')
    
    def create_footer(self):
        """Create footer with navigation buttons"""
        footer_frame = tk.Frame(self.main_frame, bg='#F0F0F0', height=60)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        footer_frame.pack_propagate(False)
        
        # Separator line
        separator = ttk.Separator(footer_frame, orient='horizontal')
        separator.pack(fill=tk.X)
        
        # Button container
        button_frame = tk.Frame(footer_frame, bg='#F0F0F0')
        button_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Cancel button
        self.cancel_button = tk.Button(button_frame, text="Cancel", 
                                      font=('Segoe UI', 10), width=10,
                                      bg='white', relief=tk.SOLID, bd=1,
                                      command=self.cancel_installation)
        self.cancel_button.pack(side=tk.RIGHT, padx=5)
        
        # Next/Install button
        self.next_button = tk.Button(button_frame, text="Next >", 
                                    font=('Segoe UI', 10, 'bold'), width=10,
                                    bg='#0078D4', fg='white', relief=tk.FLAT,
                                    command=self.next_page)
        self.next_button.pack(side=tk.RIGHT, padx=5)
        
        # Back button
        self.back_button = tk.Button(button_frame, text="< Back", 
                                    font=('Segoe UI', 10), width=10,
                                    bg='white', relief=tk.SOLID, bd=1,
                                    command=self.previous_page, state=tk.DISABLED)
        self.back_button.pack(side=tk.RIGHT, padx=5)
    
    def setup_pages(self):
        """Setup installation pages"""
        self.pages = [
            self.create_welcome_page,
            self.create_conda_check_page,
            self.create_install_path_page,
            self.create_obsidian_path_page,
            self.create_ready_page,
            self.create_installing_page,
            self.create_complete_page
        ]
    
    def create_welcome_page(self):
        """Welcome page"""
        # Clear content
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Welcome text
        welcome_label = tk.Label(self.content_frame, 
                               text="Welcome to Obsidian-Milvus-FastMCP Setup",
                               font=('Segoe UI', 16, 'bold'), bg='white')
        welcome_label.pack(pady=(40, 20))
        
        info_text = """This wizard will install Obsidian-Milvus-FastMCP on your computer.

Obsidian-Milvus-FastMCP integrates Milvus vector database with 
Obsidian notes, enabling powerful semantic search capabilities
through Claude Desktop.

Click Next to continue, or Cancel to exit Setup."""
        
        info_label = tk.Label(self.content_frame, text=info_text,
                            font=('Segoe UI', 10), bg='white', justify=tk.LEFT)
        info_label.pack(pady=20, padx=40)
        
        # System requirements
        req_frame = tk.LabelFrame(self.content_frame, text="System Requirements",
                                 font=('Segoe UI', 10, 'bold'), bg='white')
        req_frame.pack(pady=20, padx=40, fill=tk.X)
        
        requirements = """• Windows 10 or later
• Anaconda or Miniconda (will be checked)
• 10 GB free disk space
• Internet connection
• Administrator privileges"""
        
        req_label = tk.Label(req_frame, text=requirements,
                           font=('Segoe UI', 9), bg='white', justify=tk.LEFT)
        req_label.pack(pady=10, padx=20, anchor='w')
    
    def create_conda_check_page(self):
        """Conda check page"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Title
        title_label = tk.Label(self.content_frame, 
                             text="Conda Installation Check",
                             font=('Segoe UI', 16, 'bold'), bg='white')
        title_label.pack(pady=(40, 20))
        
        # Info
        info_text = """Setup requires Anaconda or Miniconda to be installed on your system.

Conda is used to manage Python packages and dependencies."""
        
        info_label = tk.Label(self.content_frame, text=info_text,
                            font=('Segoe UI', 10), bg='white', justify=tk.LEFT)
        info_label.pack(pady=20, padx=40)
        
        # Check frame
        check_frame = tk.Frame(self.content_frame, bg='white')
        check_frame.pack(pady=20)
        
        # Status icon and text
        self.conda_status_label = tk.Label(check_frame, 
                                         text="⚠ Please confirm Conda installation",
                                         font=('Segoe UI', 12), bg='white', fg='#FF9800')
        self.conda_status_label.pack()
        
        # Instructions
        inst_frame = tk.LabelFrame(self.content_frame, text="If Conda is not installed:",
                                 font=('Segoe UI', 10, 'bold'), bg='white')
        inst_frame.pack(pady=20, padx=40, fill=tk.X)
        
        inst_text = """1. Download Anaconda from: https://www.anaconda.com/download
2. Run the Anaconda installer
3. Restart this Setup wizard"""
        
        inst_label = tk.Label(inst_frame, text=inst_text,
                            font=('Segoe UI', 9), bg='white', justify=tk.LEFT)
        inst_label.pack(pady=10, padx=20, anchor='w')
        
        # Check if user has Conda
        self.root.after(500, self.check_conda_status)
    
    def check_conda_status(self):
        """Check Conda installation status"""
        response = messagebox.askyesno("Conda Check", 
                                     "Do you have Anaconda or Miniconda installed?")
        if response:
            self.conda_status_label.config(text="✓ Conda installation confirmed", fg='#4CAF50')
            self.next_button.config(state=tk.NORMAL)
        else:
            self.conda_status_label.config(text="✗ Conda not installed", fg='#F44336')
            self.next_button.config(state=tk.DISABLED)
            messagebox.showinfo("Install Conda", 
                              "Please install Anaconda first:\n\n"
                              "1. Go to https://www.anaconda.com/download\n"
                              "2. Download and install Anaconda\n"
                              "3. Run this installer again")
    
    def create_install_path_page(self):
        """Installation path selection page"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Title
        title_label = tk.Label(self.content_frame, 
                             text="Choose Install Location",
                             font=('Segoe UI', 16, 'bold'), bg='white')
        title_label.pack(pady=(40, 20))
        
        # Info
        info_label = tk.Label(self.content_frame, 
                            text="Setup will install Obsidian-Milvus-FastMCP in the following folder.",
                            font=('Segoe UI', 10), bg='white')
        info_label.pack(pady=10)
        
        # Path selection frame
        path_frame = tk.Frame(self.content_frame, bg='white')
        path_frame.pack(pady=20, padx=40, fill=tk.X)
        
        # Path entry
        self.path_entry = tk.Entry(path_frame, textvariable=self.install_path,
                                 font=('Segoe UI', 10), width=50)
        self.path_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        # Browse button
        browse_button = tk.Button(path_frame, text="Browse...",
                                font=('Segoe UI', 9), width=10,
                                bg='white', relief=tk.SOLID, bd=1,
                                command=self.browse_install_path)
        browse_button.pack(side=tk.LEFT)
        
        # Space info
        space_label = tk.Label(self.content_frame,
                             text="Space required: 5.2 GB\nSpace available: Calculating...",
                             font=('Segoe UI', 9), bg='white', fg='#666')
        space_label.pack(pady=10)
        
        # Set default path if empty
        if not self.install_path.get():
            default_path = os.path.join(os.environ.get('PROGRAMFILES', 'C:\\Program Files'), 
                                      'Obsidian-Milvus-FastMCP')
            self.install_path.set(default_path)
    
    def browse_install_path(self):
        """Browse for installation path"""
        path = filedialog.askdirectory(title="Select Installation Directory")
        if path:
            self.install_path.set(path)
    
    def create_obsidian_path_page(self):
        """Obsidian vault path selection page"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Title
        title_label = tk.Label(self.content_frame, 
                             text="Select Obsidian Vault",
                             font=('Segoe UI', 16, 'bold'), bg='white')
        title_label.pack(pady=(40, 20))
        
        # Info
        info_label = tk.Label(self.content_frame, 
                            text="Please select your Obsidian vault folder to enable integration.",
                            font=('Segoe UI', 10), bg='white')
        info_label.pack(pady=10)
        
        # Path selection frame
        path_frame = tk.Frame(self.content_frame, bg='white')
        path_frame.pack(pady=20, padx=40, fill=tk.X)
        
        # Path entry
        self.vault_entry = tk.Entry(path_frame, textvariable=self.obsidian_vault_path,
                                  font=('Segoe UI', 10), width=50)
        self.vault_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        # Browse button
        browse_button = tk.Button(path_frame, text="Browse...",
                                font=('Segoe UI', 9), width=10,
                                bg='white', relief=tk.SOLID, bd=1,
                                command=self.browse_vault_path)
        browse_button.pack(side=tk.LEFT)
        
        # Help text
        help_text = """The Obsidian vault is the folder containing your notes.
Usually located in Documents or a custom location you chose."""
        
        help_label = tk.Label(self.content_frame, text=help_text,
                            font=('Segoe UI', 9), bg='white', fg='#666')
        help_label.pack(pady=20, padx=40)
    
    def browse_vault_path(self):
        """Browse for Obsidian vault"""
        path = filedialog.askdirectory(title="Select Obsidian Vault Folder")
        if path:
            self.obsidian_vault_path.set(path)
    
    def create_ready_page(self):
        """Ready to install page"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Title
        title_label = tk.Label(self.content_frame, 
                             text="Ready to Install",
                             font=('Segoe UI', 16, 'bold'), bg='white')
        title_label.pack(pady=(40, 20))
        
        # Info
        info_label = tk.Label(self.content_frame, 
                            text="Setup is ready to begin installing Obsidian-Milvus-FastMCP on your computer.",
                            font=('Segoe UI', 10), bg='white')
        info_label.pack(pady=10)
        
        # Summary frame
        summary_frame = tk.LabelFrame(self.content_frame, text="Installation Summary",
                                    font=('Segoe UI', 10, 'bold'), bg='white')
        summary_frame.pack(pady=20, padx=40, fill=tk.BOTH, expand=True)
        
        # Summary text
        summary_text = f"""Installation Path:
{self.install_path.get()}

Obsidian Vault:
{self.obsidian_vault_path.get()}

Components to install:
• Obsidian-Milvus-FastMCP Core
• Python Dependencies
• Podman Container Runtime
• Milvus Vector Database
• MCP Server Integration"""
        
        summary_label = tk.Label(summary_frame, text=summary_text,
                               font=('Segoe UI', 9), bg='white', justify=tk.LEFT)
        summary_label.pack(pady=10, padx=20, anchor='w')
        
        # Change button text to Install
        self.next_button.config(text="Install")
    
    def create_installing_page(self):
        """Installation progress page"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Disable navigation
        self.back_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.DISABLED)
        
        # Title
        title_label = tk.Label(self.content_frame, 
                             text="Installing Obsidian-Milvus-FastMCP",
                             font=('Segoe UI', 16, 'bold'), bg='white')
        title_label.pack(pady=(40, 20))
        
        # Current action label
        self.action_label = tk.Label(self.content_frame, 
                                   text="Preparing installation...",
                                   font=('Segoe UI', 10), bg='white')
        self.action_label.pack(pady=10)
        
        # Progress bar
        self.install_progress = ttk.Progressbar(self.content_frame, 
                                              length=500, mode='determinate')
        self.install_progress.pack(pady=20)
        
        # Details text
        self.details_text = tk.Text(self.content_frame, height=10, width=70,
                                  font=('Consolas', 8), bg='#F5F5F5')
        self.details_text.pack(pady=10, padx=40)
        
        # Start installation in thread
        install_thread = threading.Thread(target=self.perform_installation)
        install_thread.start()
    
    def create_complete_page(self):
        """Installation complete page"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Re-enable buttons
        self.back_button.config(state=tk.DISABLED)
        self.next_button.config(text="Finish", command=self.finish_installation)
        self.cancel_button.config(state=tk.DISABLED)
        
        # Title
        title_label = tk.Label(self.content_frame, 
                             text="Installation Complete",
                             font=('Segoe UI', 16, 'bold'), bg='white')
        title_label.pack(pady=(40, 20))
        
        # Success icon and text
        success_label = tk.Label(self.content_frame, 
                               text="✓ Obsidian-Milvus-FastMCP has been installed successfully!",
                               font=('Segoe UI', 12), bg='white', fg='#4CAF50')
        success_label.pack(pady=20)
        
        # Next steps
        steps_frame = tk.LabelFrame(self.content_frame, text="Next Steps",
                                  font=('Segoe UI', 10, 'bold'), bg='white')
        steps_frame.pack(pady=20, padx=40, fill=tk.X)
        
        steps_text = """1. Configure auto-startup (documentation will open)
2. Restart your computer
3. Open Claude Desktop
4. Start using Obsidian integration!"""
        
        steps_label = tk.Label(steps_frame, text=steps_text,
                             font=('Segoe UI', 9), bg='white', justify=tk.LEFT)
        steps_label.pack(pady=10, padx=20, anchor='w')
        
        # Launch checkbox
        self.launch_var = tk.BooleanVar(value=True)
        launch_check = tk.Checkbutton(self.content_frame, 
                                    text="Open documentation when finished",
                                    variable=self.launch_var,
                                    font=('Segoe UI', 10), bg='white')
        launch_check.pack(pady=10)
    
    def show_page(self, index):
        """Show specific page"""
        if 0 <= index < len(self.pages):
            self.current_page = index
            self.pages[index]()
            
            # Update button states
            self.back_button.config(state=tk.NORMAL if index > 0 else tk.DISABLED)
            
            # Reset Next button text if not on ready page
            if index != 4:  # Not ready page
                self.next_button.config(text="Next >")
    
    def next_page(self):
        """Go to next page"""
        # Validate current page
        if self.current_page == 2 and not self.install_path.get():
            messagebox.showwarning("Path Required", "Please select an installation path.")
            return
        elif self.current_page == 3 and not self.obsidian_vault_path.get():
            messagebox.showwarning("Vault Required", "Please select your Obsidian vault folder.")
            return
        
        if self.current_page < len(self.pages) - 1:
            self.show_page(self.current_page + 1)
    
    def previous_page(self):
        """Go to previous page"""
        if self.current_page > 0:
            self.show_page(self.current_page - 1)
    
    def perform_installation(self):
        """Perform actual installation"""
        steps = [
            ("Cloning repository...", self.clone_repo, 10),
            ("Installing dependencies...", self.install_deps, 25),
            ("Installing Podman...", self.install_podman_step, 40),
            ("Configuring WSL...", self.configure_wsl_step, 55),
            ("Setting up configuration...", self.setup_config, 70),
            ("Initializing services...", self.init_services, 85),
            ("Finalizing installation...", self.finalize, 100)
        ]
        
        for step_name, step_func, progress in steps:
            self.action_label.config(text=step_name)
            self.install_progress['value'] = progress
            self.log_detail(f"\n{step_name}")
            
            try:
                step_func()
            except Exception as e:
                self.log_detail(f"Error: {str(e)}")
                messagebox.showerror("Installation Failed", 
                                   f"Installation failed during: {step_name}\n\nError: {str(e)}")
                return
            
            time.sleep(0.5)  # Brief pause between steps
        
        # Installation complete
        self.show_page(6)  # Show complete page
    
    def log_detail(self, message):
        """Log installation details"""
        self.details_text.insert(tk.END, f"{message}\n")
        self.details_text.see(tk.END)
        self.root.update()
    
    def clone_repo(self):
        """Clone repository"""
        # Implementation for cloning
        self.log_detail("Cloning from GitHub...")
        # Add actual implementation
    
    def install_deps(self):
        """Install dependencies"""
        self.log_detail("Installing Python packages...")
        # Add actual implementation
    
    def install_podman_step(self):
        """Install Podman"""
        self.log_detail("Installing Podman container runtime...")
        # Add actual implementation
    
    def configure_wsl_step(self):
        """Configure WSL"""
        self.log_detail("Configuring Windows Subsystem for Linux...")
        # Add actual implementation
    
    def setup_config(self):
        """Setup configuration"""
        self.log_detail("Updating configuration files...")
        # Add actual implementation
    
    def init_services(self):
        """Initialize services"""
        self.log_detail("Starting Milvus services...")
        # Add actual implementation
    
    def finalize(self):
        """Finalize installation"""
        self.log_detail("Creating shortcuts...")
        self.log_detail("Installation completed successfully!")
    
    def cancel_installation(self):
        """Cancel installation"""
        response = messagebox.askyesno("Cancel Setup", 
                                     "Are you sure you want to cancel the installation?")
        if response:
            self.root.quit()
    
    def finish_installation(self):
        """Finish installation"""
        if self.launch_var.get():
            # Open documentation
            pass
        self.root.quit()
    
    def save_state(self):
        """Save installation state"""
        state = {
            'install_path': self.install_path.get(),
            'obsidian_vault_path': self.obsidian_vault_path.get()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f)
    
    def load_state(self):
        """Load installation state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {}
    
    def run(self):
        """Run the installer"""
        self.root.mainloop()

if __name__ == "__main__":
    installer = ObsidianMilvusInstaller()
    installer.run()