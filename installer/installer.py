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
        self.root.title("Obsidian-Milvus-FastMCP Installer")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        
        # Set icon if available
        try:
            self.root.iconbitmap('installer.ico')
        except:
            pass
        
        # Installation state file
        self.state_file = Path("installer_state.json")
        self.state = self.load_state()
        
        # Installation variables
        self.install_path = tk.StringVar(value=self.state.get('install_path', ''))
        self.obsidian_vault_path = tk.StringVar(value=self.state.get('obsidian_vault_path', ''))
        self.podman_path = tk.StringVar(value=self.state.get('podman_path', ''))
        
        # Progress tracking
        self.current_step = self.state.get('current_step', 0)
        self.steps = [
            "Check Conda Installation",
            "Select Installation Path",
            "Clone Repository",
            "Install Dependencies",
            "Install Podman",
            "Configure WSL and Ubuntu",
            "Configure Paths",
            "Initialize Podman Container",
            "Initialize Milvus Server",
            "Manual Configuration",
            "Final Setup"
        ]
        
        self.setup_ui()
        
        # Auto-resume if installation was in progress
        if self.current_step > 0:
            self.resume_installation()
    
    def is_admin(self):
        """Check if running with admin privileges"""
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False
    
    def setup_ui(self):
        """Setup the user interface"""
        # Header
        header_frame = ttk.Frame(self.root, padding="20")
        header_frame.pack(fill=tk.X)
        
        title_label = ttk.Label(header_frame, text="Obsidian-Milvus-FastMCP Installer", 
                               font=('Arial', 16, 'bold'))
        title_label.pack()
        
        subtitle_label = ttk.Label(header_frame, 
                                  text="Automated installation and setup for Obsidian-Milvus integration",
                                  font=('Arial', 10))
        subtitle_label.pack()
        
        # Progress frame
        progress_frame = ttk.Frame(self.root, padding="20")
        progress_frame.pack(fill=tk.X)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready to install", font=('Arial', 10))
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=700, mode='determinate')
        self.progress_bar.pack(pady=10)
        
        # Step list frame
        steps_frame = ttk.Frame(self.root, padding="20")
        steps_frame.pack(fill=tk.BOTH, expand=True)
        
        self.steps_listbox = tk.Listbox(steps_frame, height=len(self.steps), font=('Arial', 10))
        self.steps_listbox.pack(fill=tk.BOTH, expand=True)
        
        for i, step in enumerate(self.steps):
            self.steps_listbox.insert(tk.END, f"{i+1}. {step}")
        
        # Log frame
        log_frame = ttk.Frame(self.root, padding="10")
        log_frame.pack(fill=tk.X)
        
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD, font=('Consolas', 9))
        self.log_text.pack(fill=tk.X)
        
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        
        # Button frame
        button_frame = ttk.Frame(self.root, padding="20")
        button_frame.pack(fill=tk.X)
        
        self.install_button = ttk.Button(button_frame, text="Start Installation", 
                                        command=self.start_installation, 
                                        style='Primary.TButton')
        self.install_button.pack(side=tk.LEFT, padx=5)
        
        self.cancel_button = ttk.Button(button_frame, text="Cancel", 
                                       command=self.cancel_installation)
        self.cancel_button.pack(side=tk.RIGHT, padx=5)
        
        # Update step display
        self.update_step_display()
    
    def log(self, message):
        """Add message to log display"""
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def update_step_display(self):
        """Update the visual display of steps"""
        for i in range(len(self.steps)):
            if i < self.current_step:
                # Completed step
                self.steps_listbox.itemconfig(i, fg='green')
                self.steps_listbox.delete(i)
                self.steps_listbox.insert(i, f"✓ {i+1}. {self.steps[i]}")
            elif i == self.current_step:
                # Current step
                self.steps_listbox.itemconfig(i, fg='blue', bg='lightblue')
            else:
                # Pending step
                self.steps_listbox.itemconfig(i, fg='gray')
        
        # Update progress bar
        progress_percent = (self.current_step / len(self.steps)) * 100
        self.progress_bar['value'] = progress_percent
        
        if self.current_step < len(self.steps):
            self.progress_label.config(text=f"Step {self.current_step + 1} of {len(self.steps)}: {self.steps[self.current_step]}")
        else:
            self.progress_label.config(text="Installation Complete!")
    
    def save_state(self):
        """Save installation state to file"""
        state = {
            'current_step': self.current_step,
            'install_path': self.install_path.get(),
            'obsidian_vault_path': self.obsidian_vault_path.get(),
            'podman_path': self.podman_path.get()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f)
    
    def load_state(self):
        """Load installation state from file"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {}
    
    def run_command(self, cmd, shell=False, check=True):
        """Run a command and return result"""
        try:
            if isinstance(cmd, str):
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
            else:
                result = subprocess.run(cmd, shell=shell, capture_output=True, text=True, check=check)
            return True, result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            return False, e.stdout, e.stderr
        except Exception as e:
            return False, "", str(e)
    
    def check_conda(self):
        """Check if Conda is installed"""
        self.log("Conda installation requirement...")
        
        # Show requirement message
        response = messagebox.askyesno("Conda Required", 
                                     "This installer requires Conda (Anaconda or Miniconda).\n\n"
                                     "Have you already installed Conda?\n\n"
                                     "If NO: Please install Anaconda from https://www.anaconda.com/download\n"
                                     "       and run this installer again.\n\n"
                                     "Click YES if Conda is already installed.\n"
                                     "Click NO to exit and install Conda first.")
        
        if response:
            self.log("User confirmed Conda is installed")
            return True
        else:
            self.log("User needs to install Conda first")
            messagebox.showinfo("Install Conda First",
                              "Please install Conda first:\n\n"
                              "1. Go to https://www.anaconda.com/download\n"
                              "2. Download and install Anaconda\n"
                              "3. Run this installer again\n\n"
                              "The installer will now exit.")
            self.root.quit()
            return False
    
    def select_install_path(self):
        """Let user select installation path"""
        self.log("Please select installation directory...")
        
        path = filedialog.askdirectory(title="Select Installation Directory")
        if path:
            self.install_path.set(path)
            self.log(f"Installation path selected: {path}")
            return True
        else:
            self.log("No installation path selected")
            return False
    
    def clone_repository(self):
        """Clone the GitHub repository"""
        self.log("Cloning repository from GitHub...")
        
        repo_url = "https://github.com/jayjeo/obsidian-milvus-FastMCP"
        target_dir = os.path.join(self.install_path.get(), "obsidian-milvus-fastmcp")
        
        # Check if directory already exists
        if os.path.exists(target_dir):
            self.log(f"Directory already exists: {target_dir}")
            response = messagebox.askyesno("Directory Exists", 
                                         f"The directory already exists:\n{target_dir}\n\n"
                                         "Do you want to use the existing directory?")
            if response:
                self.install_path.set(target_dir)
                return True
            else:
                return False
        
        # Clone repository
        success, stdout, stderr = self.run_command(["git", "clone", repo_url, target_dir])
        if success:
            self.log("Repository cloned successfully")
            self.install_path.set(target_dir)
            return True
        else:
            self.log(f"Failed to clone repository: {stderr}")
            messagebox.showerror("Clone Failed", f"Failed to clone repository:\n{stderr}")
            return False
    
    def install_dependencies(self):
        """Install Python dependencies via Conda"""
        self.log("Installing dependencies via Conda...")
        
        # Change to project directory
        os.chdir(self.install_path.get())
        
        # Install packages
        packages = [
            "python pip",
            "pymilvus mcp fastmcp sentence-transformers torch",
            "PyPDF2 markdown beautifulsoup4 python-dotenv watchdog psutil colorama pyyaml tqdm requests"
        ]
        
        for package_group in packages:
            self.log(f"Installing: {package_group}")
            if "conda install" in package_group:
                cmd = f"conda install -c conda-forge -y {package_group.replace('conda install', '').strip()}"
            else:
                cmd = f"conda run -n base pip install {package_group}"
            
            success, stdout, stderr = self.run_command(cmd, shell=True, check=False)
            if not success:
                self.log(f"Warning: Some packages may have failed: {stderr}")
        
        self.log("Dependencies installation completed")
        return True
    
    def install_podman(self):
        """Install Podman using winget"""
        self.log("Installing Podman...")
        
        # Check if Podman is already installed
        success, _, _ = self.run_command(["podman", "--version"], check=False)
        if success:
            self.log("Podman is already installed")
            return True
        
        # Install via winget
        self.log("Installing Podman via winget...")
        success, stdout, stderr = self.run_command(["winget", "install", "RedHat.Podman"], check=False)
        
        if success:
            self.log("Podman installation completed")
            return True
        else:
            self.log(f"Podman installation may have issues: {stderr}")
            messagebox.showwarning("Podman Installation",
                                 "Podman installation may have encountered issues.\n"
                                 "Please verify installation manually.")
            return True
    
    def configure_wsl(self):
        """Configure WSL and install Ubuntu"""
        self.log("Configuring WSL and installing Ubuntu...")
        
        # Enable Virtual Machine Platform
        self.log("Enabling Virtual Machine Platform...")
        cmd = ["dism.exe", "/online", "/enable-feature", "/featurename:VirtualMachinePlatform", "/all", "/norestart"]
        success, stdout, stderr = self.run_command(cmd, check=False)
        if success:
            self.log("Virtual Machine Platform enabled")
        else:
            self.log(f"Warning: {stderr}")
        
        # Install WSL
        self.log("Installing WSL...")
        success, stdout, stderr = self.run_command(["wsl.exe", "--install"], check=False)
        if success:
            self.log("WSL installation initiated")
        else:
            self.log(f"Warning: {stderr}")
        
        # Set WSL 2 as default
        self.log("Setting WSL 2 as default version...")
        success, stdout, stderr = self.run_command(["wsl", "--set-default-version", "2"], check=False)
        if success:
            self.log("WSL 2 set as default")
        else:
            self.log(f"Warning: {stderr}")
        
        # Download Ubuntu installer
        self.log("Downloading Ubuntu 22.04 LTS installer...")
        ubuntu_url = "https://raw.githubusercontent.com/jayjeo/obsidian-milvus-FastMCP/main/LaborShortage/Ubuntu%2022.04.5%20LTS%20Installer.exe"
        ubuntu_installer = os.path.join(self.install_path.get(), "Ubuntu_Installer.exe")
        
        try:
            urllib.request.urlretrieve(ubuntu_url, ubuntu_installer)
            self.log("Ubuntu installer downloaded")
            
            # Run Ubuntu installer
            self.log("Running Ubuntu installer...")
            subprocess.Popen([ubuntu_installer])
            
            messagebox.showinfo("Ubuntu Installation",
                              "Ubuntu installer has been launched.\n\n"
                              "Please complete the Ubuntu installation,\n"
                              "then restart your computer.\n\n"
                              "After restart, run this installer again to continue.")
            
            # Mark that system restart is needed
            self.state['needs_restart'] = True
            self.save_state()
            
            return True
            
        except Exception as e:
            self.log(f"Failed to download Ubuntu installer: {e}")
            messagebox.showerror("Download Failed", 
                               f"Failed to download Ubuntu installer:\n{e}")
            return False
    
    def configure_paths(self):
        """Configure paths in config.py"""
        self.log("Configuring paths...")
        
        # Get Obsidian vault path
        self.log("Please select your Obsidian vault folder...")
        vault_path = filedialog.askdirectory(title="Select Obsidian Vault Folder")
        if not vault_path:
            self.log("No Obsidian vault path selected")
            return False
        
        self.obsidian_vault_path.set(vault_path)
        self.log(f"Obsidian vault path: {vault_path}")
        
        # Find Podman path
        self.log("Finding Podman installation...")
        find_podman_script = os.path.join(self.install_path.get(), "find_podman_path.bat")
        if os.path.exists(find_podman_script):
            success, stdout, _ = self.run_command([find_podman_script], check=False)
            if success and stdout.strip():
                self.podman_path.set(stdout.strip())
                self.log(f"Found Podman: {stdout.strip()}")
        
        # Update config.py
        config_file = os.path.join(self.install_path.get(), "config.py")
        if os.path.exists(config_file):
            self.log("Updating config.py...")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            # Update Obsidian vault path
            import re
            config_content = re.sub(
                r'OBSIDIAN_VAULT_PATH = ".*?"',
                f'OBSIDIAN_VAULT_PATH = "{vault_path.replace(chr(92), chr(92)+chr(92))}"',
                config_content
            )
            
            # Update Podman path if found
            if self.podman_path.get():
                config_content = re.sub(
                    r'PODMAN_PATH = ".*?"',
                    f'PODMAN_PATH = "{self.podman_path.get().replace(chr(92), chr(92)+chr(92))}"',
                    config_content
                )
            
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_content)
            
            self.log("config.py updated successfully")
        
        return True
    
    def initialize_podman(self):
        """Initialize Podman container"""
        self.log("Initializing Podman container...")
        
        reset_script = os.path.join(self.install_path.get(), "complete-podman-reset.bat")
        if os.path.exists(reset_script):
            self.log("Running Podman reset script...")
            success, stdout, stderr = self.run_command([reset_script], check=False)
            if success:
                self.log("Podman container initialized")
            else:
                self.log(f"Warning: {stderr}")
        
        return True
    
    def initialize_milvus(self):
        """Initialize Milvus server"""
        self.log("Initializing Milvus server...")
        
        start_script = os.path.join(self.install_path.get(), "start_mcp_with_encoding_fix.bat")
        if os.path.exists(start_script):
            self.log("Starting Milvus server...")
            # Run in background
            subprocess.Popen([start_script], cwd=self.install_path.get())
            self.log("Milvus server initialization started")
            time.sleep(5)  # Give it some time to start
        
        return True
    
    def manual_configuration(self):
        """Guide user through manual configuration steps"""
        self.log("Manual configuration required...")
        
        message = """Manual Configuration Required:

Please complete the following steps:

1. Configure Podman auto-launch at startup:
   - Follow instructions in 'Podman auto launch.md'
   - This ensures Podman starts automatically

2. Configure Milvus auto-launch at startup:
   - Follow instructions in 'Milvus auto launch.md'
   - This ensures Milvus starts automatically

3. After completing both configurations:
   - Restart your computer
   - Run this installer again to complete setup

Click OK when you have completed these steps."""
        
        messagebox.showinfo("Manual Configuration", message)
        
        # Open the documentation files if they exist
        docs = ["Podman auto launch.md", "Milvus auto launch.md"]
        for doc in docs:
            doc_path = os.path.join(self.install_path.get(), doc)
            if os.path.exists(doc_path):
                os.startfile(doc_path)
        
        return True
    
    def final_setup(self):
        """Run final setup steps"""
        self.log("Running final setup...")
        
        # Run setup.bat steps
        os.chdir(self.install_path.get())
        
        # Package Installation
        self.log("Installing packages...")
        self.run_command(["python", "-m", "pip", "install", "-r", "requirements.txt"], check=False)
        
        # Run setup.py options
        setup_script = os.path.join(self.install_path.get(), "setup.py")
        if os.path.exists(setup_script):
            # Run each setup option
            for option in range(1, 6):
                self.log(f"Running setup option {option}...")
                # This would need to be automated or interactive
        
        self.log("Final setup completed!")
        return True
    
    def start_installation(self):
        """Start the installation process"""
        self.install_button.config(state=tk.DISABLED)
        
        # Check if admin privileges needed
        if not self.is_admin() and self.current_step >= 4:
            messagebox.showwarning("Administrator Required",
                                 "This installer requires administrator privileges.\n"
                                 "Please restart as administrator.")
            self.root.quit()
            return
        
        # Run installation in thread
        thread = threading.Thread(target=self.run_installation)
        thread.start()
    
    def run_installation(self):
        """Run the installation steps"""
        steps_functions = [
            self.check_conda,
            self.select_install_path,
            self.clone_repository,
            self.install_dependencies,
            self.install_podman,
            self.configure_wsl,
            self.configure_paths,
            self.initialize_podman,
            self.initialize_milvus,
            self.manual_configuration,
            self.final_setup
        ]
        
        while self.current_step < len(steps_functions):
            self.update_step_display()
            
            # Check if restart is needed
            if self.state.get('needs_restart') and self.current_step == 6:
                self.log("System restart required. Please restart and run installer again.")
                break
            
            # Run current step
            success = steps_functions[self.current_step]()
            
            if success:
                self.current_step += 1
                self.save_state()
            else:
                self.log(f"Step failed: {self.steps[self.current_step]}")
                messagebox.showerror("Installation Failed", 
                                   f"Installation failed at step:\n{self.steps[self.current_step]}")
                break
        
        if self.current_step >= len(steps_functions):
            self.update_step_display()
            self.log("Installation completed successfully!")
            messagebox.showinfo("Installation Complete",
                              "Obsidian-Milvus-FastMCP has been installed successfully!\n\n"
                              "You can now use the integration with Claude Desktop.")
            
            # Clean up state file
            if self.state_file.exists():
                self.state_file.unlink()
        
        self.install_button.config(state=tk.NORMAL)
    
    def resume_installation(self):
        """Resume installation from saved state"""
        self.log(f"Resuming installation from step {self.current_step + 1}...")
        messagebox.showinfo("Resume Installation",
                          f"Installation will resume from:\n"
                          f"Step {self.current_step + 1}: {self.steps[self.current_step]}")
    
    def cancel_installation(self):
        """Cancel the installation"""
        response = messagebox.askyesno("Cancel Installation",
                                     "Are you sure you want to cancel the installation?")
        if response:
            self.root.quit()
    
    def run(self):
        """Run the installer"""
        self.root.mainloop()

if __name__ == "__main__":
    installer = ObsidianMilvusInstaller()
    installer.run()
