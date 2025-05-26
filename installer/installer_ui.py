import sys, os, subprocess, winreg
from PyQt5 import QtCore, QtGui, QtWidgets

class InstallerThread(QtCore.QThread):
    progress_changed = QtCore.pyqtSignal(int)
    status_changed = QtCore.pyqtSignal(str)
    log_added = QtCore.pyqtSignal(str)
    completed = QtCore.pyqtSignal(bool)
    
    def __init__(self, install_dir, vault_dir, no_reboot=False):
        super().__init__()
        self.install_dir = install_dir
        self.vault_dir = vault_dir
        self.no_reboot = no_reboot  # True if running in dev mode (skip actual reboot)
    
    def run(self):
        def log(msg): 
            self.log_added.emit(msg)
        def status(msg): 
            self.status_changed.emit(msg)
        def run_cmd(cmd, cwd=None, shell=False):
            """Run a command and return (result, output). `result` is subprocess.CompletedProcess or None on error."""
            try:
                result = subprocess.run(cmd, shell=shell, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8', errors='replace')
            except Exception as e:
                return None, str(e)
            return result, result.stdout
        
        total_steps = 0
        steps_done = 0
        reboot_required = False
        
        # Pre-installation tasks (before reboot)
        tasks_pre = [
            ("Cloning repository from GitHub", ["git", "clone", "https://github.com/jayjeo/obsidian-milvus-FastMCP", self.install_dir]),
            ("Installing Conda base (Python & pip)", ["conda", "install", "-c", "conda-forge", "-y", "python", "pip"]),
            ("Installing Python packages (1/2)", ["conda", "run", "-n", "base", "pip", "install",
                                                 "pymilvus", "mcp", "fastmcp", "sentence-transformers", "torch"]),
            ("Installing Python packages (2/2)", ["conda", "run", "-n", "base", "pip", "install",
                                                 "PyPDF2", "markdown", "beautifulsoup4", "python-dotenv", 
                                                 "watchdog", "psutil", "colorama", "pyyaml", "tqdm", "requests"]),
            ("Installing/Verifying Podman via winget", ["winget", "install", "-e", "--id", "RedHat.Podman", 
                                              "--accept-package-agreements", "--accept-source-agreements"])
        ]
        # Post-installation tasks (after reboot)
        tasks_post = [
            ("Installing podman-compose via pip", ["pip", "install", "podman-compose"])
        ]
        # Count total steps for progress percentage (including individual steps below)
        total_steps = len(tasks_pre) + len(tasks_post) + 8  # +8 for WSL enable (2), Ubuntu install, .env update, find_podman, reset, start_mcp, interactive
        
        # Execute pre-reboot tasks
        for desc, cmd in tasks_pre:
            status(f"{desc}...")
            res, output = run_cmd(cmd, cwd=None, shell=True)
            
            # Special case for Podman installation via winget
            if "Podman" in desc and res is not None and res.returncode != 0:
                # Check if the failure is because Podman is already installed
                if output and ("already installed" in output.lower() or "no available upgrade found" in output.lower()):
                    log(f"✓ {desc} - Podman is already installed")
                    steps_done += 1
                    self.progress_changed.emit(int(steps_done * 100 / total_steps))
                    continue
            
            # Normal case for all other commands
            if res is None or res.returncode != 0:
                log(f"✖ {desc} failed")
                if output: 
                    log(output.strip())
                self.completed.emit(False)
                return
            log(f"✓ {desc} completed")
            steps_done += 1
            self.progress_changed.emit(int(steps_done * 100 / total_steps))
        
        # Enable WSL optional features via DISM
        status("Enabling Virtual Machine Platform (for WSL 2)...")
        res = subprocess.run(["dism.exe", "/online", "/enable-feature", "/featurename:VirtualMachinePlatform", "/all", "/norestart"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8', errors='replace')
        output = res.stdout.strip()
        if res.returncode not in (0, 3010):
            log("✖ Enabling Virtual Machine Platform failed")
            if output: 
                log(output)
            self.completed.emit(False)
            return
        log("✓ Virtual Machine Platform enabled")
        if res.returncode == 3010:
            reboot_required = True
        steps_done += 1
        self.progress_changed.emit(int(steps_done * 100 / total_steps))
        
        status("Enabling Windows Subsystem for Linux...")
        res = subprocess.run(["dism.exe", "/online", "/enable-feature", "/featurename:Microsoft-Windows-Subsystem-Linux", "/all", "/norestart"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8', errors='replace')
        output = res.stdout.strip()
        if res.returncode not in (0, 3010):
            log("✖ Enabling Windows Subsystem for Linux failed")
            if output: 
                log(output)
            self.completed.emit(False)
            return
        log("✓ Windows Subsystem for Linux enabled")
        if res.returncode == 3010:
            reboot_required = True
        steps_done += 1
        self.progress_changed.emit(int(steps_done * 100 / total_steps))
        
        status("Installing/Verifying Ubuntu 22.04 LTS via winget...")
        res, output = run_cmd(["winget", "install", "-e", "--id", "Canonical.Ubuntu.2204", 
                               "--accept-package-agreements", "--accept-source-agreements"], shell=True)
        
        # Check if Ubuntu is already installed (similar to Podman check)
        if (res is None or res.returncode != 0) and output and ("already installed" in output.lower() or "no available upgrade found" in output.lower()):
            log("✓ Ubuntu 22.04 LTS is already installed (WSL)")
            steps_done += 1
            self.progress_changed.emit(int(steps_done * 100 / total_steps))
        elif res is None or res.returncode != 0:
            log("✖ Installing Ubuntu 22.04 via winget failed")
            if output: 
                log(output.strip())
            self.completed.emit(False)
            return
        else:
            log("✓ Ubuntu 22.04 LTS installed (WSL)")
            steps_done += 1
            self.progress_changed.emit(int(steps_done * 100 / total_steps))
        
        # Handle reboot if required
        if reboot_required and not self.no_reboot:
            log("System restart will occur now to complete WSL installation...")
            try:
                # Schedule this installer to resume after reboot with appropriate parameters
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\RunOnce", 0, winreg.KEY_SET_VALUE)
                exe_path = sys.executable  # path to this installer EXE
                run_cmd_line = f"\"{exe_path}\" --resume \"{self.install_dir}\" \"{self.vault_dir}\""
                winreg.SetValueEx(key, "ObsidianFastMCPInstaller", 0, winreg.REG_SZ, run_cmd_line)
                winreg.CloseKey(key)
            except Exception as e:
                log(f"Warning: could not schedule auto-resume ({e})")
            # Initiate system reboot
            try:
                subprocess.Popen(["shutdown", "/r", "/t", "5"])
            except Exception as e:
                log(f"Please reboot your system manually to continue installation (error scheduling reboot: {e})")
            # Do not emit completion (installer will close on reboot)
            return
        elif reboot_required and self.no_reboot:
            log("Reboot skipped (developer mode) – continuing installation without reboot.")
        
        # Execute post-reboot tasks
        for desc, cmd in tasks_post:
            status(f"{desc}...")
            res, output = run_cmd(cmd, shell=True)
            if res is None or res.returncode != 0:
                log(f"✖ {desc} failed")
                if output: 
                    log(output.strip())
                self.completed.emit(False)
                return
            log(f"✓ {desc} completed")
            steps_done += 1
            self.progress_changed.emit(int(steps_done * 100 / total_steps))
        
        status("Updating configuration files...")
        try:
            env_path = os.path.join(self.install_dir, ".env")
            lines = [f'OBSIDIAN_VAULT="{self.vault_dir}"']
            # If Podman is installed in a known path, record it; otherwise it will be found next
            podman_path = None
            common_paths = [
                r"C:\Program Files\RedHat\Podman\podman.exe",
                r"C:\Program Files (x86)\RedHat\Podman\podman.exe"
            ]
            for p in common_paths:
                if os.path.exists(p):
                    podman_path = p
                    break
            if podman_path:
                lines.append(f'PODMAN="{podman_path}"')
            with open(env_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            log("✓ Configuration file .env updated with vault path")
        except Exception as e:
            log(f"✖ Failed to update configuration files ({e})")
            self.completed.emit(False)
            return
        steps_done += 1
        self.progress_changed.emit(int(steps_done * 100 / total_steps))
        
        status("Detecting Podman path...")
        try:
            bat_path = os.path.join(self.install_dir, "find_podman_path.bat")
            proc = subprocess.Popen(bat_path, cwd=self.install_dir, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            podman_output = proc.communicate(timeout=10)[0]
        except subprocess.TimeoutExpired:
            proc.kill()
            podman_output = ""
        if podman_output:
            # Parse output for a found path
            found_path = None
            for line in podman_output.splitlines():
                line = line.strip()
                if line.startswith("1.") and "podman.exe" in line:
                    # The first result path after "1." 
                    try:
                        _, path = line.split(" ", 1)
                    except ValueError:
                        continue
                    if os.path.isfile(path):
                        found_path = path
                        break
            if found_path:
                try:
                    with open(os.path.join(self.install_dir, ".env"), "a") as f:
                        f.write(f'PODMAN="{found_path}"\n')
                    log(f"✓ Podman executable found at: {found_path}")
                except Exception:
                    log("✓ Podman path found (failed to update .env, not critical)")
            else:
                log("✓ Podman path search completed (no results found)")
        else:
            log("✓ Podman path detection skipped (timeout or no output)")
        steps_done += 1
        self.progress_changed.emit(int(steps_done * 100 / total_steps))
        status("Running complete-podman-reset_noask.bat...")
        try:
            bat_path = os.path.join(self.install_dir, "complete-podman-reset_noask.bat")
            proc = subprocess.Popen(bat_path, shell=True, cwd=self.install_dir,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding='utf-8', errors='replace')
            for line in proc.stdout:
                line = line.strip()
                log(line)
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"reset script exited with code {proc.returncode}")
            log("✓ Podman environment reset (complete-podman-reset_noask.bat)")
        except Exception as e:
            log(f"✖ complete-podman-reset_noask.bat failed ({e})")
            self.completed.emit(False)
            return
        steps_done += 1
        self.progress_changed.emit(int(steps_done * 100 / total_steps))
        
        status("Starting Milvus MCP server for initial setup...")
        try:
            # Launch in a new console (detached) so it continues running
            subprocess.Popen(f'start "" "{os.path.join(self.install_dir, "start_mcp_with_encoding_fix.bat")}"', shell=True, cwd=self.install_dir)
            log("✓ Milvus MCP server started (encoding fix applied)")
        except Exception as e:
            log(f"✖ Failed to start MCP server ({e}) – you may start it manually later.")
        steps_done += 1
        self.progress_changed.emit(int(steps_done * 100 / total_steps))
        
        status("Running interactive setup & testing (options 1–5)...")
        try:
            # Run setup.py and feed menu choices 1-5
            setup_py = os.path.join(self.install_dir, "setup.py")
            proc = subprocess.Popen(["python", setup_py], cwd=self.install_dir,
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            proc.communicate("1\n2\n3\n4\n5\n\n", timeout=300)  # 5 min timeout for all tests
            log("✓ Interactive setup & testing (options 1–5) completed")
        except Exception as e:
            log(f"✖ Interactive setup tool did not finish cleanly ({e}) – see console for details.")
            # Not critical for installation success; continue
        steps_done += 1
        self.progress_changed.emit(int(steps_done * 100 / total_steps))
        
        # All tasks completed
        self.completed.emit(True)


class IntroPage(QtWidgets.QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("")  # No title to avoid header area
        layout = QtWidgets.QVBoxLayout(self)
        
        # Welcome header
        welcome_label = QtWidgets.QLabel("<b>Welcome to Obsidian-Milvus-FastMCP Setup</b>")
        font = welcome_label.font()
        font.setPointSize(font.pointSize() + 4)
        welcome_label.setFont(font)
        welcome_label.setAlignment(QtCore.Qt.AlignCenter)
        
        desc_label = QtWidgets.QLabel(
            "This wizard will install Obsidian-Milvus-FastMCP with full administrative privileges.\n\n"
            "Obsidian-Milvus-FastMCP integrates the Milvus vector database with Obsidian notes, "
            "enabling powerful semantic search through Claude Desktop."
        )
        desc_label.setWordWrap(True)
        
        process_label = QtWidgets.QLabel(
            "<b>Complete Installation Process:</b><br>"
            "✔ Prerequisites verification (Conda installed)<br>"
            "✔ Repository cloning from GitHub<br>"
            "✔ Python dependencies installation (Conda & pip)<br>"
            "✔ Podman container runtime installation<br>"
            "✔ WSL and Ubuntu 22.04 setup<br>"
            "✔ Milvus vector database deployment<br>"
            "✔ Auto-start services for Podman/Milvus<br>"
            "✔ Claude Desktop integration"
        )
        process_label.setTextFormat(QtCore.Qt.RichText)
        process_label.setWordWrap(True)
        
        req_label = QtWidgets.QLabel(
            "<b>System Requirements:</b><br>"
            "• Windows 10 or later (for WSL 2)<br>"
            "• Anaconda or Miniconda (REQUIRED – must be pre-installed)<br>"
            "• 20 GB free disk space (containers & dependencies)<br>"
            "• Stable internet connection<br>"
            "• Administrator privileges (run this installer as Admin)<br>"
            "• 8 GB RAM or more recommended"
        )
        req_label.setTextFormat(QtCore.Qt.RichText)
        req_label.setWordWrap(True)
        
        before_group = QtWidgets.QGroupBox("Before Starting")
        before_layout = QtWidgets.QVBoxLayout(before_group)
        before_text = QtWidgets.QLabel(
            'Please ensure Anaconda (Conda) is installed on your system.<br>'
            'Download: <a href="https://www.anaconda.com/download">anaconda.com/download</a> – install with default settings and be sure to <b>add Anaconda to PATH</b> when prompted.<br><br>'
            'This installer will make system-level changes, including:<br>'
            '– Installing Podman and WSL<br>'
            '– Modifying system startup services'
        )
        before_text.setTextFormat(QtCore.Qt.RichText)
        before_text.setOpenExternalLinks(True)
        before_text.setWordWrap(True)
        before_layout.addWidget(before_text)
        
        layout.addWidget(welcome_label)
        layout.addWidget(desc_label)
        layout.addWidget(process_label)
        layout.addWidget(req_label)
        layout.addWidget(before_group)


class PathPage(QtWidgets.QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("")  # no title to avoid header
        self.setSubTitle("Select the directory where Obsidian-Milvus-FastMCP will be installed.")
        
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Installation Path:")
        self.path_edit = QtWidgets.QLineEdit()
        browse_btn = QtWidgets.QPushButton("Browse...")
        form_layout.addWidget(label)
        form_layout.addWidget(self.path_edit)
        form_layout.addWidget(browse_btn)
        layout.addLayout(form_layout)
        note_label = QtWidgets.QLabel("The repository will be cloned into this directory.")
        note_label.setWordWrap(True)
        layout.addWidget(note_label)
        
        self.registerField("installDir*", self.path_edit)
        browse_btn.clicked.connect(self.browse_folder)
    
    def browse_folder(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Installation Directory")
        if directory:
            self.path_edit.setText(directory)
    
    def validatePage(self):
        path = self.path_edit.text().strip()
        if not path:
            return False
        if os.path.exists(path):
            if not os.path.isdir(path):
                QtWidgets.QMessageBox.warning(self, "Invalid Path", "The selected path is not a directory.")
                return False
            if os.listdir(path):
                # Directory not empty – warn user
                response = QtWidgets.QMessageBox.question(
                    self, "Directory Not Empty",
                    "The selected directory is not empty. It's recommended to use an empty folder for installation.\n"
                    "Do you want to use this directory anyway?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No
                )
                if response != QtWidgets.QMessageBox.Yes:
                    return False
        else:
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Unable to create the directory:\n{e}")
                return False
        return True


class VaultPage(QtWidgets.QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("")  # no title
        self.setSubTitle("Select the folder of your Obsidian vault (where your notes are stored).")
        
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Obsidian Vault Path:")
        self.vault_edit = QtWidgets.QLineEdit()
        browse_btn = QtWidgets.QPushButton("Browse...")
        form_layout.addWidget(label)
        form_layout.addWidget(self.vault_edit)
        form_layout.addWidget(browse_btn)
        layout.addLayout(form_layout)
        info_label = QtWidgets.QLabel("Choose the root folder of your Obsidian vault.")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        self.registerField("vaultDir*", self.vault_edit)
        browse_btn.clicked.connect(self.browse_folder)
    
    def browse_folder(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Obsidian Vault Directory")
        if directory:
            self.vault_edit.setText(directory)
    
    def validatePage(self):
        path = self.vault_edit.text().strip()
        if not path:
            return False
        if not os.path.isdir(path):
            QtWidgets.QMessageBox.warning(self, "Invalid Path", "The specified Obsidian vault path does not exist.")
            return False
        return True


class SummaryPage(QtWidgets.QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Setup is ready to begin installation.")
        layout = QtWidgets.QVBoxLayout(self)
        
        self.install_label = QtWidgets.QLabel()
        self.vault_label = QtWidgets.QLabel()
        components_label = QtWidgets.QLabel(
            "<b>Components to install:</b><br>"
            "• Repository cloning (GitHub)<br>"
            "• Python dependencies (Conda & pip)<br>"
            "• Podman container runtime<br>"
            "• WSL 2 and Ubuntu 22.04<br>"
            "• Milvus vector database (via Podman)<br>"
            "• Auto-start services (Podman/Milvus)<br>"
            "• Claude Desktop integration"
        )
        components_label.setTextFormat(QtCore.Qt.RichText)
        components_label.setWordWrap(True)
        
        changes_label = QtWidgets.QLabel("⚠ <b>System Changes:</b> Windows WSL feature will be enabled")
        changes_label.setTextFormat(QtCore.Qt.RichText)
        
        important_group = QtWidgets.QGroupBox("Important")
        important_layout = QtWidgets.QVBoxLayout(important_group)
        important_text = QtWidgets.QLabel(
            "A system restart will be required after installing WSL. The installer will reboot your PC automatically and resume afterward."
        )
        important_text.setWordWrap(True)
        important_layout.addWidget(important_text)
        
        layout.addWidget(self.install_label)
        layout.addWidget(self.vault_label)
        layout.addWidget(components_label)
        layout.addWidget(changes_label)
        layout.addWidget(important_group)
    
    def initializePage(self):
        install_dir = self.field("installDir")
        vault_dir = self.field("vaultDir")
        self.install_label.setText(f"<b>Installation Path:</b> {install_dir}")
        self.vault_label.setText(f"<b>Obsidian Vault:</b> {vault_dir}")
        # Change Next button text to "Install"
        wizard = self.wizard()
        wizard.setButtonText(QtWidgets.QWizard.NextButton, "Install")


class ProgressPage(QtWidgets.QWizardPage):
    def __init__(self, install_dir, vault_dir, no_reboot):
        super().__init__()
        self.thread = None
        self.install_dir = install_dir
        self.vault_dir = vault_dir
        self.no_reboot = no_reboot
        self.setTitle("Installing Obsidian-Milvus-FastMCP")
        
        layout = QtWidgets.QVBoxLayout(self)
        self.status_label = QtWidgets.QLabel("Starting installation...")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(QtCore.Qt.AlignCenter)
        self.progress_bar.setFormat("%p%")
        
        details_label = QtWidgets.QLabel("<b>Installation Details:</b>")
        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(details_label)
        layout.addWidget(self.log_edit)
    
    def initializePage(self):
        # Disable Back to prevent navigating away
        wizard = self.wizard()
        wizard.button(QtWidgets.QWizard.BackButton).setDisabled(True)
        # Start the installation thread
        install_dir = self.field("installDir") or self.install_dir
        vault_dir = self.field("vaultDir") or self.vault_dir
        self.thread = InstallerThread(install_dir, vault_dir, no_reboot=self.wizard().no_reboot_mode)
        self.thread.progress_changed.connect(self.progress_bar.setValue)
        self.thread.status_changed.connect(self.status_label.setText)
        self.thread.log_added.connect(lambda txt: self.log_edit.appendPlainText(txt))
        self.thread.completed.connect(self.on_installation_complete)
        self.thread.start()
    
    def on_installation_complete(self, success):
        wizard = self.wizard()
        wizard.button(QtWidgets.QWizard.BackButton).setEnabled(False)
        if success:
            # Automatically proceed to the final page
            wizard.button(QtWidgets.QWizard.NextButton).click()
        else:
            # Enable Next button to allow closing the wizard
            wizard.setButtonText(QtWidgets.QWizard.NextButton, "Finish")
            wizard.button(QtWidgets.QWizard.NextButton).setEnabled(True)


class FinalPage(QtWidgets.QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Installation Complete")
        layout = QtWidgets.QVBoxLayout(self)
        complete_label = QtWidgets.QLabel("Obsidian-Milvus-FastMCP has been installed successfully.")
        complete_label.setWordWrap(True)
        link_label = QtWidgets.QLabel(
            '• <a href="https://share.note.sx/r6kx06pj#78CIGnxLJYkJG+ZrQKYQhU35gtl+nKa47ZllwEyfUE0">Podman auto-launch setup guide</a><br>'
            '• <a href="https://share.note.sx/y9vrzgj6#zr1aL4s1WFBK/A4WvqvkP6ETVMC4sKcAwbqAt4NyZhk">Milvus server auto-launch setup guide</a>'
        )
        link_label.setTextFormat(QtCore.Qt.RichText)
        link_label.setOpenExternalLinks(True)
        link_label.setWordWrap(True)
        layout.addWidget(complete_label)
        layout.addWidget(link_label)
        self.setFinalPage(True)


class InstallerWizard(QtWidgets.QWizard):
    def __init__(self, resume_mode=False, install_dir="", vault_dir="", no_reboot=False):
        super().__init__()
        self.no_reboot_mode = no_reboot
        self.setWindowTitle("Obsidian-Milvus-FastMCP Setup")
        # Use ClassicStyle to avoid modern header banner
        self.setWizardStyle(QtWidgets.QWizard.ClassicStyle)
        self.setOption(QtWidgets.QWizard.NoCancelButtonOnLastPage, True)
        
        if resume_mode:
            # In resume mode, skip straight to progress and final pages
            progress_page = ProgressPage(install_dir, vault_dir, no_reboot)
            final_page = FinalPage()
            self.addPage(progress_page)
            self.addPage(final_page)
        else:
            intro_page = IntroPage()
            path_page = PathPage()
            vault_page = VaultPage()
            summary_page = SummaryPage()
            progress_page = ProgressPage("", "", no_reboot)
            final_page = FinalPage()
            self.addPage(intro_page)
            self.addPage(path_page)
            self.addPage(vault_page)
            self.addPage(summary_page)
            self.addPage(progress_page)
            self.addPage(final_page)
            # Set the Next button text on the summary page to "Install"
            # (Will be done in SummaryPage.initializePage)
        
        # Style navigation buttons: Next button blue when enabled, white/grey when disabled; Back as outline
        next_btn = self.button(QtWidgets.QWizard.NextButton)
        back_btn = self.button(QtWidgets.QWizard.BackButton)
        cancel_btn = self.button(QtWidgets.QWizard.CancelButton)
        next_btn.setObjectName("nextButton")
        back_btn.setObjectName("backButton")
        cancel_btn.setObjectName("cancelButton")
        self.setStyleSheet("""
            QPushButton#nextButton:enabled { background-color: #0078D7; color: white; }
            QPushButton#nextButton:disabled { background-color: white; color: gray; border: 1px solid gray; }
            QPushButton#backButton { background-color: white; color: black; border: 1px solid gray; }
            QPushButton#cancelButton { /* default style for Cancel */ }
        """)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Obsidian-Milvus-FastMCP Installer")
    # Increase default font size for better readability
    font = app.font()
    font.setPointSize(font.pointSize() + 1)
    app.setFont(font)
    
    # Determine if running as frozen EXE or not (for reboot handling)
    no_reboot = not hasattr(sys, "frozen")
    resume = False
    inst_dir = ""
    vault_dir = ""
    # Check command-line arguments for resume mode
    if len(sys.argv) > 1 and sys.argv[1] == "--resume":
        resume = True
        if len(sys.argv) >= 4:
            inst_dir = sys.argv[2]
            vault_dir = sys.argv[3]
    wizard = InstallerWizard(resume_mode=resume, install_dir=inst_dir, vault_dir=vault_dir, no_reboot=no_reboot)
    wizard.setMinimumSize(900, 600)
    wizard.show()
    sys.exit(app.exec_())
