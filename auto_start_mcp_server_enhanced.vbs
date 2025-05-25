' Auto Start MCP Server - Enhanced Python Environment Detection
' This script handles Python environment issues and ensures proper module loading
' Enhanced with Python environment validation and fallback strategies

Dim objShell, objFSO, scriptDir, logFile, startTime, tempPythonFile
Dim debugLogFile, errorCount, warningCount

' Initialize objects
Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

' Get script directory (relative path approach)
scriptDir = objFSO.GetParentFolderName(WScript.ScriptFullName)
logFile = scriptDir & "\vbs_startup.log"
debugLogFile = scriptDir & "\vbs_debug.log"
tempPythonFile = scriptDir & "\temp_mcp_option1_only.py"

' Initialize counters
errorCount = 0
warningCount = 0

' Get current timestamp
startTime = Now()

' Enhanced logging functions with different log levels
Sub WriteLog(message)
    WriteLogWithLevel "INFO", message, logFile
End Sub

Sub WriteDebugLog(message)
    WriteLogWithLevel "DEBUG", message, debugLogFile
End Sub

Sub WriteWarningLog(message)
    warningCount = warningCount + 1
    WriteLogWithLevel "WARNING", message, logFile
    WriteLogWithLevel "WARNING", message, debugLogFile
End Sub

Sub WriteErrorLog(message)
    errorCount = errorCount + 1
    WriteLogWithLevel "ERROR", message, logFile
    WriteLogWithLevel "ERROR", message, debugLogFile
End Sub

Sub WriteLogWithLevel(level, message, targetLogFile)
    Dim logText, timestamp
    timestamp = FormatDateTime(Now(), 0) & "." & Right("000" & Timer * 1000 Mod 1000, 3)
    logText = timestamp & " [" & level & "] " & message & vbCrLf
    
    On Error Resume Next
    Dim file
    Set file = objFSO.OpenTextFile(targetLogFile, 8, True) ' 8 = ForAppending, True = Create if not exists
    If Err.Number = 0 Then
        file.Write logText
        file.Close
    End If
    On Error GoTo 0
End Sub

' Enhanced Python environment detection
Function FindBestPythonEnvironment()
    WriteDebugLog "Starting comprehensive Python environment detection..."
    
    ' Array of possible Python commands and their priorities
    Dim pythonCandidates(20), candidateCount, i
    candidateCount = 0
    
    ' Add common Python commands
    pythonCandidates(candidateCount) = "python"
    candidateCount = candidateCount + 1
    pythonCandidates(candidateCount) = "python3"
    candidateCount = candidateCount + 1
    pythonCandidates(candidateCount) = "py"
    candidateCount = candidateCount + 1
    pythonCandidates(candidateCount) = "python.exe"
    candidateCount = candidateCount + 1
    pythonCandidates(candidateCount) = "python3.exe"
    candidateCount = candidateCount + 1
    
    ' Add common installation paths
    pythonCandidates(candidateCount) = "C:\Python39\python.exe"
    candidateCount = candidateCount + 1
    pythonCandidates(candidateCount) = "C:\Python310\python.exe"
    candidateCount = candidateCount + 1
    pythonCandidates(candidateCount) = "C:\Python311\python.exe"
    candidateCount = candidateCount + 1
    pythonCandidates(candidateCount) = "C:\Python312\python.exe"
    candidateCount = candidateCount + 1
    
    ' Add AppData paths
    Dim userProfile
    userProfile = objShell.ExpandEnvironmentStrings("%USERPROFILE%")
    pythonCandidates(candidateCount) = userProfile & "\AppData\Local\Programs\Python\Python39\python.exe"
    candidateCount = candidateCount + 1
    pythonCandidates(candidateCount) = userProfile & "\AppData\Local\Programs\Python\Python310\python.exe"
    candidateCount = candidateCount + 1
    pythonCandidates(candidateCount) = userProfile & "\AppData\Local\Programs\Python\Python311\python.exe"
    candidateCount = candidateCount + 1
    pythonCandidates(candidateCount) = userProfile & "\AppData\Local\Programs\Python\Python312\python.exe"
    candidateCount = candidateCount + 1
    
    ' Add Anaconda/Miniconda paths
    pythonCandidates(candidateCount) = userProfile & "\anaconda3\python.exe"
    candidateCount = candidateCount + 1
    pythonCandidates(candidateCount) = userProfile & "\miniconda3\python.exe"
    candidateCount = candidateCount + 1
    pythonCandidates(candidateCount) = "C:\Anaconda3\python.exe"
    candidateCount = candidateCount + 1
    pythonCandidates(candidateCount) = "C:\Miniconda3\python.exe"
    candidateCount = candidateCount + 1
    
    ' Test each candidate
    For i = 0 To candidateCount - 1
        Dim candidate, versionResult, moduleResult
        candidate = pythonCandidates(i)
        
        WriteDebugLog "Testing Python candidate: " & candidate
        
        On Error Resume Next
        
        ' Test if python executable exists and works
        versionResult = objShell.Run(candidate & " --version", 0, True)
        
        If Err.Number = 0 And versionResult = 0 Then
            WriteDebugLog "Python version check passed for: " & candidate
            
            ' Test if markdown module is available
            moduleResult = objShell.Run(candidate & " -c ""import markdown; print('markdown OK')""", 0, True)
            
            If Err.Number = 0 And moduleResult = 0 Then
                WriteLog "Found working Python with markdown module: " & candidate
                FindBestPythonEnvironment = candidate
                Exit Function
            Else
                WriteWarningLog "Python found but markdown module missing: " & candidate
            End If
        Else
            WriteDebugLog "Python version check failed for: " & candidate & " (Error: " & Err.Number & ")"
        End If
        
        On Error GoTo 0
    Next
    
    ' If no Python with markdown found, return the first working Python
    For i = 0 To candidateCount - 1
        candidate = pythonCandidates(i)
        
        On Error Resume Next
        versionResult = objShell.Run(candidate & " --version", 0, True)
        
        If Err.Number = 0 And versionResult = 0 Then
            WriteWarningLog "Using Python without markdown module verification: " & candidate
            FindBestPythonEnvironment = candidate
            Exit Function
        End If
        
        On Error GoTo 0
    Next
    
    ' No working Python found
    WriteErrorLog "No working Python installation found"
    FindBestPythonEnvironment = ""
End Function

' Function to install missing modules
Function InstallMissingModules(pythonCommand)
    WriteLog "Attempting to install missing modules..."
    
    On Error Resume Next
    
    ' Try to install requirements.txt
    Dim installCommand, installResult
    installCommand = pythonCommand & " -m pip install -r """ & scriptDir & "\requirements.txt"""
    
    WriteLog "Installing command: " & installCommand
    installResult = objShell.Run(installCommand, 1, True) ' Show window for user feedback
    
    If Err.Number = 0 And installResult = 0 Then
        WriteLog "Successfully installed requirements"
        InstallMissingModules = True
    Else
        WriteErrorLog "Failed to install requirements. Error code: " & installResult
        
        ' Try individual module installation
        WriteLog "Trying individual module installation..."
        Dim markdownInstall
        markdownInstall = objShell.Run(pythonCommand & " -m pip install markdown>=3.4.3", 1, True)
        
        If markdownInstall = 0 Then
            WriteLog "Successfully installed markdown module individually"
            InstallMissingModules = True
        Else
            WriteErrorLog "Failed to install markdown module"
            InstallMissingModules = False
        End If
    End If
    
    On Error GoTo 0
End Function

' Function to create environment checking script
Sub CreateEnvironmentChecker()
    WriteDebugLog "Creating Python environment checker..."
    
    On Error Resume Next
    Dim envCheckFile, file
    envCheckFile = scriptDir & "\temp_env_check.py"
    
    Set file = objFSO.CreateTextFile(envCheckFile, True, False)
    If Err.Number <> 0 Then
        WriteErrorLog "Failed to create environment checker file: " & Err.Description
        Exit Sub
    End If
    
    file.WriteLine "#!/usr/bin/env python3"
    file.WriteLine "import sys"
    file.WriteLine "import os"
    file.WriteLine "print('Python executable:', sys.executable)"
    file.WriteLine "print('Python version:', sys.version)"
    file.WriteLine "print('Current working directory:', os.getcwd())"
    file.WriteLine "print('Python path:', sys.path)"
    file.WriteLine "print()"
    file.WriteLine "try:"
    file.WriteLine "    import markdown"
    file.WriteLine "    print('SUCCESS: markdown module is available')"
    file.WriteLine "    print('markdown version:', getattr(markdown, '__version__', 'unknown'))"
    file.WriteLine "except ImportError as e:"
    file.WriteLine "    print('ERROR: markdown module not found -', str(e))"
    file.WriteLine "    sys.exit(1)"
    file.WriteLine "except Exception as e:"
    file.WriteLine "    print('ERROR: unexpected error -', str(e))"
    file.WriteLine "    sys.exit(1)"
    file.WriteLine "print('Environment check completed successfully')"
    
    file.Close
    
    If Err.Number = 0 Then
        WriteLog "Environment checker created successfully"
    Else
        WriteErrorLog "Failed to create environment checker: " & Err.Description
    End If
    
    On Error GoTo 0
End Sub

' Function to run environment check
Function RunEnvironmentCheck(pythonCommand)
    WriteLog "Running environment check..."
    
    On Error Resume Next
    
    Dim envCheckFile, checkResult
    envCheckFile = scriptDir & "\temp_env_check.py"
    
    checkResult = objShell.Run(pythonCommand & " """ & envCheckFile & """", 1, True) ' Show output
    
    If Err.Number = 0 And checkResult = 0 Then
        WriteLog "Environment check passed"
        RunEnvironmentCheck = True
    Else
        WriteWarningLog "Environment check failed with code: " & checkResult
        RunEnvironmentCheck = False
    End If
    
    ' Cleanup temp file
    If objFSO.FileExists(envCheckFile) Then
        objFSO.DeleteFile envCheckFile
    End If
    
    On Error GoTo 0
End Function

' Enhanced main execution with comprehensive Python environment handling
Sub Main()
    WriteLog "=== ENHANCED VBS MCP SERVER STARTUP ==="
    WriteLog "Starting execution at: " & Now()
    WriteLog "Enhanced Python environment detection enabled"
    WriteLog "Script path: " & WScript.ScriptFullName
    WriteLog "Working directory: " & scriptDir
    
    ' Step 1: Find the best Python environment
    WriteLog "=== STEP 1: PYTHON ENVIRONMENT DETECTION ==="
    Dim pythonCommand
    pythonCommand = FindBestPythonEnvironment()
    
    If pythonCommand = "" Then
        WriteErrorLog "No working Python installation found"
        WriteErrorLog "Please install Python and ensure it's in your PATH"
        WriteErrorLog "Or install Python from: https://www.python.org/downloads/"
        Exit Sub
    End If
    
    WriteLog "Selected Python: " & pythonCommand
    
    ' Step 2: Create and run environment checker
    WriteLog "=== STEP 2: ENVIRONMENT VALIDATION ==="
    CreateEnvironmentChecker()
    
    Dim envCheckPassed
    envCheckPassed = RunEnvironmentCheck(pythonCommand)
    
    If Not envCheckPassed Then
        WriteWarningLog "Environment check failed - attempting to install missing modules"
        
        ' Step 3: Try to install missing modules
        WriteLog "=== STEP 3: MODULE INSTALLATION ==="
        Dim installSuccess
        installSuccess = InstallMissingModules(pythonCommand)
        
        If installSuccess Then
            WriteLog "Module installation completed - retrying environment check"
            envCheckPassed = RunEnvironmentCheck(pythonCommand)
        End If
        
        If Not envCheckPassed Then
            WriteErrorLog "Could not resolve Python environment issues"
            WriteErrorLog "Please manually run: " & pythonCommand & " -m pip install -r requirements.txt"
            WriteErrorLog "Or run the check_python_env.py script for detailed diagnostics"
            Exit Sub
        End If
    End If
    
    WriteLog "Python environment validation passed!"
    
    ' Step 4: Create and run the MCP server
    WriteLog "=== STEP 4: MCP SERVER STARTUP ==="
    
    ' Create the temporary Python script (same as original)
    CreatePureOption1PythonScript()
    
    If Not objFSO.FileExists(tempPythonFile) Then
        WriteErrorLog "Failed to create MCP server script"
        Exit Sub
    End If
    
    ' Execute the MCP server
    Dim command
    command = "cmd /c cd /d """ & scriptDir & """ && """ & pythonCommand & """ """ & tempPythonFile & """"
    
    WriteLog "Executing MCP server with validated Python environment"
    WriteLog "Command: " & command
    
    On Error Resume Next
    Dim result
    result = objShell.Run(command, 0, False) ' Hidden execution
    
    If Err.Number = 0 Then
        WriteLog "MCP Server started successfully with enhanced environment detection"
        WriteLog "Python environment: " & pythonCommand
    Else
        WriteErrorLog "Failed to start MCP server: " & Err.Description
    End If
    
    On Error GoTo 0
    
    ' Cleanup
    WScript.Sleep 5000
    CleanupTempFiles()
End Sub

' Function to create Pure Option 1 Python script (same as original)
Sub CreatePureOption1PythonScript()
    WriteDebugLog "Creating Pure Option 1 Python script..."
    
    On Error Resume Next
    Dim file
    Set file = objFSO.CreateTextFile(tempPythonFile, True, False)
    
    If Err.Number <> 0 Then
        WriteErrorLog "Failed to create Python script: " & Err.Description
        Exit Sub
    End If
    
    ' Write the same Python code as original
    file.WriteLine "#!/usr/bin/env python3"
    file.WriteLine "# Pure Option 1 MCP Server - Enhanced Environment"
    file.WriteLine "import os"
    file.WriteLine "import sys"
    file.WriteLine "import threading"
    file.WriteLine "import time"
    file.WriteLine "import logging"
    file.WriteLine "import subprocess"
    file.WriteLine "from pathlib import Path"
    file.WriteLine ""
    file.WriteLine "# Configure logging"
    file.WriteLine "log_file = Path(__file__).parent / 'auto_startup_mcp.log'"
    file.WriteLine "logging.basicConfig("
    file.WriteLine "    level=logging.INFO,"
    file.WriteLine "    format='%(asctime)s - %(levelname)s - %(message)s',"
    file.WriteLine "    handlers=["
    file.WriteLine "        logging.FileHandler(log_file, encoding='utf-8'),"
    file.WriteLine "        logging.StreamHandler(sys.stdout)"
    file.WriteLine "    ]"
    file.WriteLine ")"
    file.WriteLine "logger = logging.getLogger(__name__)"
    file.WriteLine ""
    file.WriteLine "def check_environment():"
    file.WriteLine "    logger.info('Checking Python environment...')"
    file.WriteLine "    try:"
    file.WriteLine "        import markdown"
    file.WriteLine "        logger.info(f'markdown module available: {markdown.__version__}')"
    file.WriteLine "        return True"
    file.WriteLine "    except ImportError as e:"
    file.WriteLine "        logger.error(f'markdown module not available: {e}')"
    file.WriteLine "        return False"
    file.WriteLine ""
    file.WriteLine "def initialize():"
    file.WriteLine "    logger.info('Obsidian-Milvus-MCP system initializing...')"
    file.WriteLine "    if not check_environment():"
    file.WriteLine "        raise RuntimeError('Python environment check failed')"
    file.WriteLine "    try:"
    file.WriteLine "        from milvus_manager import MilvusManager"
    file.WriteLine "        from obsidian_processor import ObsidianProcessor"
    file.WriteLine "        import config"
    file.WriteLine "        logger.info('Connecting to Milvus...')"
    file.WriteLine "        milvus_manager = MilvusManager()"
    file.WriteLine "        logger.info('Initializing Obsidian processor...')"
    file.WriteLine "        processor = ObsidianProcessor(milvus_manager)"
    file.WriteLine "        return processor"
    file.WriteLine "    except Exception as e:"
    file.WriteLine "        logger.error(f'Failed to initialize system: {e}')"
    file.WriteLine "        raise"
    file.WriteLine ""
    file.WriteLine "def start_file_watcher(processor):"
    file.WriteLine "    try:"
    file.WriteLine "        from watcher import start_watcher"
    file.WriteLine "        import config"
    file.WriteLine "        logger.info('Starting file watcher...')"
    file.WriteLine "        watcher_thread = threading.Thread(target=start_watcher, args=(processor,))"
    file.WriteLine "        watcher_thread.daemon = True"
    file.WriteLine "        watcher_thread.start()"
    file.WriteLine "        logger.info('File watcher started successfully')"
    file.WriteLine "        return watcher_thread"
    file.WriteLine "    except Exception as e:"
    file.WriteLine "        logger.error(f'Failed to start file watcher: {e}')"
    file.WriteLine "        return None"
    file.WriteLine ""
    file.WriteLine "def start_mcp_server_blocking():"
    file.WriteLine "    try:"
    file.WriteLine "        import config"
    file.WriteLine "        logger.info('Starting MCP Server for Claude Desktop...')"
    file.WriteLine "        mcp_server_path = os.path.join(os.path.dirname(__file__), 'mcp_server.py')"
    file.WriteLine "        result = subprocess.run([sys.executable, mcp_server_path], cwd=os.path.dirname(__file__))"
    file.WriteLine "        logger.info(f'MCP Server exited with return code: {result.returncode}')"
    file.WriteLine "    except Exception as e:"
    file.WriteLine "        logger.error(f'Failed to start MCP server: {e}')"
    file.WriteLine "        raise"
    file.WriteLine ""
    file.WriteLine "def main():"
    file.WriteLine "    try:"
    file.WriteLine "        logger.info('=' * 70)"
    file.WriteLine "        logger.info('ENHANCED MCP SERVER - AUTO START MODE')"
    file.WriteLine "        logger.info('Enhanced Python environment detection enabled')"
    file.WriteLine "        logger.info('=' * 70)"
    file.WriteLine "        logger.info('Step 1/3: Initializing system...')"
    file.WriteLine "        processor = initialize()"
    file.WriteLine "        logger.info('Step 2/3: Starting file watcher...')"
    file.WriteLine "        watcher_thread = start_file_watcher(processor)"
    file.WriteLine "        logger.info('Step 3/3: Starting MCP Server...')"
    file.WriteLine "        start_mcp_server_blocking()"
    file.WriteLine "    except Exception as e:"
    file.WriteLine "        logger.error(f'Critical error: {e}')"
    file.WriteLine "        import traceback"
    file.WriteLine "        logger.error(f'Stack trace: {traceback.format_exc()}')"
    file.WriteLine "        sys.exit(1)"
    file.WriteLine ""
    file.WriteLine "if __name__ == '__main__':"
    file.WriteLine "    main()"
    
    file.Close
    
    If Err.Number = 0 Then
        WriteLog "Enhanced Python script created successfully"
    Else
        WriteErrorLog "Failed to write Python script: " & Err.Description
    End If
    
    On Error GoTo 0
End Sub

' Function to cleanup temporary files
Sub CleanupTempFiles()
    On Error Resume Next
    
    ' Clean up temporary Python file
    If objFSO.FileExists(tempPythonFile) Then
        objFSO.DeleteFile tempPythonFile
        WriteLog "Cleaned up temporary Python file"
    End If
    
    ' Clean up environment checker
    Dim envCheckFile
    envCheckFile = scriptDir & "\temp_env_check.py"
    If objFSO.FileExists(envCheckFile) Then
        objFSO.DeleteFile envCheckFile
        WriteLog "Cleaned up environment checker file"
    End If
    
    On Error GoTo 0
End Sub

' Execute main function
Main()

' Clean up objects
Set objShell = Nothing
Set objFSO = Nothing
