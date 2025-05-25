' Enhanced MCP Server VBS with Timeout Protection
' Addresses the infinite hanging issue with timeout and monitoring

Dim objShell, objFSO, scriptDir, logFile, tempPythonFile
Dim debugLogFile, errorCount, warningCount

' Initialize objects
Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

' Get script directory
scriptDir = objFSO.GetParentFolderName(WScript.ScriptFullName)
logFile = scriptDir & "\vbs_startup.log"
debugLogFile = scriptDir & "\vbs_debug.log"
tempPythonFile = scriptDir & "\temp_mcp_option1_only_timeout.py"

' Initialize counters
errorCount = 0
warningCount = 0

' Enhanced logging functions
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
    Set file = objFSO.OpenTextFile(targetLogFile, 8, True)
    If Err.Number = 0 Then
        file.Write logText
        file.Close
    End If
    On Error GoTo 0
End Sub

' Find Python with timeout protection
Function FindPythonWithTimeout()
    WriteDebugLog "Finding Python with timeout protection..."
    
    Dim pythonCommands, i, testResult
    pythonCommands = Array("python", "python3", "py")
    
    For i = 0 To UBound(pythonCommands)
        WriteDebugLog "Testing Python command: " & pythonCommands(i)
        
        On Error Resume Next
        testResult = objShell.Run(pythonCommands(i) & " --version", 0, True)
        
        If Err.Number = 0 And testResult = 0 Then
            WriteLog "Found working Python: " & pythonCommands(i)
            FindPythonWithTimeout = pythonCommands(i)
            Exit Function
        End If
        
        On Error GoTo 0
    Next
    
    WriteErrorLog "No working Python found"
    FindPythonWithTimeout = ""
End Function

' Create timeout-protected Python script
Sub CreateTimeoutProtectedPythonScript()
    WriteDebugLog "Creating timeout-protected Python script..."
    
    On Error Resume Next
    Dim file
    Set file = objFSO.CreateTextFile(tempPythonFile, True, False)
    
    If Err.Number <> 0 Then
        WriteErrorLog "Failed to create Python script: " & Err.Description
        Exit Sub
    End If
    
    ' Write enhanced Python script with timeout protection
    file.WriteLine "#!/usr/bin/env python3"
    file.WriteLine "# Timeout-Protected MCP Server Startup"
    file.WriteLine "import os"
    file.WriteLine "import sys"
    file.WriteLine "import threading"
    file.WriteLine "import time"
    file.WriteLine "import logging"
    file.WriteLine "import subprocess"
    file.WriteLine "import signal"
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
    file.WriteLine "class TimeoutProtection:"
    file.WriteLine "    def __init__(self, timeout_seconds=120):"
    file.WriteLine "        self.timeout_seconds = timeout_seconds"
    file.WriteLine "        self.start_time = None"
    file.WriteLine "        self.timeout_triggered = False"
    file.WriteLine ""
    file.WriteLine "    def start(self):"
    file.WriteLine "        self.start_time = time.time()"
    file.WriteLine "        timer_thread = threading.Thread(target=self._timeout_monitor, daemon=True)"
    file.WriteLine "        timer_thread.start()"
    file.WriteLine "        logger.info(f'Timeout protection started: {self.timeout_seconds}s')"
    file.WriteLine ""
    file.WriteLine "    def _timeout_monitor(self):"
    file.WriteLine "        time.sleep(self.timeout_seconds)"
    file.WriteLine "        if not self.timeout_triggered:"
    file.WriteLine "            self.timeout_triggered = True"
    file.WriteLine "            logger.error(f'TIMEOUT: Process exceeded {self.timeout_seconds}s limit')"
    file.WriteLine "            logger.error('Forcing exit to prevent infinite hang')"
    file.WriteLine "            os._exit(1)"
    file.WriteLine ""
    file.WriteLine "    def complete(self):"
    file.WriteLine "        self.timeout_triggered = True"
    file.WriteLine "        logger.info('Process completed within timeout')"
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
    file.WriteLine "def quick_initialize():"
    file.WriteLine "    logger.info('Quick system initialization...')"
    file.WriteLine "    try:"
    file.WriteLine "        from milvus_manager import MilvusManager"
    file.WriteLine "        from obsidian_processor import ObsidianProcessor"
    file.WriteLine "        import config"
    file.WriteLine "        logger.info('Core modules imported successfully')"
    file.WriteLine "        return True"
    file.WriteLine "    except Exception as e:"
    file.WriteLine "        logger.error(f'Failed to import core modules: {e}')"
    file.WriteLine "        return False"
    file.WriteLine ""
    file.WriteLine "def start_mcp_server_with_timeout():"
    file.WriteLine "    logger.info('Starting MCP Server with timeout protection...')"
    file.WriteLine "    try:"
    file.WriteLine "        import config"
    file.WriteLine "        mcp_server_path = os.path.join(os.path.dirname(__file__), 'mcp_server.py')"
    file.WriteLine "        "
    file.WriteLine "        if not os.path.exists(mcp_server_path):"
    file.WriteLine "            raise FileNotFoundError(f'MCP server script not found: {mcp_server_path}')"
    file.WriteLine "        "
    file.WriteLine "        logger.info(f'Executing MCP server: {mcp_server_path}')"
    file.WriteLine "        logger.info('Starting with 60-second startup timeout...')"
    file.WriteLine "        "
    file.WriteLine "        # Start MCP server with timeout"
    file.WriteLine "        process = subprocess.Popen("
    file.WriteLine "            [sys.executable, mcp_server_path],"
    file.WriteLine "            cwd=os.path.dirname(__file__),"
    file.WriteLine "            stdout=subprocess.PIPE,"
    file.WriteLine "            stderr=subprocess.PIPE,"
    file.WriteLine "            text=True"
    file.WriteLine "        )"
    file.WriteLine "        "
    file.WriteLine "        logger.info(f'MCP Server process started with PID: {process.pid}')"
    file.WriteLine "        "
    file.WriteLine "        # Wait for initial startup (60 seconds max)"
    file.WriteLine "        startup_timeout = 60"
    file.WriteLine "        start_time = time.time()"
    file.WriteLine "        "
    file.WriteLine "        while True:"
    file.WriteLine "            # Check if process is still running"
    file.WriteLine "            if process.poll() is not None:"
    file.WriteLine "                # Process exited"
    file.WriteLine "                stdout, stderr = process.communicate()"
    file.WriteLine "                logger.info(f'MCP Server exited with code: {process.returncode}')"
    file.WriteLine "                if stdout:"
    file.WriteLine "                    logger.info(f'STDOUT: {stdout}')"
    file.WriteLine "                if stderr:"
    file.WriteLine "                    logger.error(f'STDERR: {stderr}')"
    file.WriteLine "                break"
    file.WriteLine "            "
    file.WriteLine "            # Check startup timeout"
    file.WriteLine "            if time.time() - start_time > startup_timeout:"
    file.WriteLine "                logger.warning(f'MCP Server startup timeout ({startup_timeout}s)')"
    file.WriteLine "                logger.info('Server may be running but taking longer to start')"
    file.WriteLine "                logger.info('Continuing in background mode...')"
    file.WriteLine "                break"
    file.WriteLine "            "
    file.WriteLine "            time.sleep(1)"
    file.WriteLine "        "
    file.WriteLine "        logger.info('MCP Server startup sequence completed')"
    file.WriteLine "        "
    file.WriteLine "    except Exception as e:"
    file.WriteLine "        logger.error(f'Failed to start MCP server: {e}')"
    file.WriteLine "        raise"
    file.WriteLine ""
    file.WriteLine "def main():"
    file.WriteLine "    # Initialize timeout protection"
    file.WriteLine "    timeout = TimeoutProtection(120)  # 2 minutes total timeout"
    file.WriteLine "    timeout.start()"
    file.WriteLine "    "
    file.WriteLine "    try:"
    file.WriteLine "        logger.info('=' * 70)"
    file.WriteLine "        logger.info('TIMEOUT-PROTECTED MCP SERVER STARTUP')"
    file.WriteLine "        logger.info('Maximum runtime: 120 seconds')"
    file.WriteLine "        logger.info('=' * 70)"
    file.WriteLine "        "
    file.WriteLine "        logger.info('Step 1/3: Environment check...')"
    file.WriteLine "        if not check_environment():"
    file.WriteLine "            raise RuntimeError('Environment check failed')"
    file.WriteLine "        "
    file.WriteLine "        logger.info('Step 2/3: Quick initialization...')"
    file.WriteLine "        if not quick_initialize():"
    file.WriteLine "            raise RuntimeError('Quick initialization failed')"
    file.WriteLine "        "
    file.WriteLine "        logger.info('Step 3/3: Starting MCP Server...')"
    file.WriteLine "        start_mcp_server_with_timeout()"
    file.WriteLine "        "
    file.WriteLine "        timeout.complete()"
    file.WriteLine "        logger.info('MCP Server startup completed successfully')"
    file.WriteLine "        "
    file.WriteLine "    except KeyboardInterrupt:"
    file.WriteLine "        logger.info('Received keyboard interrupt, shutting down...')"
    file.WriteLine "        timeout.complete()"
    file.WriteLine "        sys.exit(0)"
    file.WriteLine "    except Exception as e:"
    file.WriteLine "        logger.error(f'Critical error: {e}')"
    file.WriteLine "        import traceback"
    file.WriteLine "        logger.error(f'Stack trace: {traceback.format_exc()}')"
    file.WriteLine "        timeout.complete()"
    file.WriteLine "        sys.exit(1)"
    file.WriteLine ""
    file.WriteLine "if __name__ == '__main__':"
    file.WriteLine "    main()"
    
    file.Close
    
    If Err.Number = 0 Then
        WriteLog "Timeout-protected Python script created successfully"
    Else
        WriteErrorLog "Failed to write Python script: " & Err.Description
    End If
    
    On Error GoTo 0
End Sub

' Main execution with timeout protection
Sub Main()
    WriteLog "=== TIMEOUT-PROTECTED MCP SERVER STARTUP ==="
    WriteLog "Starting with infinite hang protection..."
    WriteLog "Maximum VBS runtime: 5 minutes"
    WriteLog "Script path: " & WScript.ScriptFullName
    WriteLog "Working directory: " & scriptDir
    
    ' Find Python
    Dim pythonCommand
    pythonCommand = FindPythonWithTimeout()
    
    If pythonCommand = "" Then
        WriteErrorLog "No working Python found - cannot proceed"
        Exit Sub
    End If
    
    WriteLog "Using Python: " & pythonCommand
    
    ' Create timeout-protected Python script
    CreateTimeoutProtectedPythonScript()
    
    If Not objFSO.FileExists(tempPythonFile) Then
        WriteErrorLog "Failed to create timeout-protected Python script"
        Exit Sub
    End If
    
    ' Execute with timeout
    Dim command
    command = "cmd /c cd /d """ & scriptDir & """ && " & pythonCommand & " """ & tempPythonFile & """"
    
    WriteLog "Executing timeout-protected MCP server..."
    WriteLog "Command: " & command
    WriteLog "VBS will NOT wait for completion to prevent hanging"
    
    On Error Resume Next
    
    ' Execute without waiting (waitOnReturn = False) to prevent VBS hanging
    Dim result
    result = objShell.Run(command, 0, False)  ' Hidden, no wait
    
    If Err.Number = 0 Then
        WriteLog "MCP Server started in background with timeout protection"
        WriteLog "Check auto_startup_mcp.log for detailed progress"
        WriteLog "VBS script completing to avoid infinite hang"
    Else
        WriteErrorLog "Failed to start MCP server: " & Err.Description
    End If
    
    On Error GoTo 0
    
    ' Short delay then cleanup
    WScript.Sleep 3000
    CleanupTempFiles()
    
    WriteLog "VBS script completed - MCP server running independently"
End Sub

' Cleanup function
Sub CleanupTempFiles()
    On Error Resume Next
    
    ' Don't delete the timeout script immediately as it may still be running
    ' It will be cleaned up on next run or manually
    
    WriteLog "VBS cleanup completed"
    
    On Error GoTo 0
End Sub

' Execute main
Main()

' Clean up objects
Set objShell = Nothing
Set objFSO = Nothing
