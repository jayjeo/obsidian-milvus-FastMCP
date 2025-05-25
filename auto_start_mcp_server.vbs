' Auto Start MCP Server - Pure Option 1 Only (Invisible Mode)
' This script runs ONLY option 1 from main.py - no menu loop
' Completely invisible execution without command windows
' Enhanced with comprehensive logging for troubleshooting

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

' Function to log system information
Sub LogSystemInfo()
    WriteLog "=== SYSTEM INFORMATION ==="
    WriteDebugLog "VBS Script: " & WScript.ScriptFullName
    WriteDebugLog "Working Directory: " & scriptDir
    On Error Resume Next
    WriteDebugLog "Windows Version: " & objShell.RegRead("HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\ProductName")
    On Error GoTo 0
    WriteDebugLog "Computer Name: " & objShell.ExpandEnvironmentStrings("%COMPUTERNAME%")
    WriteDebugLog "User Name: " & objShell.ExpandEnvironmentStrings("%USERNAME%")
    WriteDebugLog "System Architecture: " & objShell.ExpandEnvironmentStrings("%PROCESSOR_ARCHITECTURE%")
    WriteDebugLog "Current Time: " & Now()
    WriteDebugLog "Script Start Time: " & startTime
End Sub

' Function to check Python installation
Function CheckPythonInstallation()
    WriteDebugLog "Checking Python installation..."
    
    On Error Resume Next
    ' Try different Python commands
    Dim pythonCommands, i, testResult, pythonFound
    pythonCommands = Array("python", "python3", "py")
    pythonFound = False
    
    For i = 0 To UBound(pythonCommands)
        WriteDebugLog "Testing Python command: " & pythonCommands(i)
        testResult = objShell.Run(pythonCommands(i) & " --version", 0, True)
        WriteDebugLog "Python test result for '" & pythonCommands(i) & "': " & testResult
        
        If testResult = 0 Then
            WriteLog "Python found with command: " & pythonCommands(i)
            pythonFound = True
            CheckPythonInstallation = pythonCommands(i)
            Exit Function
        End If
    Next
    
    If Not pythonFound Then
        WriteErrorLog "Python not found in PATH. Tested commands: python, python3, py"
        WriteErrorLog "Please ensure Python is installed and added to PATH"
        CheckPythonInstallation = ""
    End If
    
    On Error GoTo 0
End Function

' Function to check file dependencies
Sub CheckFileDependencies()
    WriteDebugLog "Checking file dependencies..."
    
    Dim requiredFiles, i
    requiredFiles = Array("config.py", "mcp_server.py", "milvus_manager.py", "obsidian_processor.py", "watcher.py")
    
    For i = 0 To UBound(requiredFiles)
        Dim filePath
        filePath = scriptDir & "\" & requiredFiles(i)
        If objFSO.FileExists(filePath) Then
            WriteDebugLog "Required file found: " & requiredFiles(i)
        Else
            WriteErrorLog "Required file missing: " & requiredFiles(i) & " (" & filePath & ")"
        End If
    Next
End Sub

' Function to check disk space
Sub CheckDiskSpace()
    On Error Resume Next
    Dim drive, freeSpace, totalSpace, usedPercent
    Set drive = objFSO.GetDrive(objFSO.GetDriveName(scriptDir))
    
    If Err.Number = 0 Then
        freeSpace = drive.FreeSpace / 1024 / 1024 / 1024 ' Convert to GB
        totalSpace = drive.TotalSize / 1024 / 1024 / 1024 ' Convert to GB
        usedPercent = ((totalSpace - freeSpace) / totalSpace) * 100
        
        WriteDebugLog "Disk Space - Free: " & Round(freeSpace, 2) & " GB, Total: " & Round(totalSpace, 2) & " GB, Used: " & Round(usedPercent, 1) & "%"
        
        If freeSpace < 1 Then
            WriteWarningLog "Low disk space warning: Only " & Round(freeSpace, 2) & " GB free"
        End If
    Else
        WriteWarningLog "Could not check disk space: " & Err.Description
    End If
    On Error GoTo 0
End Sub

' Function to create Pure Option 1 Python script - Fixed version
Sub CreatePureOption1PythonScript()
    WriteDebugLog "Creating Pure Option 1 Python script..."
    WriteDebugLog "Target file: " & tempPythonFile
    
    ' Write Python code to temporary file directly to avoid VBScript string issues
    On Error Resume Next
    Dim file
    
    ' Check if we can write to the directory
    If Not objFSO.FolderExists(scriptDir) Then
        WriteErrorLog "Script directory does not exist: " & scriptDir
        Exit Sub
    End If
    
    ' Try to create the file
    Set file = objFSO.CreateTextFile(tempPythonFile, True, False) ' True = overwrite, False = ASCII
    If Err.Number <> 0 Then
        WriteErrorLog "Failed to create temporary Python file: " & Err.Description & " (Error: " & Err.Number & ")"
        WriteErrorLog "Target path: " & tempPythonFile
        WriteErrorLog "Please check file permissions and disk space"
        Exit Sub
    End If
    
    ' Write Python code line by line to avoid VBScript string concatenation issues
    file.WriteLine "#!/usr/bin/env python3"
    file.WriteLine "# Pure Option 1 MCP Server - No Menu Loop"
    file.WriteLine "# This script replicates ONLY option 1 from main.py"
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
    file.WriteLine "def initialize():"
    file.WriteLine "    logger.info('Obsidian-Milvus-MCP system initializing...')"
    file.WriteLine "    try:"
    file.WriteLine "        from milvus_manager import MilvusManager"
    file.WriteLine "        from obsidian_processor import ObsidianProcessor"
    file.WriteLine "        import config"
    file.WriteLine "        logger.info('Connecting to Milvus...')"
    file.WriteLine "        milvus_manager = MilvusManager()"
    file.WriteLine "        logger.info('Initializing Obsidian processor...')"
    file.WriteLine "        processor = ObsidianProcessor(milvus_manager)"
    file.WriteLine "        try:"
    file.WriteLine "            results = milvus_manager.query('id >= 0', limit=1)"
    file.WriteLine "            if not results:"
    file.WriteLine "                logger.info('No existing data found. MCP server ready for indexing.')"
    file.WriteLine "            else:"
    file.WriteLine "                entity_count = milvus_manager.count_entities()"
    file.WriteLine "                logger.info(f'Found existing data: {entity_count} documents indexed.')"
    file.WriteLine "        except Exception as e:"
    file.WriteLine "            logger.warning(f'Could not check existing data: {e}')"
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
    file.WriteLine "        logger.info(f'Monitoring directory: {config.OBSIDIAN_VAULT_PATH}')"
    file.WriteLine "        return watcher_thread"
    file.WriteLine "    except Exception as e:"
    file.WriteLine "        logger.error(f'Failed to start file watcher: {e}')"
    file.WriteLine "        return None"
    file.WriteLine ""
    file.WriteLine "def start_mcp_server_blocking():"
    file.WriteLine "    try:"
    file.WriteLine "        import config"
    file.WriteLine "        logger.info('Starting MCP Server for Claude Desktop...')"
    file.WriteLine "        logger.info(f'Server name: {config.FASTMCP_SERVER_NAME}')"
    file.WriteLine "        logger.info(f'Transport: {config.FASTMCP_TRANSPORT}')"
    file.WriteLine "        mcp_server_path = os.path.join(os.path.dirname(__file__), 'mcp_server.py')"
    file.WriteLine "        if not os.path.exists(mcp_server_path):"
    file.WriteLine "            raise FileNotFoundError(f'MCP server script not found: {mcp_server_path}')"
    file.WriteLine "        logger.info(f'Executing MCP server: {mcp_server_path}')"
    file.WriteLine "        logger.info('This is PURE OPTION 1 - server will run continuously')"
    file.WriteLine "        result = subprocess.run(["
    file.WriteLine "            sys.executable, mcp_server_path"
    file.WriteLine "        ], cwd=os.path.dirname(__file__))"
    file.WriteLine "        logger.info(f'MCP Server exited with return code: {result.returncode}')"
    file.WriteLine "    except Exception as e:"
    file.WriteLine "        logger.error(f'Failed to start MCP server: {e}')"
    file.WriteLine "        raise"
    file.WriteLine ""
    file.WriteLine "def main():"
    file.WriteLine "    try:"
    file.WriteLine "        logger.info('=' * 70)"
    file.WriteLine "        logger.info('PURE OPTION 1 MCP SERVER - AUTO START MODE')"
    file.WriteLine "        logger.info('This script runs ONLY option 1 behavior from main.py')"
    file.WriteLine "        logger.info('NO MENU LOOP - Continuous MCP server operation')"
    file.WriteLine "        logger.info('=' * 70)"
    file.WriteLine "        logger.info('Step 1/3: Initializing system...')"
    file.WriteLine "        processor = initialize()"
    file.WriteLine "        logger.info('Step 2/3: Starting file watcher...')"
    file.WriteLine "        watcher_thread = start_file_watcher(processor)"
    file.WriteLine "        if watcher_thread is None:"
    file.WriteLine "            logger.warning('File watcher failed to start, but continuing with MCP server...')"
    file.WriteLine "        else:"
    file.WriteLine "            logger.info('File watcher started successfully')"
    file.WriteLine "        logger.info('Waiting for complete initialization...')"
    file.WriteLine "        time.sleep(3)"
    file.WriteLine "        logger.info('Step 3/3: Starting MCP Server (blocking mode)...')"
    file.WriteLine "        logger.info('MCP Server will now run continuously')"
    file.WriteLine "        logger.info('No menu will appear - this is pure option 1 mode')"
    file.WriteLine "        start_mcp_server_blocking()"
    file.WriteLine "        logger.info('MCP Server has exited - script ending')"
    file.WriteLine "    except KeyboardInterrupt:"
    file.WriteLine "        logger.info('Received keyboard interrupt, shutting down...')"
    file.WriteLine "        sys.exit(0)"
    file.WriteLine "    except Exception as e:"
    file.WriteLine "        logger.error(f'Critical error in main: {e}')"
    file.WriteLine "        import traceback"
    file.WriteLine "        logger.error(f'Stack trace: {traceback.format_exc()}')"
    file.WriteLine "        logger.error('Exiting with error code 1')"
    file.WriteLine "        sys.exit(1)"
    file.WriteLine ""
    file.WriteLine "if __name__ == '__main__':"
    file.WriteLine "    main()"
    
    If Err.Number <> 0 Then
        WriteErrorLog "Failed to write Python code to file: " & Err.Description & " (Error: " & Err.Number & ")"
        file.Close
        Exit Sub
    End If
    
    file.Close
    
    ' Verify file was created and has content
    If objFSO.FileExists(tempPythonFile) Then
        Dim fileSize
        fileSize = objFSO.GetFile(tempPythonFile).Size
        WriteLog "Pure Option 1 Python script created successfully: " & tempPythonFile
        WriteDebugLog "Python script file size: " & fileSize & " bytes"
        If fileSize < 2000 Then
            WriteWarningLog "Python script file seems too small: " & fileSize & " bytes"
        End If
    Else
        WriteErrorLog "Python script file was not created despite no error reported"
    End If
    
    On Error GoTo 0
End Sub

' Function to cleanup temporary files with enhanced logging
Sub CleanupTempFiles()
    WriteDebugLog "Starting cleanup of temporary files..."
    
    On Error Resume Next
    If objFSO.FileExists(tempPythonFile) Then
        WriteDebugLog "Attempting to delete: " & tempPythonFile
        objFSO.DeleteFile tempPythonFile
        If Err.Number = 0 Then
            WriteLog "Temporary Python file cleaned up successfully: " & tempPythonFile
        Else
            WriteWarningLog "Failed to delete temporary file: " & Err.Description
        End If
    Else
        WriteDebugLog "Temporary file does not exist, no cleanup needed: " & tempPythonFile
    End If
    On Error GoTo 0
End Sub

' Function to generate final summary
Sub LogFinalSummary()
    Dim endTime, duration
    endTime = Now()
    duration = DateDiff("s", startTime, endTime)
    
    WriteLog "=== EXECUTION SUMMARY ==="
    WriteLog "Start Time: " & startTime
    WriteLog "End Time: " & endTime
    WriteLog "Duration: " & duration & " seconds"
    WriteLog "Errors: " & errorCount
    WriteLog "Warnings: " & warningCount
    
    If errorCount > 0 Then
        WriteLog "Status: FAILED (" & errorCount & " errors)"
        WriteLog "Please check the debug log for detailed error information: " & debugLogFile
    ElseIf warningCount > 0 Then
        WriteLog "Status: SUCCESS with warnings (" & warningCount & " warnings)"
    Else
        WriteLog "Status: SUCCESS"
    End If
    
    WriteLog "Log files:"
    WriteLog "  Main log: " & logFile
    WriteLog "  Debug log: " & debugLogFile
    WriteLog "  Python log: " & scriptDir & "\auto_startup_mcp.log"
End Sub

' Enhanced main execution with comprehensive logging
Sub Main()
    WriteLog "=== VBS PURE OPTION 1 MCP SERVER (INVISIBLE) ==="
    WriteLog "Starting execution at: " & Now()
    WriteLog "Script version: Pure Option 1 Only - No Menu Loop"
    WriteLog "Script path: " & WScript.ScriptFullName
    WriteLog "Working directory: " & scriptDir
    WriteLog "Temporary Python script: " & tempPythonFile
    
    ' Log system information
    LogSystemInfo()
    
    ' Check disk space
    CheckDiskSpace()
    
    ' Check file dependencies
    CheckFileDependencies()
    
    ' Check Python installation
    WriteLog "=== PYTHON ENVIRONMENT CHECK ==="
    Dim pythonCommand
    pythonCommand = CheckPythonInstallation()
    
    If pythonCommand = "" Then
        WriteErrorLog "Python installation check failed - cannot proceed"
        LogFinalSummary()
        Exit Sub
    End If
    
    ' Create Pure Option 1 Python script
    WriteLog "=== CREATING PURE OPTION 1 PYTHON SCRIPT ==="
    CreatePureOption1PythonScript()
    
    ' Verify Python script was created
    If Not objFSO.FileExists(tempPythonFile) Then
        WriteErrorLog "Failed to create Pure Option 1 Python script - cannot proceed"
        LogFinalSummary()
        Exit Sub
    End If
    
    WriteLog "Pure Option 1 Python script verified, proceeding to execute"
    
    ' Build and execute command
    WriteLog "=== EXECUTING PURE OPTION 1 PYTHON SCRIPT ==="
    Dim command
    command = "cmd /c cd /d """ & scriptDir & """ && " & pythonCommand & " """ & tempPythonFile & """"
    
    WriteLog "Command to execute: " & command
    WriteDebugLog "Execution method: Completely invisible (windowStyle=0, waitOnReturn=False)"
    WriteLog "NOTE: This will run ONLY Option 1 behavior - no menu loop!"
    
    On Error Resume Next
    Dim result, executionStartTime
    executionStartTime = Now()
    
    WriteDebugLog "Starting Pure Option 1 execution at: " & executionStartTime
    result = objShell.Run(command, 0, False) ' Completely hidden, don't wait
    
    If Err.Number = 0 Then
        WriteLog "Pure Option 1 script execution initiated successfully (invisible mode)"
        WriteLog "Return code: " & result
        WriteLog "MCP Server (Option 1 only) should be starting invisibly in background"
        WriteLog "NO MENU LOOP will appear - this is pure option 1 mode"
        WriteDebugLog "Execution initiated in: " & DateDiff("s", executionStartTime, Now()) & " seconds"
    Else
        WriteErrorLog "Failed to execute Pure Option 1 script"
        WriteErrorLog "Error Number: " & Err.Number
        WriteErrorLog "Error Description: " & Err.Description
        WriteErrorLog "Error Source: " & Err.Source
        WriteErrorLog "Command attempted: " & command
    End If
    On Error GoTo 0
    
    ' Wait a bit before cleanup to allow process to start
    WriteDebugLog "Waiting 5 seconds before cleanup to allow process startup..."
    WScript.Sleep 5000
    
    ' Cleanup temporary files
    WriteLog "=== CLEANUP ==="
    CleanupTempFiles()
    
    ' Generate final summary
    LogFinalSummary()
End Sub

' Execute main function
Main()

' Clean up objects
Set objShell = Nothing
Set objFSO = Nothing
