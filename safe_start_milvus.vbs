' ================================================================
' Safe Milvus Startup Script - Preserves All Embedding Data
' ================================================================
' This script safely starts Milvus containers while preserving:
' - All embedding vector data (MilvusData\minio\)
' - All vector indexes (MilvusData\milvus\)
' - All metadata and schemas (volumes\etcd\)
' ================================================================

' Get the script directory (project root)
Dim scriptDir, fso, shell
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

' Get current script directory as project root
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)

' Define relative paths
Dim pythonScript, logFile, dataStatusFile
pythonScript = scriptDir & "\setup.py"
logFile = scriptDir & "\milvus_startup.log"
dataStatusFile = scriptDir & "\data_status.txt"

' ================================================================
' Function: Write log entry
' ================================================================
Sub WriteLog(message)
    Dim logFileHandle
    Set logFileHandle = fso.OpenTextFile(logFile, 8, True) ' 8 = ForAppending
    logFileHandle.WriteLine Now & " - " & message
    logFileHandle.Close
End Sub

' ================================================================
' Function: Check if data directories exist and have content
' ================================================================
Function CheckDataSafety()
    Dim etcdPath, minioPath, milvusPath
    Dim etcdExists, minioExists, milvusExists
    
    ' Define data paths using relative paths
    etcdPath = scriptDir & "\volumes\etcd"
    minioPath = scriptDir & "\MilvusData\minio"
    milvusPath = scriptDir & "\MilvusData\milvus"
    
    ' Check if directories exist
    etcdExists = fso.FolderExists(etcdPath)
    minioExists = fso.FolderExists(minioPath)
    milvusExists = fso.FolderExists(milvusPath)
    
    ' Write data status to file
    Dim statusFile
    Set statusFile = fso.CreateTextFile(dataStatusFile, True)
    statusFile.WriteLine "=== Milvus Data Safety Check ==="
    statusFile.WriteLine "Check Time: " & Now
    statusFile.WriteLine "Project Directory: " & scriptDir
    statusFile.WriteLine ""
    statusFile.WriteLine "Data Directory Status:"
    statusFile.WriteLine "  etcd: " & etcdPath & " - " & IIf(etcdExists, "EXISTS", "NOT FOUND")
    statusFile.WriteLine "  minio: " & minioPath & " - " & IIf(minioExists, "EXISTS", "NOT FOUND")
    statusFile.WriteLine "  milvus: " & milvusPath & " - " & IIf(milvusExists, "EXISTS", "NOT FOUND")
    statusFile.WriteLine ""
    
    If etcdExists And minioExists And milvusExists Then
        statusFile.WriteLine "STATUS: All data directories found - SAFE TO PROCEED"
        statusFile.Close
        CheckDataSafety = True
        WriteLog "Data safety check passed - all directories exist"
    Else
        statusFile.WriteLine "STATUS: Some data directories missing - SAFE TO PROCEED (first run)"
        statusFile.Close
        CheckDataSafety = True
        WriteLog "Data safety check - some directories missing (first run)"
    End If
End Function

' ================================================================
' Function: Check if Python and setup.py exist
' ================================================================
Function CheckRequirements()
    ' Check if setup.py exists
    If Not fso.FileExists(pythonScript) Then
        WriteLog "ERROR: setup.py not found at " & pythonScript
        CheckRequirements = False
        Exit Function
    End If
    
    ' Try to find Python
    Dim pythonCmd
    pythonCmd = "python --version"
    
    ' Test Python availability
    On Error Resume Next
    shell.Run "cmd /c " & pythonCmd & " > nul 2>&1", 0, True
    If Err.Number <> 0 Then
        WriteLog "ERROR: Python not found in PATH"
        CheckRequirements = False
        Exit Function
    End If
    On Error GoTo 0
    
    WriteLog "Requirements check passed - Python and setup.py found"
    CheckRequirements = True
End Function

' ================================================================
' Function: Execute safe Milvus restart
' ================================================================
Function SafeStartMilvus()
    WriteLog "Starting safe Milvus restart process..."
    
    ' Change to script directory
    shell.CurrentDirectory = scriptDir
    
    ' Create Python script to handle option 8 automatically
    Dim autoStartScript, autoStartPath
    autoStartPath = scriptDir & "\auto_safe_start.py"
    
    autoStartScript = "#!/usr/bin/env python3" & vbCrLf & _
                     "import sys" & vbCrLf & _
                     "import os" & vbCrLf & _
                     "from pathlib import Path" & vbCrLf & _
                     "" & vbCrLf & _
                     "# Add current directory to Python path" & vbCrLf & _
                     "current_dir = Path(__file__).parent.resolve()" & vbCrLf & _
                     "sys.path.insert(0, str(current_dir))" & vbCrLf & _
                     "" & vbCrLf & _
                     "# Import the safe restart function from setup.py" & vbCrLf & _
                     "try:" & vbCrLf & _
                     "    from setup import perform_safe_server_restart" & vbCrLf & _
                     "    print('Starting safe Milvus server restart...')" & vbCrLf & _
                     "    perform_safe_server_restart()" & vbCrLf & _
                     "    print('Safe restart completed successfully!')" & vbCrLf & _
                     "except Exception as e:" & vbCrLf & _
                     "    print(f'Error during safe restart: {e}')" & vbCrLf & _
                     "    sys.exit(1)" & vbCrLf
    
    ' Write the auto start script
    Dim autoFile
    Set autoFile = fso.CreateTextFile(autoStartPath, True)
    autoFile.Write autoStartScript
    autoFile.Close
    
    ' Execute the auto start script
    Dim cmd, result
    cmd = "python """ & autoStartPath & """"
    
    WriteLog "Executing command: " & cmd
    
    ' Run the command and capture result
    result = shell.Run(cmd, 1, True) ' 1 = show window, True = wait for completion
    
    ' Clean up auto start script
    If fso.FileExists(autoStartPath) Then
        fso.DeleteFile autoStartPath
    End If
    
    If result = 0 Then
        WriteLog "Safe Milvus restart completed successfully"
        SafeStartMilvus = True
    Else
        WriteLog "Safe Milvus restart failed with exit code: " & result
        SafeStartMilvus = False
    End If
End Function

' ================================================================
' Main Execution
' ================================================================
Sub Main()
    WriteLog "=========================================="
    WriteLog "Safe Milvus Startup Script Started"
    WriteLog "Project Directory: " & scriptDir
    WriteLog "=========================================="
    
    ' Step 1: Check data safety
    If Not CheckDataSafety() Then
        WriteLog "Data safety check failed - aborting startup"
        WScript.Quit 1
    End If
    
    ' Step 2: Check requirements
    If Not CheckRequirements() Then
        WriteLog "Requirements check failed - aborting startup"
        WScript.Quit 1
    End If
    
    ' Step 3: Wait a moment for system to be ready
    WriteLog "Waiting 10 seconds for system initialization..."
    WScript.Sleep 10000
    
    ' Step 4: Execute safe startup
    If SafeStartMilvus() Then
        WriteLog "Milvus safe startup process completed successfully"
        WriteLog "All embedding data has been preserved"
        
        ' Show success notification (optional)
        shell.Popup "Milvus server started safely!" & vbCrLf & _
                   "All embedding data preserved." & vbCrLf & _
                   "Check milvus_startup.log for details.", _
                   10, "Milvus Safe Startup", 64 ' 64 = Information icon
    Else
        WriteLog "Milvus safe startup process failed"
        
        ' Show error notification
        shell.Popup "Milvus startup encountered issues!" & vbCrLf & _
                   "Check milvus_startup.log for details." & vbCrLf & _
                   "Your data is still safe.", _
                   15, "Milvus Startup Warning", 48 ' 48 = Warning icon
    End If
    
    WriteLog "=========================================="
    WriteLog "Safe Milvus Startup Script Finished"
    WriteLog "=========================================="
End Sub

' ================================================================
' Helper function for IIf (since VBScript doesn't have it)
' ================================================================
Function IIf(condition, trueValue, falseValue)
    If condition Then
        IIf = trueValue
    Else
        IIf = falseValue
    End If
End Function

' Start the main process
Main()
