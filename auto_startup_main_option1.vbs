' Auto-Startup Script for main.py Option 1 (Start MCP Server)
' This script automatically starts the MCP server on Windows boot
' Uses only relative paths and English text to avoid encoding issues

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Get project directory (relative path only)
projectDir = fso.GetParentFolderName(WScript.ScriptFullName)
logFile = projectDir & "\auto_startup_main.log"

' Function to write log with timestamp
Sub WriteLog(message)
    On Error Resume Next
    Set logHandle = fso.OpenTextFile(logFile, 8, True)
    logHandle.WriteLine Now & " - " & message
    logHandle.Close
    On Error GoTo 0
End Sub

' Function to find Python executable
Function FindPythonPath()
    Dim pythonPath, testResult
    
    ' Try python command in PATH first
    Set testResult = shell.Exec("python --version")
    Do While testResult.Status = 0
        WScript.Sleep 100
    Loop
    
    If testResult.ExitCode = 0 Then
        WriteLog "Found Python in PATH"
        FindPythonPath = "python"
        Exit Function
    End If
    
    ' Try python3 command
    Set testResult = shell.Exec("python3 --version")
    Do While testResult.Status = 0
        WScript.Sleep 100
    Loop
    
    If testResult.ExitCode = 0 Then
        WriteLog "Found Python3 in PATH"
        FindPythonPath = "python3"
        Exit Function
    End If
    
    ' Try py launcher
    Set testResult = shell.Exec("py --version")
    Do While testResult.Status = 0
        WScript.Sleep 100
    Loop
    
    If testResult.ExitCode = 0 Then
        WriteLog "Found Python via py launcher"
        FindPythonPath = "py"
        Exit Function
    End If
    
    WriteLog "ERROR: Python not found in PATH"
    FindPythonPath = ""
End Function

' Function to check if Milvus is already running
Function IsMilvusRunning()
    On Error Resume Next
    
    ' Change to project directory first
    shell.CurrentDirectory = projectDir
    
    ' Try to find Podman path from config using relative path
    Set result = shell.Exec("python -c ""from config import get_podman_path; print(get_podman_path())""")
    Do While result.Status = 0
        WScript.Sleep 100
    Loop
    
    If result.ExitCode = 0 Then
        podmanPath = Trim(result.StdOut.ReadAll())
        WriteLog "Found Podman path: " & podmanPath
        
        ' Check if Milvus containers are running
        Set psResult = shell.Exec("""" & podmanPath & """ ps --filter name=milvus")
        Do While psResult.Status = 0
            WScript.Sleep 100
        Loop
        
        If psResult.ExitCode = 0 Then
            psOutput = psResult.StdOut.ReadAll()
            If InStr(psOutput, "milvus") > 0 Then
                WriteLog "Milvus containers are already running"
                IsMilvusRunning = True
            Else
                WriteLog "Milvus containers are not running"
                IsMilvusRunning = False
            End If
        Else
            WriteLog "Could not check container status"
            IsMilvusRunning = False
        End If
    Else
        WriteLog "Could not get Podman path from config"
        IsMilvusRunning = False
    End If
    
    On Error GoTo 0
End Function

' Function to start Milvus if not running
Sub EnsureMilvusRunning()
    If Not IsMilvusRunning() Then
        WriteLog "Starting Milvus services first..."
        
        ' Try to start Milvus using start-milvus.bat (relative path)
        startScript = projectDir & "\start-milvus.bat"
        If fso.FileExists(startScript) Then
            WriteLog "Executing start-milvus.bat"
            shell.Run """" & startScript & """", 0, True
            
            ' Wait for Milvus to start
            WriteLog "Waiting for Milvus to initialize (60 seconds)"
            WScript.Sleep 60000
            
            ' Verify Milvus is running
            If IsMilvusRunning() Then
                WriteLog "Milvus started successfully"
            Else
                WriteLog "Warning: Milvus may not have started properly"
            End If
        Else
            WriteLog "ERROR: start-milvus.bat not found"
        End If
    Else
        WriteLog "Milvus is already running"
    End If
End Sub

' Main execution starts here
WriteLog "========================================="
WriteLog "Starting auto-startup for main.py option 1"
WriteLog "Project directory: " & projectDir

' Wait for system to be ready (Windows needs time to fully boot)
WriteLog "Waiting for system to be ready (90 seconds)"
WScript.Sleep 90000

' Find Python executable
pythonPath = FindPythonPath()
If Len(pythonPath) = 0 Then
    WriteLog "ERROR: Python not found. Cannot start MCP server."
    WScript.Quit 1
End If

WriteLog "Found Python: " & pythonPath

' Change to project directory
shell.CurrentDirectory = projectDir
WriteLog "Working directory: " & projectDir

' Check if main.py exists (relative path)
mainScript = projectDir & "\main.py"
If Not fso.FileExists(mainScript) Then
    WriteLog "ERROR: main.py not found in " & projectDir
    WScript.Quit 1
End If

WriteLog "Found main.py: " & mainScript

' Ensure Milvus is running before starting MCP server
EnsureMilvusRunning()

' Start main.py with option 1 (Start MCP Server)
' This reflects exactly what main.py option 1 does: start MCP server
WriteLog "Starting MCP server (main.py option 1)"

' Create a temporary input file with option 1 (avoiding encoding issues)
tempInput = projectDir & "\temp_input.txt"
Set tempFile = fso.CreateTextFile(tempInput, True, False)  ' False = ANSI encoding to avoid UTF-8 BOM
tempFile.WriteLine "1"
tempFile.Close

' Create a temporary batch file to run main.py with input redirection
tempBatch = projectDir & "\temp_start_option1.bat"
Set tempFile = fso.CreateTextFile(tempBatch, True, False)  ' ANSI encoding
tempFile.WriteLine "@echo off"
tempFile.WriteLine "cd /d """ & projectDir & """"
tempFile.WriteLine "chcp 65001 >nul 2>&1"  ' Set UTF-8 for Python
tempFile.WriteLine "" & pythonPath & " main.py < temp_input.txt"
tempFile.WriteLine "del temp_input.txt >nul 2>&1"
tempFile.WriteLine "del temp_start_option1.bat >nul 2>&1"
tempFile.Close

' Execute the temporary batch file hidden and let it run independently
WriteLog "Executing main.py with option 1 input"
shell.Run """" & tempBatch & """", 0, False

WriteLog "MCP server startup initiated"
WriteLog "Auto-startup completed successfully"
WriteLog "========================================="
