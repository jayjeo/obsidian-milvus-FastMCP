' Auto Start MCP Server - VBS Script for Windows Startup
' This script automatically starts the Milvus MCP server on Windows boot
' Place this script in Windows Startup folder or create a scheduled task

Dim objShell, objFSO, scriptDir, pythonScript, logFile, startTime

' Initialize objects
Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

' Get script directory (relative path approach)
scriptDir = objFSO.GetParentFolderName(WScript.ScriptFullName)
pythonScript = scriptDir & "\auto_start_mcp_server.py"
logFile = scriptDir & "\vbs_startup.log"

' Get current timestamp
startTime = Now()

' Function to write log messages
Sub WriteLog(message)
    Dim logText
    logText = FormatDateTime(Now(), 0) & " - " & message & vbCrLf
    
    On Error Resume Next
    Dim file
    Set file = objFSO.OpenTextFile(logFile, 8, True) ' 8 = ForAppending, True = Create if not exists
    If Err.Number = 0 Then
        file.Write logText
        file.Close
    End If
    On Error GoTo 0
End Sub

' Main execution
Sub Main()
    WriteLog "VBS Auto Start Script - Beginning execution"
    WriteLog "Script directory: " & scriptDir
    WriteLog "Python script path: " & pythonScript
    
    ' Check if Python script exists
    If Not objFSO.FileExists(pythonScript) Then
        WriteLog "ERROR: Python script not found: " & pythonScript
        MsgBox "Error: Python script not found at " & pythonScript, vbCritical, "Auto Start MCP Server"
        Exit Sub
    End If
    
    WriteLog "Python script found, proceeding to execute"
    
    ' Check if Python is available
    Dim pythonPath
    pythonPath = "python"
    
    ' Try to find Python executable
    On Error Resume Next
    objShell.Run "python --version", 0, False
    If Err.Number <> 0 Then
        WriteLog "WARNING: Python may not be in PATH, trying alternative paths"
        ' Could add more Python path detection here if needed
    End If
    On Error GoTo 0
    
    ' Build command to execute Python script
    Dim command
    command = "cmd /c cd /d """ & scriptDir & """ && python """ & pythonScript & """"
    
    WriteLog "Executing command: " & command
    
    ' Execute the Python script
    ' Using Run with arguments: command, windowStyle, waitOnReturn
    ' windowStyle: 0 = hidden, 1 = normal, 2 = minimized, 3 = maximized
    ' waitOnReturn: True = wait for completion, False = continue immediately
    
    On Error Resume Next
    Dim result
    result = objShell.Run(command, 2, False) ' Minimized window, don't wait
    
    If Err.Number = 0 Then
        WriteLog "Python script execution initiated successfully"
        WriteLog "MCP Server should be starting in minimized window"
    Else
        WriteLog "ERROR: Failed to execute Python script - " & Err.Description
        MsgBox "Failed to start MCP Server: " & Err.Description, vbCritical, "Auto Start MCP Server"
    End If
    On Error GoTo 0
    
    WriteLog "VBS script execution completed"
End Sub

' Execute main function
Main()

' Clean up objects
Set objShell = Nothing
Set objFSO = Nothing
