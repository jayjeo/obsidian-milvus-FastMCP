' ================================================================
' Podman Auto-Startup Script for Windows
' This script starts Podman machine and Milvus containers
' Uses relative paths from config.py for portability
' ================================================================

Dim fso, shell, projectDir
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

' Get project directory from script location
projectDir = fso.GetParentFolderName(WScript.ScriptFullName)

' Define paths using config.py variables
Dim podmanPath, logFile, startupComplete, composeFile
podmanPath = "C:\\Program Files\\RedHat\\Podman\\podman.EXE"
logFile = projectDir & "\podman_startup.log"
startupComplete = projectDir & "\startup_complete.flag"
composeFile = projectDir & "\milvus-podman-compose.yml"

' Function to write log entries
Sub WriteLog(message)
    Dim logFileHandle
    Set logFileHandle = fso.OpenTextFile(logFile, 8, True)
    logFileHandle.WriteLine Now & " - " & message
    logFileHandle.Close
End Sub

' Function to run command silently
Function RunCommand(cmd)
    RunCommand = shell.Run(cmd, 0, True)
End Function

' Main startup process
Sub Main()
    WriteLog "=========================================="
    WriteLog "Podman Auto-Startup Script Started"
    WriteLog "Project Directory: " & projectDir
    WriteLog "Podman Path: " & podmanPath
    WriteLog "=========================================="
    
    ' Wait for system to be ready
    WriteLog "Waiting 30 seconds for system initialization..."
    WScript.Sleep 30000
    
    ' Check if Podman executable exists
    If Not fso.FileExists(podmanPath) Then
        WriteLog "ERROR: Podman not found at " & podmanPath
        Exit Sub
    End If
    
    ' Start Podman machine (Windows may need this)
    WriteLog "Starting Podman machine..."
    Dim result
    result = RunCommand(""" & podmanPath & "" machine start")
    If result = 0 Then
        WriteLog "Podman machine started successfully"
    Else
        WriteLog "Podman machine start returned code: " & result & " (may already be running)"
    End If
    
    ' Wait for Podman machine to be ready
    WriteLog "Waiting 20 seconds for Podman machine to be ready..."
    WScript.Sleep 20000
    
    ' Check if compose file exists and start containers
    If fso.FileExists(composeFile) Then
        WriteLog "Starting Milvus containers using compose file..."
        
        ' Change to project directory and start containers
        shell.CurrentDirectory = projectDir
        result = RunCommand(""" & podmanPath & "" compose -f "" & composeFile & "" up -d")
        
        If result = 0 Then
            WriteLog "Milvus containers started successfully"
        Else
            WriteLog "Container startup returned code: " & result
        End If
        
        ' Additional wait for containers to be ready
        WriteLog "Waiting 30 seconds for containers to be ready..."
        WScript.Sleep 30000
        
    Else
        WriteLog "WARNING: Compose file not found: " & composeFile
        WriteLog "Skipping container startup"
    End If
    
    ' Create completion flag
    Dim flagFile
    Set flagFile = fso.CreateTextFile(startupComplete, True)
    flagFile.WriteLine "Startup completed at: " & Now
    flagFile.WriteLine "Podman Path: " & podmanPath
    flagFile.WriteLine "Project Directory: " & projectDir
    flagFile.Close
    
    WriteLog "=========================================="
    WriteLog "Podman Auto-Startup Script Completed"
    WriteLog "=========================================="
End Sub

' Start the main process
Main()
