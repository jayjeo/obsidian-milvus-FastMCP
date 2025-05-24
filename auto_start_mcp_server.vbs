Option Explicit

' Obsidian-Milvus MCP Server Auto Start Script - Universal Version with Fixed Process Detection
' Auto-start mcp_server.py without window on computer restart and show notification
' Works with relative paths for easy sharing

Dim objShell, objFSO, objWMIService, objProcesses
Dim projectPath, pythonPath, scriptPath, logPath
Dim objProcess, objLogFile
Dim serverStarted, checkCount

' Get current script directory as project path
Set objFSO = CreateObject("Scripting.FileSystemObject")
projectPath = objFSO.GetParentFolderName(WScript.ScriptFullName)
scriptPath = projectPath & "\mcp_server.py"
logPath = projectPath & "\auto_start_detailed.log"

Set objShell = CreateObject("WScript.Shell")

' Initialize log file
Set objLogFile = objFSO.CreateTextFile(logPath, True)
objLogFile.WriteLine "[" & Now() & "] MCP Server Auto Start Script Started"
objLogFile.WriteLine "[" & Now() & "] Project path (auto-detected): " & projectPath

' Detect Python path (check multiple possible locations)
pythonPath = FindPythonPath()

If pythonPath = "" Then
    objLogFile.WriteLine "[" & Now() & "] Error: Python executable not found."
    objLogFile.Close
    MsgBox "Error: Python is not installed or not registered in PATH." & vbCrLf & vbCrLf & _
           "Please install Python and make sure it's in your PATH environment variable.", _
           vbCritical, "MCP Server Auto Start - Python Not Found"
    WScript.Quit
End If

objLogFile.WriteLine "[" & Now() & "] Python path: " & pythonPath
objLogFile.WriteLine "[" & Now() & "] Script path: " & scriptPath

' Check script file exists
If Not objFSO.FileExists(scriptPath) Then
    objLogFile.WriteLine "[" & Now() & "] Error: mcp_server.py file not found: " & scriptPath
    objLogFile.Close
    MsgBox "Error: mcp_server.py file not found in current directory." & vbCrLf & vbCrLf & _
           "Expected location: " & scriptPath & vbCrLf & vbCrLf & _
           "Please make sure this VBS file is in the same folder as mcp_server.py", _
           vbCritical, "MCP Server Auto Start - Script Not Found"
    WScript.Quit
End If

' Check if MCP server is already running (improved detection)
If IsMCPServerActuallyRunning() Then
    objLogFile.WriteLine "[" & Now() & "] MCP server is already running."
    objLogFile.Close
    MsgBox "MCP server is already running." & vbCrLf & vbCrLf & _
           "Project folder: " & projectPath, _
           vbInformation, "MCP Server Auto Start - Already Running"
    WScript.Quit
End If

objLogFile.WriteLine "[" & Now() & "] No MCP server found. Starting server..."

' Change working directory and execute Python script (without window)
objLogFile.WriteLine "[" & Now() & "] Starting MCP server..."

Dim command
command = "cmd /c cd /d """ & projectPath & """ && """ & pythonPath & """ mcp_server.py"

' Execute in background (hide window)
objShell.Run command, 0, False

objLogFile.WriteLine "[" & Now() & "] Execute command: " & command
objLogFile.WriteLine "[" & Now() & "] Checking server startup completion..."

' Check server startup (wait up to 30 seconds)
serverStarted = False
checkCount = 0

Do While checkCount < 60 And Not serverStarted
    WScript.Sleep 500  ' Wait 0.5 seconds
    checkCount = checkCount + 1
    
    ' Check if Python process is running
    If IsMCPServerActuallyRunning() Then
        serverStarted = True
        objLogFile.WriteLine "[" & Now() & "] Server process confirmed (check " & checkCount & ")"
        Exit Do
    End If
    
    ' Update log every 10 seconds
    If checkCount Mod 20 = 0 Then
        objLogFile.WriteLine "[" & Now() & "] Waiting for server startup... (" & (checkCount * 0.5) & " seconds elapsed)"
    End If
Loop

objLogFile.Close

If serverStarted Then
    ' Show success notification
    MsgBox "MCP Server started successfully!" & vbCrLf & vbCrLf & _
           "Project folder: " & projectPath & vbCrLf & _
           "Python interpreter: " & pythonPath & vbCrLf & _
           "Server status: Running in background" & vbCrLf & _
           "Log file: auto_start_detailed.log" & vbCrLf & vbCrLf & _
           "Startup time: " & (checkCount * 0.5) & " seconds", _
           vbInformation, "MCP Server Auto Start - Success"
Else
    ' Show failure notification
    MsgBox "MCP Server startup failed!" & vbCrLf & vbCrLf & _
           "Project folder: " & projectPath & vbCrLf & _
           "Python interpreter: " & pythonPath & vbCrLf & _
           "Check log file: auto_start_detailed.log" & vbCrLf & vbCrLf & _
           "Please verify Python dependencies are installed.", _
           vbCritical, "MCP Server Auto Start - Failed"
End If

' ===== Function Definitions =====

Function FindPythonPath()
    Dim pythonCandidates, i
    
    ' Possible Python paths (prioritize common locations)
    pythonCandidates = Array( _
        "python", _
        "python3", _
        "py", _
        "C:\Python312\python.exe", _
        "C:\Python311\python.exe", _
        "C:\Python310\python.exe", _
        "C:\Python39\python.exe", _
        "C:\Users\" & objShell.ExpandEnvironmentStrings("%USERNAME%") & "\AppData\Local\Programs\Python\Python312\python.exe", _
        "C:\Users\" & objShell.ExpandEnvironmentStrings("%USERNAME%") & "\AppData\Local\Programs\Python\Python311\python.exe", _
        "C:\Users\" & objShell.ExpandEnvironmentStrings("%USERNAME%") & "\AppData\Local\Programs\Python\Python310\python.exe", _
        "C:\Users\" & objShell.ExpandEnvironmentStrings("%USERNAME%") & "\AppData\Local\Programs\Python\Python39\python.exe", _
        "C:\ProgramData\miniconda3\python.exe", _
        "C:\ProgramData\anaconda3\python.exe", _
        "C:\Users\" & objShell.ExpandEnvironmentStrings("%USERNAME%") & "\miniconda3\python.exe", _
        "C:\Users\" & objShell.ExpandEnvironmentStrings("%USERNAME%") & "\anaconda3\python.exe", _
        "C:\Users\" & objShell.ExpandEnvironmentStrings("%USERNAME%") & "\AppData\Local\Microsoft\WindowsApps\python.exe" _
    )
    
    For i = 0 To UBound(pythonCandidates)
        If TestPythonPath(pythonCandidates(i)) Then
            FindPythonPath = pythonCandidates(i)
            Exit Function
        End If
    Next
    
    FindPythonPath = ""
End Function

Function TestPythonPath(path)
    On Error Resume Next
    Dim result
    result = objShell.Run("cmd /c """ & path & """ --version >nul 2>&1", 0, True)
    If Err.Number = 0 And result = 0 Then
        TestPythonPath = True
    Else
        TestPythonPath = False
    End If
    On Error GoTo 0
End Function

Function IsMCPServerActuallyRunning()
    On Error Resume Next
    Dim objWMI, processes, process
    Dim cmdLine, foundActualServer
    foundActualServer = False
    
    Set objWMI = GetObject("winmgmts:\\.\root\cimv2")
    Set processes = objWMI.ExecQuery("SELECT * FROM Win32_Process WHERE Name = 'python.exe'")
    
    For Each process In processes
        If Not IsNull(process.CommandLine) And process.CommandLine <> "" Then
            cmdLine = LCase(Trim(process.CommandLine))
            
            ' Check if this is actually running mcp_server.py as the main script
            ' Not just referencing it (like mypy, pylint, etc.)
            If InStr(cmdLine, "mcp_server.py") > 0 Then
                ' Additional checks to ensure it's actually running the server
                ' and not just analyzing/checking it
                If (InStr(cmdLine, " mcp_server.py") > 0 Or _
                    InStr(cmdLine, """mcp_server.py""") > 0 Or _
                    InStr(cmdLine, "\mcp_server.py") > 0) And _
                   InStr(cmdLine, "-m mypy") = 0 And _
                   InStr(cmdLine, "-m pylint") = 0 And _
                   InStr(cmdLine, "-m black") = 0 And _
                   InStr(cmdLine, "-m flake8") = 0 And _
                   InStr(cmdLine, "lsp_server.py") = 0 And _
                   InStr(cmdLine, "--show-error-end") = 0 Then
                    ' This looks like an actual execution of mcp_server.py
                    objLogFile.WriteLine "[" & Now() & "] Found actual MCP server process: " & process.CommandLine
                    foundActualServer = True
                    Exit For
                Else
                    ' This is just a tool analyzing the file
                    objLogFile.WriteLine "[" & Now() & "] Ignoring analysis tool: " & process.CommandLine
                End If
            End If
        End If
    Next
    
    IsMCPServerActuallyRunning = foundActualServer
    On Error GoTo 0
End Function