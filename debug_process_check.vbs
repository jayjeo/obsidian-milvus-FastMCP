Option Explicit

' Debug version to test process detection
Dim objShell, objFSO
Dim projectPath, pythonPath, scriptPath, logPath

' Get current script directory as project path
Set objFSO = CreateObject("Scripting.FileSystemObject")
projectPath = objFSO.GetParentFolderName(WScript.ScriptFullName)
scriptPath = projectPath & "\mcp_server.py"
logPath = projectPath & "\debug_process_check.log"

Set objShell = CreateObject("WScript.Shell")

' Create debug log
Dim objLogFile
Set objLogFile = objFSO.CreateTextFile(logPath, True)
objLogFile.WriteLine "[" & Now() & "] Debug Process Check Started"

' Test process detection
objLogFile.WriteLine "[" & Now() & "] Testing process detection..."

Dim result
result = IsProcessRunning("python.exe", "mcp_server.py")

objLogFile.WriteLine "[" & Now() & "] Process detection result: " & result

' List all python processes for debugging
objLogFile.WriteLine "[" & Now() & "] Listing all Python processes:"
ListAllPythonProcesses objLogFile

objLogFile.Close

MsgBox "Debug completed. Check debug_process_check.log for details." & vbCrLf & _
       "Process detection result: " & result

Function IsProcessRunning(processName, scriptName)
    On Error Resume Next
    Dim objWMI, processes, process
    Dim foundCount
    foundCount = 0
    
    Set objWMI = GetObject("winmgmts:\\.\root\cimv2")
    Set processes = objWMI.ExecQuery("SELECT * FROM Win32_Process WHERE Name = '" & processName & "'")
    
    For Each process In processes
        If Not IsNull(process.CommandLine) And process.CommandLine <> "" Then
            ' Log each command line for debugging
            objLogFile.WriteLine "[DEBUG] Process " & process.ProcessId & ": " & process.CommandLine
            
            If InStr(LCase(process.CommandLine), LCase(scriptName)) > 0 Then
                foundCount = foundCount + 1
                objLogFile.WriteLine "[DEBUG] MATCH FOUND: " & process.CommandLine
            End If
        Else
            objLogFile.WriteLine "[DEBUG] Process " & process.ProcessId & ": (null or empty command line)"
        End If
    Next
    
    objLogFile.WriteLine "[DEBUG] Total matches found: " & foundCount
    IsProcessRunning = (foundCount > 0)
    On Error GoTo 0
End Function

Sub ListAllPythonProcesses(logFile)
    On Error Resume Next
    Dim objWMI, processes, process
    
    Set objWMI = GetObject("winmgmts:\\.\root\cimv2")
    Set processes = objWMI.ExecQuery("SELECT * FROM Win32_Process WHERE Name = 'python.exe'")
    
    For Each process In processes
        If Not IsNull(process.CommandLine) And process.CommandLine <> "" Then
            logFile.WriteLine "[LIST] PID " & process.ProcessId & ": " & process.CommandLine
        Else
            logFile.WriteLine "[LIST] PID " & process.ProcessId & ": (null command line)"
        End If
    Next
    On Error GoTo 0
End Sub