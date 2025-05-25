' Podman Auto-Startup Script
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
projectDir = fso.GetParentFolderName(WScript.ScriptFullName)
podmanPath = "C:\\Program Files\\RedHat\\Podman\\podman.exe"
logFile = projectDir & "\\podman_startup.log"

' Write to log
Set logFileHandle = fso.OpenTextFile(logFile, 8, True)
logFileHandle.WriteLine Now & " - Starting Podman"
logFileHandle.Close

' Wait for system to be ready
WScript.Sleep 30000

' Start Podman machine
shell.Run """" & podmanPath & """ machine start", 0, True

' Wait for machine to be ready
WScript.Sleep 20000

' Start containers
shell.CurrentDirectory = projectDir
shell.Run """" & podmanPath & """ compose -f """ & projectDir & "\\milvus-podman-compose.yml"" up -d", 0, True

' Write completion to log
Set logFileHandle = fso.OpenTextFile(logFile, 8, True)
logFileHandle.WriteLine Now & " - Startup completed"
logFileHandle.Close
