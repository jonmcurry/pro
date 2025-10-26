' StopServiceAndWait.vbs
' Custom action to stop service and wait for it to fully terminate before file deletion
' This ensures pro-service.exe can be deleted without locks
' Uses only WMI - no command line tools that show windows

Option Explicit

Function StopServiceAndWait()
    On Error Resume Next

    Dim serviceName, timeout, elapsed
    Dim wmi, service, services
    Dim stopResult

    serviceName = "ProfessionalSMART"

    ' Log start
    LogMessage "StopServiceAndWait: Starting service stop process for " & serviceName

    ' Connect to WMI
    Set wmi = GetObject("winmgmts:{impersonationLevel=impersonate}!\\.\root\cimv2")
    If Err.Number <> 0 Then
        LogMessage "StopServiceAndWait: ERROR - Failed to connect to WMI: " & Err.Description
        StopServiceAndWait = 0
        Exit Function
    End If

    ' Query for the service
    Set services = wmi.ExecQuery("SELECT * FROM Win32_Service WHERE Name='" & serviceName & "'")

    If services.Count = 0 Then
        LogMessage "StopServiceAndWait: Service does not exist"
        Set wmi = Nothing
        StopServiceAndWait = 0
        Exit Function
    End If

    ' Get the service object
    For Each service In services
        LogMessage "StopServiceAndWait: Found service in state: " & service.State

        ' Only try to stop if not already stopped
        If service.State <> "Stopped" Then
            LogMessage "StopServiceAndWait: Stopping service..."
            stopResult = service.StopService()
            LogMessage "StopServiceAndWait: StopService returned: " & stopResult
        End If
    Next

    ' Wait for service to fully stop (up to 60 seconds)
    timeout = 60
    elapsed = 0

    Do While elapsed < timeout
        ' Re-query service status
        Set services = wmi.ExecQuery("SELECT * FROM Win32_Service WHERE Name='" & serviceName & "'")

        If services.Count = 0 Then
            LogMessage "StopServiceAndWait: Service no longer exists"
            Exit Do
        End If

        ' Check service state
        Dim svc, isStopped
        isStopped = True
        For Each svc In services
            If svc.State = "Stopped" Then
                LogMessage "StopServiceAndWait: Service is STOPPED in " & elapsed & " seconds"
                Set services = Nothing
                Set wmi = Nothing
                StopServiceAndWait = 0
                Exit Function
            ElseIf svc.State = "Stop Pending" Then
                LogMessage "StopServiceAndWait: Service is STOP_PENDING, waiting..."
                isStopped = False
            ElseIf svc.State = "Running" Then
                LogMessage "StopServiceAndWait: Service still RUNNING, waiting..."
                isStopped = False
            Else
                LogMessage "StopServiceAndWait: Service state is " & svc.State
                isStopped = False
            End If
        Next

        Set services = Nothing

        ' Exit if stopped
        If isStopped Then Exit Do

        ' Wait 1 second before checking again
        WScript.Sleep 1000
        elapsed = elapsed + 1
    Loop

    ' Check if we timed out
    If elapsed >= timeout Then
        LogMessage "StopServiceAndWait: WARNING - Service did not stop within " & timeout & " seconds"
        LogMessage "StopServiceAndWait: Attempting to terminate process via WMI..."

        ' Try to force kill the process using WMI (no visible windows)
        Dim processes, proc
        Set processes = wmi.ExecQuery("SELECT * FROM Win32_Process WHERE Name='pro-service.exe'")
        For Each proc In processes
            LogMessage "StopServiceAndWait: Terminating process ID " & proc.ProcessId
            proc.Terminate()
        Next
        Set processes = Nothing

        WScript.Sleep 2000

        LogMessage "StopServiceAndWait: Process termination attempted"
    Else
        LogMessage "StopServiceAndWait: Service stopped successfully"
    End If

    ' Clean up
    Set wmi = Nothing

    ' Always return success - don't block uninstall even if service won't stop
    StopServiceAndWait = 0
End Function

' Logging function
Sub LogMessage(message)
    On Error Resume Next
    Dim fso, logFile, logPath
    Set fso = CreateObject("Scripting.FileSystemObject")
    logPath = "C:\Temp\ProfessionalSMART_Uninstall.log"

    ' Create C:\Temp if it doesn't exist
    If Not fso.FolderExists("C:\Temp") Then
        fso.CreateFolder("C:\Temp")
    End If

    ' Append to log file
    Set logFile = fso.OpenTextFile(logPath, 8, True)  ' 8 = ForAppending
    logFile.WriteLine Now() & " - " & message
    logFile.Close
    Set logFile = Nothing
    Set fso = Nothing
End Sub

' Entry point
StopServiceAndWait()
