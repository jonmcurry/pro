' Stop Professional SMART Service
' This script stops the service silently without showing console windows

Function StopService()
    On Error Resume Next

    Dim shell, wmi, processes, process, services, service
    Set shell = CreateObject("WScript.Shell")
    Set wmi = GetObject("winmgmts:\\.\root\cimv2")

    ' Kill any running pro-service.exe processes
    Set processes = wmi.ExecQuery("SELECT * FROM Win32_Process WHERE Name='pro-service.exe'")
    For Each process In processes
        process.Terminate()
    Next

    ' Stop the Windows service
    Set services = wmi.ExecQuery("SELECT * FROM Win32_Service WHERE Name='ProfessionalSMART'")
    For Each service In services
        service.StopService()
    Next

    ' Wait for service to stop
    WScript.Sleep 2000

    ' Delete the service using sc.exe (hidden)
    shell.Run "sc delete ProfessionalSMART", 0, True

    StopService = 0
End Function
